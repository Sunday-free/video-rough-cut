"""test_v3.py — V3 读稿错误检测的快速测试台（默认以 man 姐为例）。

V3 = 检测 + 确认两个 Agent，对照原稿找"说错"（仅 resay 残句重说）。
纯删除模型：残句重说（off_topic 增读跑题已停用，不再检测）。

用法：
  # 单轮 detect + confirm（~30s，快，只看这一轮找出了什么、确认/驳回了什么）
  python -m speech_error_detector.test.test_v3

  # 换一篇文稿（验证泛化）
  python -m speech_error_detector.test.test_v3 \
      --sentences ../2026-07-07_红姐/2_分析/sentences.txt \
      --original-script ../2026-07-07_红姐/original_script.txt

  # 完整 V3 循环（忠实复刻，落 review_loop_decisions.json；仅标注项随主文件，action=="report"）
  python -m speech_error_detector.test.test_v3 --full

  # 指定模型（端点退化时）
  python -m speech_error_detector.test.test_v3 --model deepseek-chat
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from speech_error_detector.ai.chat import DEFAULT_MODEL

from speech_error_detector.utils.sentence_io import load_sentences, write_sentences
from speech_error_detector.utils.compare_truth import compare_with_truth
from speech_error_detector.detect_agent.review_loop import (
    _run_detect,
    _run_verify,
    run_agent_review_loop_v3,
)
from speech_error_detector.detect_agent.agent_apply import apply_deletions_to_sentences
from speech_error_detector.assemble.assemble import run_assemble, generate_report_markdown

# 默认以 man 姐为例
dirs = [
    {"dir": "2026-07-07_福总", "language": "zh-CN"},
    {"dir": "2026-07-07_红姐", "language": "zh-CN"},
    {"dir": "2026-07-07_man姐", "language": "zh-CN"},
]
RUN_INDEX = 1   # 当前要跑的目录索引（换数据改这一行；指向 dirs 中对应元素，含其 language）
BASE = dirs[RUN_INDEX]["dir"]
DEFAULT_SENTENCES = ROOT / BASE / "2_分析/sentences_origin.txt"
DEFAULT_WORDS = ROOT / BASE / "1_转录/subtitles_words.json"
DEFAULT_SCRIPT = ROOT / BASE / "original_script.txt"
TEST_DIR = ROOT / "speech_error_detector/test/test_output_review_v3"


def _derive_original_script(sentences_path: Path) -> Path | None:
    p = sentences_path
    if p.parent.name == "2_分析":
        cand = p.parent.parent / "original_script.txt"
        if cand.exists():
            return cand
    for cand in (p.parent / "original_script.txt", p.parent.parent / "original_script.txt"):
        if cand.exists():
            return cand
    return None


def _print_issues(issues: list[dict]) -> None:
    for i, iss in enumerate(issues):
        sid = iss.get("sentence_idx")
        sub = iss.get("subtype", iss.get("dimension"))
        sev = iss.get("severity")
        print(f"  [{sev}] #{i} {sub}: 句{sid}")
        print(f"       error_text: {iss.get('error_text')}")


def run_single_round(sentences, model, do_confirm, loop_dir, original_script, words_path=None, truth_path=None) -> None:
    current = [s for s in sentences if s.get("text", "").strip()]
    print(f"加载 {len(current)} 句非空句子"
          + (f"（已注入原稿 {len(original_script)} 字）" if original_script else "") + "\n")
    print("===== V3 Round 1（检测）=====")
    issues = _run_detect(
        loop_dir=loop_dir,
        round_num=1,
        current_sentences=current,
        model=model,
        effective_script=original_script,
        all_decisions=[],
        stage="detect_v3",
    )
    if issues is None:
        print("  ⚠️ Detect LLM 调用失败")
        return
    # 仅保留 misread 维度
    issues = [iss for iss in issues if iss.get("dimension") == "misread"]
    if not issues:
        print("  无候选问题（未发现“说错”）")
        return

    if not do_confirm:
        return

    _print_issues(issues)
    
    print("\n===== V3 Round 1（确认）=====")
    verified = _run_verify(
        loop_dir=loop_dir,
        round_num=1,
        new_issues=issues,
        current_sentences=current,
        model=model,
        effective_script=original_script,
        stage="confirm_v3",
    )
    for i, v in enumerate(verified):
        mark = "✅删" if v.get("action") == "delete" else "❌保留"
        print(f"  -> #{i} {mark}: {v.get('reason')}")

    # 应用删除，保存最终 sentences
    # ⚠️ 必须用 apply_deletions_to_sentences：它按 confirm 给出的 delete_text 做【片段删除】，
    #    而不是按 idx 整句删。否则即便 delete_text 精确到片段，整句仍会被删（误删根因）。
    words = []
    if words_path and Path(words_path).exists():
        words = json.loads(Path(words_path).read_text(encoding="utf-8"))
    confirmed_issues = []
    for i, iss in enumerate(issues):
        v = verified[i]
        if v.get("action") == "delete" and (v.get("delete_text") or "").strip():
            aug = dict(iss)
            aug["delete_text"] = v["delete_text"].strip()
            confirmed_issues.append(aug)
    applied, applied_records = apply_deletions_to_sentences(
        sentences=current, words=words, confirmed_issues=confirmed_issues,
    )
    # 逐条精确打印实际删除内容
    for iss, rec in zip(confirmed_issues, applied_records):
        sid = rec.get("sid")
        mode = rec.get("mode")
        dt = rec.get("deleted_text", "")
        tag = {"full": "整句", "partial": "片段", None: "未命中"}.get(mode, "未命中")
        print(f"      🗑 句{sid} [{tag}删除] 实际删除: 「{dt}」")
    final_path = loop_dir / "sentences_final.txt"
    write_sentences(final_path, applied, original=current)
    print(f"\n📄 删除后句子已保存: {final_path}")

    # 写出 review_loop_decisions.json 供口误分析报告使用
    # 使用 _make_decision_obj 确保字段名与 _build_loop_section 兼容
    from speech_error_detector.detect_agent.review_loop import _make_decision_obj
    loop_decisions = []
    for i, iss in enumerate(issues):
        v = verified[i] if i < len(verified) else {"index": i, "confirmed": False, "action": "delete", "reason": "无验证结果"}
        loop_decisions.append(_make_decision_obj(iss, v, 1, current))
    dec_path = loop_dir / "review_loop_decisions.json"
    dec_path.write_text(json.dumps(loop_decisions, ensure_ascii=False, indent=2), encoding="utf-8")

    # 生成口误分析报告
    _write_misread_report(loop_dir, words_path, applied, original=current)

    # 跑完自动比对（full/single 通用）
    if truth_path is not None:
        compare_with_truth(final_path, truth_path)



def _write_misread_report(
    analysis_dir: Path,
    words_path: Path | None,
    applied: list[dict],
    *,
    original: list[dict],
) -> None:
    """生成口误分析.md 报告（调用 assemble 模块）。"""
    if words_path is None or not Path(words_path).exists():
        print("⚠️ 未提供或未找到 words JSON，跳过口误分析报告")
        return
    print("\n📊 生成口误分析报告…")
    auto_path, stats = run_assemble(
        analysis_dir=analysis_dir,
        words_json_path=Path(words_path),
        sentences=applied,
        original_sentences=original,
        silence_thresh=0.3,
        video_duration=0.0,
    )
    report_md = generate_report_markdown(analysis_dir, stats, silence_thresh=0.3)
    report_path = analysis_dir / "口误分析.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"📄 口误分析报告已保存: {report_path}")


def run_full(sentences, words, model, rounds, analysis_dir, original_script, words_path=None, truth_path=None) -> None:
    print(f"V3 完整循环：{len(sentences)} 句（max_rounds={rounds}）\n")
    pre_filter = [s for s in sentences if s.get("text", "").strip()]
    output_path, all_decisions, current_sentences = run_agent_review_loop_v3(
        loop_dir=analysis_dir,
        sentences=pre_filter,
        words=words,
        model=model,
        max_rounds=rounds,
        consecutive_empty_to_exit=1,
        use_original_script=bool(original_script),
        original_script=original_script,
        enable_thinking=False,
    )
    # 保存最终 sentences
    final_path = analysis_dir / "sentences_final.txt"
    write_sentences(final_path, current_sentences, original=pre_filter)
    print(f"\n📄 最终句子已保存: {final_path}")

    # 生成口误分析报告
    _write_misread_report(analysis_dir, words_path, current_sentences, original=pre_filter)

    # 跑完自动比对（full/single 通用）
    if truth_path is not None:
        compare_with_truth(final_path, truth_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="V3 读稿错误检测测试台（默认 man 姐）")
    ap.add_argument("--run-index", type=int, default=None,
                    help="目录索引（覆盖文件顶部 RUN_INDEX）；指向 dirs 中对应元素")
    ap.add_argument("--sentences", default=None)
    ap.add_argument("--words", default=None)
    ap.add_argument("--original-script", default=None)
    ap.add_argument("--detect-only", action="store_true", help="只跑 detect 不跑 confirm")
    ap.add_argument("--full", action="store_true", help="跑完整 V3 循环")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    # 按 run-index 选择数据目录，输出隔离到独立子目录（避免三份结果互相覆盖）
    idx = args.run_index if args.run_index is not None else RUN_INDEX
    base = dirs[idx]["dir"]
    sentences_default = ROOT / base / "2_分析/sentences_origin.txt"
    words_default = ROOT / base / "1_转录/subtitles_words.json"
    script_default = ROOT / base / "original_script.txt"
    analysis_dir = ROOT / "speech_error_detector/test" / f"test_output_review_v3_{idx}_{base}"
    truth_path = ROOT / base / "sentences.txt"

    args.sentences = args.sentences or str(sentences_default)
    args.words = args.words or str(words_default)
    args._base = base

    sentences = load_sentences(Path(args.sentences))
    analysis_dir.mkdir(parents=True, exist_ok=True)

    orig_path = Path(args.original_script) if args.original_script else _derive_original_script(Path(args.sentences))
    original_script = ""
    if orig_path and orig_path.exists():
        original_script = orig_path.read_text(encoding="utf-8").strip()
        print(f"📜 加载原稿: {orig_path} ({len(original_script)} 字)\n")
    else:
        print("⚠️ 未找到原稿，按无原稿模式运行（V3 将几乎不报，因为靠原稿对照）\n")

    if args.full:
        words = json.loads(Path(args.words).read_text(encoding="utf-8"))
        run_full(sentences, words, args.model, args.rounds, analysis_dir, original_script, words_path=args.words, truth_path=truth_path)
    else:
        run_single_round(sentences, args.model, not args.detect_only, analysis_dir, original_script, words_path=args.words, truth_path=truth_path)


if __name__ == "__main__":
    main()
