"""test_detect.py — Detect/Verify 快速测试台，用于迭代优化 prompt。

两种模式：
  1) 单轮 detect + verify（默认，~30s，快）：不真正应用删除，只看模型这一轮
     找出了什么、verify 确认/驳回了什么。适合快速迭代 prompt。
  2) --full 完整循环（慢，忠实复刻 pipeline）：加载 words 后跑完整
     run_agent_review_loop，输出与真实运行完全一致，适合做对照基线。

用法：
  # 在福总清洗后的文稿上跑 1 轮 detect+verify
  python -m speech_error_detector.loop.test_detect

  # 跑 3 轮（注意：单轮模式不应用删除，所以多轮会重复同一文本，仅用于看稳定性）
  python -m speech_error_detector.loop.test_detect --rounds 3

  # 只跑 detect 不看 verify（最快，~12s）
  python -m speech_error_detector.loop.test_detect --detect-only

  # 换一篇文稿（验证泛化能力）
  python -m speech_error_detector.loop.test_detect \
      --sentences ../2026-07-07_福总/剪口播/2_分析/sentences.txt \
      --words   ../2026-07-07_福总/剪口播/1_转录/subtitles_words.json

  # 完整循环（忠实复刻真实运行）
  python -m speech_error_detector.loop.test_detect --full

可用环境变量：DEEPSEEK_API_KEY（已在 .env 中）、DEEPSEEK_MODEL。
"""

import argparse
import json
import sys
from pathlib import Path

# 让本模块可作为包内模块导入
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from speech_error_detector.base.sentence_io import load_sentences
from speech_error_detector.loop.agent_review_loop import (
    _run_detect,
    _run_verify,
    run_agent_review_loop,
)
from speech_error_detector.detect.rule_corroborate import rule_corroborate

# 默认用福总清洗后的文稿做快速测试
# BASE = "2026-07-07_福总"
BASE = "2026-07-07_红姐"
DEFAULT_SENTENCES = ROOT / BASE / "剪口播/2_分析/sentences.txt"
DEFAULT_WORDS = ROOT / BASE / "剪口播/1_转录/subtitles_words.json"
TEST_DIR = ROOT / "speech_error_detector/loop/test_output"


def _derive_original_script(sentences_path: Path) -> Path | None:
    """由原稿 sentences.txt 路径推断 original_script.txt 位置：
    .../2_分析/sentences.txt -> .../1_转录/original_script.txt
    """
    p = sentences_path
    if p.parent.name == "2_分析":
        cand = p.parent.parent / "1_转录" / "original_script.txt"
        if cand.exists():
            return cand
    for cand in (p.parent / "original_script.txt", p.parent.parent / "original_script.txt"):
        if cand.exists():
            return cand
    return None


def _print_issues(issues: list[dict]) -> None:
    for i, iss in enumerate(issues):
        sid = iss.get("sentence_idx")
        dim = iss.get("dimension")
        sev = iss.get("severity")
        dt = iss.get("delete_text")
        co = iss.get("char_offset")
        print(f"  [{sev}] #{i} {dim}: 句{sid}")
        print(f"       delete_text={dt!r}  char_offset={co}  delete_sentence_idx={iss.get('delete_sentence_idx')}")
        print(f"       error_text: {iss.get('error_text')}")
        print(f"       description: {iss.get('description')}")


def run_single_rounds(sentences, model, rounds, do_verify, loop_dir, enable_rule_filter,
                      original_script="", analysis_dir=None) -> None:
    current = [s for s in sentences if s.get("text", "").strip()]
    print(f"加载 {len(current)} 句非空句子"
          + (f"（已注入原稿 {len(original_script)} 字）" if original_script else "")
          + "\n")
    for r in range(1, rounds + 1):
        print(f"===== Round {r}/{rounds} =====")
        issues = _run_detect(
            loop_dir=loop_dir,
            round_num=r,
            current_sentences=current,
            model=model,
            effective_script=original_script,
            all_decisions=[],
        )
        if issues is None:
            print("  ⚠️ Detect LLM 调用失败，退出")
            return
        if not issues:
            print("  无候选问题")
            continue
        _print_issues(issues)
        # 规则兜底过滤展示（与真实循环一致；可通过 --no-rule-filter 关闭对照）
        if enable_rule_filter:
            kept, rule_rejected = rule_corroborate(issues, current)
            for iss, reason in rule_rejected:
                print(f"  🛡️ 规则过滤 句{iss.get('sentence_idx')} {iss.get('dimension')}: {reason}")
            if rule_rejected:
                print(f"   🛡️ 规则兜底过滤 {len(rule_rejected)} 个疑似编造的重复（不进 verify）")
            issues = kept
            if not issues:
                print("   （无残留候选，跳过 verify）\n")
                continue
        if not do_verify:
            continue
        print()
        verified = _run_verify(
            analysis_dir=analysis_dir,
            round_num=r,
            new_issues=issues,
            current_sentences=current,
            model=model,
            effective_script=original_script,
        )
        for i, v in enumerate(verified):
            mark = "✅ 确认" if v.get("confirmed") else "❌ 驳回"
            print(f"  -> #{i} {mark}: {v.get('reason')}")


def run_full_loop(sentences, words, model, rounds, analysis_dir, enable_rule_filter,
                  original_script="") -> None:
    print(f"完整循环：{len(sentences)} 句, {len(words)} 词  (规则兜底={enable_rule_filter})\n")
    run_agent_review_loop(
        analysis_dir=analysis_dir,
        sentences=[s for s in sentences if s.get("text", "").strip()],
        words=words,
        model=model,
        max_rounds=rounds,
        consecutive_empty_to_exit=2,
        use_original_script=bool(original_script),
        original_script=original_script,
        enable_thinking=False,
        enable_rule_filter=enable_rule_filter,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect/Verify 快速测试台")
    ap.add_argument("--sentences", default=str(DEFAULT_SENTENCES), help="sentences.txt 路径")
    ap.add_argument("--words", default=str(DEFAULT_WORDS), help="subtitles_words.json 路径（仅 --full 用）")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--detect-only", action="store_true", help="只跑 detect 不跑 verify")
    ap.add_argument("--full", action="store_true", help="跑完整 run_agent_review_loop")
    ap.add_argument("--no-rule-filter", action="store_true",
                    help="关闭规则兜底过滤（对照用，看没有兜底时的误报）")
    ap.add_argument("--analysis-dir", default=None, help="prompt/日志落盘目录（默认 test_output）")
    ap.add_argument("--original-script", default=None,
                    help="原稿路径（默认由 sentences 路径自动推断 .../1_转录/original_script.txt）")
    ap.add_argument("--model", default="deepseek-v4-pro",
                    help="指定模型（默认 deepseek-v4-pro = pro；端点退化时可改用 deepseek-chat）")
    args = ap.parse_args()

    sentences = load_sentences(Path(args.sentences))
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else TEST_DIR
    analysis_dir.mkdir(parents=True, exist_ok=True)
    enable_rule_filter = not args.no_rule_filter

    # 加载原稿（use_original_script）
    orig_path = Path(args.original_script) if args.original_script else _derive_original_script(Path(args.sentences))
    original_script = ""
    if orig_path and orig_path.exists():
        original_script = orig_path.read_text(encoding="utf-8").strip()
        print(f"📜 加载原稿: {orig_path} ({len(original_script)} 字)\n")
    else:
        print(f"⚠️ 未找到原稿（{orig_path}），按无原稿模式运行\n")

    if args.full:
        words = json.loads(Path(args.words).read_text(encoding="utf-8"))
        run_full_loop(sentences, words, args.model, args.rounds, analysis_dir,
                      enable_rule_filter, original_script=original_script)
    else:
        run_single_rounds(
            sentences=sentences,
            model=args.model,
            rounds=args.rounds,
            do_verify=not args.detect_only,
            loop_dir=analysis_dir,
            enable_rule_filter=enable_rule_filter,
            original_script=original_script,
            analysis_dir=analysis_dir,
        )


if __name__ == "__main__":
    main()
