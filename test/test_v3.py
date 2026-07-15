"""test_v3.py — V3 读稿错误检测的快速测试台（默认以 man 姐为例）。

V3 = 检测 + 确认两个 Agent，对照原稿找"说错"（off_topic/resay）。
纯删除模型：增读跑题/残句重说 → 删。misread_content（内容/数字/专名读错）本期不检测。

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

from speech_error_detector.utils.sentence_io import load_sentences
from speech_error_detector.detect_agent.prompts import (
    DETECT_V3_SYSTEM_PROMPT,
    CONFIRM_V3_SYSTEM_PROMPT,
    build_detect_v3_prompt,
    build_confirm_v3_prompt,
)
from speech_error_detector.detect_agent.review_loop import (
    _run_detect,
    _run_verify,
    run_agent_review_loop_v3,
)

# 默认以 man 姐为例
BASE = "2026-07-07_man姐"
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
        print(f"       delete_sentence_idx={iss.get('delete_sentence_idx')}  resay_exists={iss.get('resay_exists')}")
        print(f"       orig_text: {iss.get('orig_text')}")
        print(f"       error_text: {iss.get('error_text')}")
        print(f"       description: {iss.get('description')}")


def run_single_round(sentences, model, do_confirm, loop_dir, original_script) -> None:
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
        system_prompt=DETECT_V3_SYSTEM_PROMPT,
        stage="detect_v3",
        build_prompt_fn=build_detect_v3_prompt,
    )
    if issues is None:
        print("  ⚠️ Detect LLM 调用失败")
        return
    # 仅保留 misread 维度
    issues = [iss for iss in issues if iss.get("dimension") == "misread"]
    if not issues:
        print("  无候选问题（未发现“说错”）")
        return
    _print_issues(issues)

    if not do_confirm:
        return
    print("\n===== V3 Round 1（确认）=====")
    verified = _run_verify(
        loop_dir=loop_dir,
        round_num=1,
        new_issues=issues,
        current_sentences=current,
        model=model,
        effective_script=original_script,
        system_prompt=CONFIRM_V3_SYSTEM_PROMPT,
        stage="confirm_v3",
        build_verify_prompt_fn=build_confirm_v3_prompt,
        single=True,
    )
    for i, v in enumerate(verified):
        mark = "✅删" if (v.get("confirmed") and v.get("action") == "delete") else \
               ("📌标注" if (v.get("confirmed") and v.get("action") == "report") else "❌驳回")
        print(f"  -> #{i} {mark}: {v.get('reason')}")


def run_full(sentences, words, model, rounds, analysis_dir, original_script) -> None:
    print(f"V3 完整循环：{len(sentences)} 句（max_rounds={rounds}）\n")
    run_agent_review_loop_v3(
        analysis_dir=analysis_dir,
        loop_dir=analysis_dir,
        sentences=[s for s in sentences if s.get("text", "").strip()],
        words=words,
        model=model,
        max_rounds=rounds,
        consecutive_empty_to_exit=2,
        use_original_script=bool(original_script),
        original_script=original_script,
        enable_thinking=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="V3 读稿错误检测测试台（默认 man 姐）")
    ap.add_argument("--sentences", default=str(DEFAULT_SENTENCES))
    ap.add_argument("--words", default=str(DEFAULT_WORDS))
    ap.add_argument("--original-script", default=None)
    ap.add_argument("--detect-only", action="store_true", help="只跑 detect 不跑 confirm")
    ap.add_argument("--full", action="store_true", help="跑完整 V3 循环")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--model", default="deepseek-v4-pro")
    args = ap.parse_args()

    sentences = load_sentences(Path(args.sentences))
    analysis_dir = TEST_DIR
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
        run_full(sentences, words, args.model, args.rounds, analysis_dir, original_script)
    else:
        run_single_round(sentences, args.model, not args.detect_only, analysis_dir, original_script)


if __name__ == "__main__":
    main()
