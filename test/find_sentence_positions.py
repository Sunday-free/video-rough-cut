"""find_sentence_positions.py — 用 load_sentences 读取原始 sentences，
为每一句调用 get_org_script_window 生成「以该句为中心」的原稿窗口，输出到文件。

运行:
    python find_sentence_positions.py
（不向控制台打印任何内容）

输出格式（每句一个块）:
    句序号：80
    内容：原句子内容
    原稿窗口：原稿对照啊
"""

import os
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speech_error_detector.utils.sentence_io import load_sentences
from speech_error_detector.detect_repeat.script_window import get_org_script_window

base = "2026-07-07_man姐"
_SENT_REL = f"{base}/2_分析/sentences_origin.txt"
_SCRIPT_REL = f"{base}/original_script.txt"
_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
_OUT_NAME = f"sentence_org_windows_{base}.txt"


def _resolve(rel: str) -> str:
    for base in (os.getcwd(), _ROOT):
        p = os.path.join(base, rel)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(rel)


def main() -> None:
    sent_path = _resolve(_SENT_REL)
    script_path = _resolve(_SCRIPT_REL)
    out_dir = _OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, _OUT_NAME)

    with open(script_path, encoding="utf-8") as f:
        script = f.read()
    sentences = load_sentences(Path(sent_path))

    blocks: list[str] = []
    for s in sentences:
        idx = s["idx"]
        text = (s.get("text", "") or "").strip()
        finding = {"sent_idx": idx, "text": text}
        window = get_org_script_window(
            script, sentences, finding, focus_idx=idx
        )
        body = window if window else "（未定位到原稿）"
        blocks.append(f"句序号：{idx}\n内容：{text}\n原稿窗口：{body}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))


if __name__ == "__main__":
    main()
