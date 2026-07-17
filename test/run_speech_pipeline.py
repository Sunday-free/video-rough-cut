"""run_speech_pipeline.py — 口误检测完整运行入口（从 speech_pipeline 迁出的 main + 写死参数）。

运行整条流水线（检测→研判→V3 循环审查→装配）并进入审核：
  python -m speech_error_detector.test.run_speech_pipeline

换数据改下面的 RUN_INDEX / dirs 即可。
"""

import sys
from pathlib import Path

# 让本模块可作为包内模块导入：本文件位于 speech_error_detector/test/，
# parents[2] 即工作区根目录（含 2026-07-07_福总 等数据目录）。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from speech_error_detector.speech_pipeline import run_pipeline
from speech_error_detector.server.review_entry import run_review
from speech_error_detector.ai.chat import DEFAULT_MODEL, SUPPORTED_MODELS
from speech_error_detector.utils.compare_truth import compare_with_truth


# ── 写死的默认参数 ──
# 每个元素携带目录与识别语言：dir + language
dirs = [
    {"dir": "2026-07-07_福总", "language": "zh-CN"},
    {"dir": "2026-07-07_红姐", "language": "zh-CN"},
    {"dir": "2026-07-07_man姐", "language": "zh-CN"},
]
RUN_INDEX = 2   # 当前要跑的目录索引（换数据改这一行；指向 dirs 中对应元素，含其 language）
SILENCE_THRESH = 0.9
# 删除静音时保留的句间呼吸感时长（秒）。
SILENCE_KEEP_DURATION = 0.5
DETECT_AGENT_MODEL = DEFAULT_MODEL                       # 循环审查 Agent (detect_agent) 默认模型
DETECT_REPEAT_MODEL = SUPPORTED_MODELS.QWEN_3_7_PLUS     # 机械检测 (detect_repeat) 用 DeepSeek pro
# MODEL = "deepseek-chat"
VIDEO_DURATION = 0.0
ENABLE_DEEPSEEK_THINKING = False    # 关闭 DeepSeek 思考模式（加速）
SPLIT_MODE = "hybrid"         # 按静音 gap 切分
USE_ORIGINAL_SCRIPT = True     # 启用原稿对照
SKIP_JUDGE = False
SKIP_LOOP = False
SKIP_REVIEW = False
MAX_DET_ROUNDS = 5
MAX_LOOP_ROUNDS = 3
REVIEW_SERVE = True
COMPARE_WITH_TRUTH = True     # pipeline 跑完后与原始口播 sentences.txt 比对误删/漏删


def main() -> None:
    # 参数全部写死，改上方常量即可。
    # dirs[RUN_INDEX] 指向要跑的目录（含各自 language）；换数据改 RUN_INDEX。
    # dir 相对名（如 2026-07-07_福总）一律相对 ROOT 解析；
    # 已是绝对路径则原样保留（Path 除法遇绝对路径返回绝对路径）。
    entry = dirs[RUN_INDEX]
    _dir = entry["dir"]
    _lang = entry.get("language", "zh-CN")
    _base = Path(_dir)
    if not _base.is_absolute():
        _base = ROOT / _base
    base_dir = str(_base)

    # 1) 跑完整 pipeline 检测（循环审查固定使用 V3）
    run_pipeline(
        base_dir=base_dir,
        skip_judge=SKIP_JUDGE,
        skip_loop=SKIP_LOOP,
        silence_thresh=SILENCE_THRESH,
        detect_repeat_model=DETECT_REPEAT_MODEL,
        detect_agent_model=DETECT_AGENT_MODEL,
        video_duration=VIDEO_DURATION,
        enable_deepseek_thinking=ENABLE_DEEPSEEK_THINKING,
        split_mode=SPLIT_MODE,
        use_original_script=USE_ORIGINAL_SCRIPT,
        max_det_rounds=MAX_DET_ROUNDS,
        max_loop_rounds=MAX_LOOP_ROUNDS,
        language=_lang,
    )

    # 2.5) 与真值（原始口播 sentences.txt）比对，统计误删/漏删
    #      final = 2_分析/sentences.txt（检测修正后）；truth = 顶层 sentences.txt（原始口播）
    if COMPARE_WITH_TRUTH:
        final_path = _base / "2_分析" / "sentences.txt"
        truth_path = _base / "sentences.txt"
        compare_with_truth(final_path, truth_path)

    # 2) review 默认开启：检测完直接进入审核（autoselect 静音预选 + 0.1s 呼吸感）
    if not SKIP_REVIEW:
        run_review(
            base_dir=base_dir,
            silence_gap_threshold=SILENCE_THRESH,
            silence_keep_duration=SILENCE_KEEP_DURATION,
            serve=REVIEW_SERVE,
        )


if __name__ == "__main__":
    main()
