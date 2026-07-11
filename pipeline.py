"""
pipeline.py — 口误检测主入口（五层架构）

执行流程:
  步骤0: gen_texts()     → readable.txt + sentences.txt
  步骤1-4: detect_loop    → 机械检测 + 研判 + 应用（多轮收敛）
  步骤5-7: Agent 循环审查 → review_loop_decisions.json
  步骤8:   assemble()     → auto_selected.json（合并 Judge + 循环审查 + 静音/语气词）
  步骤9:   字幕生成         → 口误标注字幕.txt
  步骤10:  修正后句子       → updated_sentences.txt

用法:
  python -m speech_error_detector.pipeline --base-dir ./数据目录 [--loop-version v1|v2]
  python -m speech_error_detector.pipeline review --base-dir ./数据目录 [--port 8899] [--no-autoselect] [--silence-keep-duration 0.1]

--loop-version: 循环审查模式，默认 v1
  v1 = 单检测 + 单验证（向后兼容）
  v2 = 按错误类型拆 6 个专职 Agent（inter/intra/fragment 三分，专职精简 prompt 降跨类误判）
"""

import json
import shutil
import sys
import time
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from speech_error_detector.detect.detect_loop import run_detect_judge_loop
from speech_error_detector.assemble.assemble import run_assemble, generate_report_markdown, generate_updated_sentences
from speech_error_detector.loop.agent_review_loop import run_review_loop
from speech_error_detector.base.sentence_io import load_sentences, write_sentences
from speech_error_detector.base.paths import loop_dir, make_subdirs
from speech_error_detector.assemble.annotated_subtitle import generate_annotated_subtitle
from speech_error_detector.assemble.subtitle_generator import (
    generate_readable_txt,
    generate_sentences_txt,
)
from speech_error_detector.server.review_entry import run_review



# ============================================================
#  步骤0: gen_texts
# ============================================================

def gen_texts(base_dir: Path, split_mode: str = "silence") -> tuple[Path, Path]:
    """由 subtitles_words.json 生成 readable.txt 和 sentences.txt（使用 subtitle_generator 模块，含序号合并）"""
    words_json = base_dir / "剪口播" / "1_转录" / "subtitles_words.json"
    volcengine_result = base_dir / "剪口播" / "1_转录" / "volcengine_result.json"
    analysis_dir = base_dir / "剪口播" / "2_分析"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # 调用 subtitle_generator 的函数（含序号合并后处理）
    readable_path = generate_readable_txt(words_json, analysis_dir)
    sentences_path, _ = generate_sentences_txt(
        words_json,
        analysis_dir,
        split_mode=split_mode,
        volcengine_result=volcengine_result if split_mode in ("utterance", "hybrid") else None,
    )
    
    return readable_path, sentences_path


# ============================================================
#  主流程辅助函数
# ============================================================

def _prepare_analysis_dir(base_dir: Path, skip_loop: bool = False) -> Path:
    """确保 2_分析 目录及子目录存在。不再删除整个目录（保留 detect 产物供 skip_judge 复用）；
    仅在没有 skip_loop 时清理 loop 子目录，避免旧循环审查文件干扰新结果。"""
    analysis_dir = base_dir / "剪口播" / "2_分析"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    make_subdirs(analysis_dir)
    # 非 skip_loop → 清空 loop 子目录（旧循环审查文件会干扰新结果）
    if not skip_loop:
        _loop = loop_dir(analysis_dir)
        for _f in _loop.iterdir():
            if _f.is_file():
                _f.unlink()
            elif _f.is_dir():
                shutil.rmtree(_f)
    return analysis_dir


def _backup_silence_sentences(sentences_path: Path, transcript_dir: Path) -> None:
    """备份原始 silence 切分的 sentences.txt 到 1_转录（供检查）。"""
    sil_backup = transcript_dir / "sentences_from_silence.txt"
    shutil.copy2(sentences_path, sil_backup)
    print(f"   已备份: {sil_backup.name} → 1_转录/")


def _load_original_script(original_script_path: Path, analysis_dir: Path) -> str:
    """读取原稿（不存在则返回空串），并拷贝一份到 2_分析 便于检查。"""
    if not original_script_path.exists():
        return ""
    original_script = original_script_path.read_text(encoding="utf-8")
    try:
        shutil.copy2(original_script_path, analysis_dir / "original_script.txt")
    except Exception:
        pass
    return original_script


def _run_review_loop_phase(
    *,
    analysis_dir: Path,
    words_data: list,
    model: str,
    use_original_script: bool,
    original_script: str,
    cleaned: list[dict],
    loop_version: str,
    enable_deepseek_thinking: bool,
    sentences_path: Path,
    original_sentences: list[dict],
    skip_loop: bool,
) -> tuple[list[dict], list[dict]]:
    """步骤5-7: Agent 循环审查。返回 (最终句子列表, 循环审查决策列表)。"""
    if skip_loop:
        print("-" * 40)
        print("[步骤5-7] 跳过 Agent 循环审查（不运行 loop agent，也不加载缓存）")
        print("-" * 40)
        print(f"   loop 补充删除: 0 项")
        return cleaned, []

    print("-" * 40)
    print("[步骤5-7] Agent 循环审查")
    print("-" * 40)

    # LLM 循环检测（复用 Judge 清洗后的 cleaned 引用，不再回读磁盘）
    # loop_version 控制审查模式：v1=单检测单验证；v2=按类型拆分的专职 Agent 三分。
    _loop_path, loop_decisions, current_sentences = run_review_loop(
        mode=loop_version,
        analysis_dir=analysis_dir,
        loop_dir=loop_dir(analysis_dir),
        words=words_data,
        model=model,
        use_original_script=use_original_script,
        original_script=original_script,
        sentences=cleaned,
        max_rounds=10,
        consecutive_empty_to_exit=2,
        enable_thinking=enable_deepseek_thinking,
        enable_rule_filter=True,
    )

    # 将循环结束后的最终句子列表写回 sentences.txt，使磁盘与内存引用一致
    # （下游 generate_updated_sentences / annotated_subtitle 等以此为准）
    # 传入 original_sentences：被整句删除的 idx 断档处用其 range 补全空行
    write_sentences(sentences_path, current_sentences, original=original_sentences)
    print(f"   已回写: {sentences_path.name} (循环审查后, {len(current_sentences)} 句)")
    return current_sentences, loop_decisions


def _run_assemble_and_report(
    *,
    analysis_dir: Path,
    words_json: Path,
    silence_thresh: float,
    video_duration: float,
    current_sentences: list[dict],
    original_sentences: list[dict],
) -> tuple[Path, dict]:
    """步骤8: 装配 auto_selected.json（Judge + 循环审查）+ 生成口误分析.md 报告。"""
    print("-" * 40)
    print(f"[步骤8] 装配 auto_selected.json（含 Judge + 循环审查）")
    print("-" * 40)

    auto_path, stats = run_assemble(
        analysis_dir=analysis_dir,
        words_json_path=words_json,
        silence_thresh=silence_thresh,
        video_duration=video_duration,
        sentences=current_sentences,
        original_sentences=original_sentences,
    )

    report_md = generate_report_markdown(analysis_dir, stats, silence_thresh)
    report_path = analysis_dir / "口误分析.md"
    report_path.write_text(report_md, encoding="utf-8")

    print(f"\n  auto_selected.json: {stats['total']} 项待删除索引")
    print(f"  口误分析.md: 已保存")
    print()
    return auto_path, stats


def _run_subtitle_generation(
    analysis_dir: Path, words_json: Path, current_sentences: list[dict],
) -> None:
    """步骤9: 生成口误标注字幕（含循环审查信息）。"""
    print("-" * 40)
    print(f"[步骤9] 生成口误标注字幕（含循环审查信息）")
    print("-" * 40)

    annotated_text = generate_annotated_subtitle(analysis_dir, words_json, sentences=current_sentences)
    subtitle_path = analysis_dir / "口误标注字幕.txt"
    subtitle_path.write_text(annotated_text, encoding="utf-8")
    print(f"  口误标注字幕.txt: 已保存")
    print()


def _generate_updated_sentences_phase(
    analysis_dir: Path, words_json: Path,
    current_sentences: list[dict], original_sentences: list[dict],
) -> None:
    """步骤10: 生成修正后句子列表 updated_sentences.txt。"""
    print("-" * 40)
    print("[步骤10] 生成修正后句子列表（updated_sentences.txt）")
    print("-" * 40)

    generate_updated_sentences(
        analysis_dir, words_json, sentences=current_sentences, original=original_sentences,
    )
    print(f"  updated_sentences.txt: 已保存")
    print()


# ============================================================
#  run_pipeline — 主编排器
# ============================================================

def run_pipeline(
    base_dir: str | Path,
    skip_judge: bool = False,
    skip_loop: bool = False,
    silence_thresh: float = 0.3,
    model: str = "deepseek-v4-pro",
    video_duration: float = 0.0,
    enable_deepseek_thinking: bool = False,
    split_mode: str = "silence",
    use_original_script: bool = False,
    loop_version: str = "v1",
    max_det_rounds: int = 3,
) -> dict:
    """
    运行完整的口误检测流水线（编排各步骤子函数）。

    步骤0:  gen_texts()          → readable.txt + sentences.txt
    步骤1-4: run_detect_judge_loop()  → 机械检测 + 研判 + 应用（多轮收敛）
    步骤5-7: _run_review_loop_phase()  → Agent 循环审查
    步骤8:   _run_assemble_and_report() → auto_selected.json + 报告
    步骤9:   _run_subtitle_generation() → 口误标注字幕.txt
    步骤10:  _generate_updated_sentences_phase() → updated_sentences.txt

    Args:
        base_dir: 数据根目录 (包含 1_转录/, 剪口播/)
        skip_judge: 跳过 LLM 研判 Layer 2（使用磁盘已有的 decisions_*.json）
        skip_loop: 跳过 Agent 循环审查（步骤5-7）
        silence_thresh: 静音删除阈值(秒)
        model: LLM 模型名
        video_duration: 视频时长(秒)，用于结尾补尾
        enable_deepseek_thinking: 开启 DeepSeek 思考模式

    Returns:
        汇总统计字典
    """
    base_dir = Path(base_dir)
    analysis_dir = _prepare_analysis_dir(base_dir, skip_loop=skip_loop)

    words_json = base_dir / "剪口播" / "1_转录" / "subtitles_words.json"
    # 原稿位于 1_转录（2_分析 每次运行会被清空重建，不能放那）
    original_script_path = base_dir / "剪口播" / "1_转录" / "original_script.txt"
    transcript_dir = base_dir / "剪口播" / "1_转录"

    t_start = time.time()

    print("=" * 60)
    print("  口误检测系统 (Judge + Agent 循环审查)")
    print("=" * 60)
    print(f"  数据目录: {base_dir}")
    print(f"  分析目录: {analysis_dir}")
    print()

    # ========== 步骤0: 生成可读稿 ==========
    print("-" * 40)
    print("[步骤0] 生成 readable.txt + sentences.txt")
    print("-" * 40)
    readable_path, sentences_path = gen_texts(base_dir, split_mode=split_mode)
    _backup_silence_sentences(sentences_path, transcript_dir)

    # ========== 加载基础数据 ==========
    original_script = _load_original_script(original_script_path, analysis_dir)
    sentences = load_sentences(sentences_path)

    print(f"\n  原稿长度: {len(original_script)} 字" if original_script else "  (无原稿)")
    print(f"  口播句子数: {len(sentences)}")
    print()

    # ========== 步骤1-4: 机械检测 + 研判 + 应用 循环 ==========
    # original_sentences：投影前原始全量句子（供 assemble 做 diff 定位整句删除）
    original_sentences = sentences
    with open(words_json, encoding="utf-8") as f:
        words_data = json.load(f)

    cleaned, total_candidates, det_round = run_detect_judge_loop(
        sentences=original_sentences,
        analysis_dir=analysis_dir,
        words_data=words_data,
        sentences_path=sentences_path,
        model=model,
        enable_deepseek_thinking=enable_deepseek_thinking,
        skip_judge=skip_judge,
        max_det_rounds=max_det_rounds,
        original_script=original_script,
    )

    # ========== 步骤5-7: Agent 循环审查 ==========
    current_sentences, loop_decisions = _run_review_loop_phase(
        analysis_dir=analysis_dir,
        words_data=words_data,
        model=model,
        use_original_script=use_original_script,
        original_script=original_script,
        cleaned=cleaned,
        loop_version=loop_version,
        enable_deepseek_thinking=enable_deepseek_thinking,
        sentences_path=sentences_path,
        original_sentences=original_sentences,
        skip_loop=skip_loop,
    )

    # ========== 步骤8: 装配 + 报告 ==========
    _, stats = _run_assemble_and_report(
        analysis_dir=analysis_dir,
        words_json=words_json,
        silence_thresh=silence_thresh,
        video_duration=video_duration,
        current_sentences=current_sentences,
        original_sentences=original_sentences,
    )

    # 循环审查补充的删除数
    _loop_applied = sum(
        1 for d in loop_decisions
        if d.get("decision", {}).get("confirmed") or d.get("confirmed") is True
    )
    if _loop_applied:
        print(f"  （含循环审查补充删除: {_loop_applied} 项）")

    # ========== 步骤9: 标注字幕 ==========
    # _run_subtitle_generation(analysis_dir, words_json, current_sentences)

    # ========== 步骤10: 修正后句子列表 ==========
    _generate_updated_sentences_phase(
        analysis_dir, words_json, current_sentences, original_sentences,
    )

    # ========== 完成 ==========
    elapsed = time.time() - t_start

    print("=" * 60)
    print("  检测完成！")
    print("=" * 60)
    print(f"  耗时: {elapsed:.1f}s")
    print(f"  候选异常: {total_candidates}")
    print(f"  最终删除: {stats['total']} 个 word 索引")
    print()
    print(f"  输出文件:")
    print(f"    {analysis_dir}/auto_selected.json")
    print(f"    {analysis_dir}/口误分析.md")
    print(f"    {analysis_dir}/口误标注字幕.txt")
    _loop_out = analysis_dir / "review_loop_decisions.json"
    if _loop_out.exists():
        print(f"    {_loop_out}")

    return {
        "candidates": total_candidates,
        "deletes": stats["total"],
        "elapsed": elapsed,
    }


# ============================================================
#  入口（参数全部写死，改这里即可）
# ============================================================


# 让本模块可作为包内模块导入
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── 写死的默认参数 ──
BASE_DIR = str(ROOT / "2026-07-07_福总")   # 数据根目录（换数据改这一行）
LOOP_VERSION = "v1"                         # 循环审查模式（v1 默认；v2=专职Agent三分）
SILENCE_THRESH = 0.9
# 删除静音时保留的句间呼吸感时长（秒）。
SILENCE_KEEP_DURATION = 0.5
MODEL = "deepseek-v4-pro"
# MODEL = "deepseek-chat"
VIDEO_DURATION = 0.0
ENABLE_DEEPSEEK_THINKING = False    # 关闭 DeepSeek 思考模式（加速）
SPLIT_MODE = "hybrid"         # 按静音 gap 切分
USE_ORIGINAL_SCRIPT = True     # 启用原稿对照
SKIP_JUDGE = False
SKIP_LOOP = True
SKIP_REVIEW = True
MAX_DET_ROUNDS = 4
REVIEW_SERVE = True



def main() -> None:
    import argparse

    # 仅 base-dir / loop-version 可由命令行覆盖；其余一律用「写死的默认参数」。
    # argparse 不设值时回退到下方写死的 BASE_DIR / LOOP_VERSION 常量。
    parser = argparse.ArgumentParser(
        description="口误检测系统 (Judge + Agent 循环审查)",
    )
    parser.add_argument("--base-dir", default=BASE_DIR,
                        help="数据根目录：红姐 或 福总（默认 2026-07-07_红姐）")
    parser.add_argument("--loop-version", default=LOOP_VERSION, choices=["v1", "v2"],
                        help="循环审查模式：v1=单检测单验证(默认)；"
                             "v2=按类型拆分的专职 Agent 三分")
    args = parser.parse_args()

    # --base-dir 相对名（如 2026-07-07_福总）一律相对 ROOT 解析；
    # 已是绝对路径则原样保留（Path 除法遇绝对路径返回绝对路径）。
    _base = Path(args.base_dir)
    if not _base.is_absolute():
        _base = ROOT / _base
    base_dir = str(_base)

    # 1) 跑完整 pipeline 检测（除 base-dir / loop-version 外全部写死）
    run_pipeline(
        base_dir=base_dir,
        skip_judge=SKIP_JUDGE,
        skip_loop=SKIP_LOOP,
        silence_thresh=SILENCE_THRESH,
        model=MODEL,
        video_duration=VIDEO_DURATION,
        enable_deepseek_thinking=ENABLE_DEEPSEEK_THINKING,
        split_mode=SPLIT_MODE,
        use_original_script=USE_ORIGINAL_SCRIPT,
        loop_version=args.loop_version,
        max_det_rounds=MAX_DET_ROUNDS
    )

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
