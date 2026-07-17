"""
speech_pipeline.py — 口误检测主入口（五层架构）

循环审查固定使用 V3（speech_error_detector.agent_review，对照原稿找"说错"）。
完整运行入口(main) 与写死参数已移到 test/run_speech_pipeline.py。

执行流程:
  步骤0: gen_texts()     → readable.txt + sentences.txt
  步骤1-4: detect_loop    → 机械检测 + 研判 + 应用（多轮收敛）
  步骤5-7: V3 Agent 循环审查 → review_loop_decisions.json
  步骤8:   assemble()     → auto_selected.json（合并 Judge + 循环审查 + 静音/语气词）
  步骤9:  修正后句子       → 直接写回 sentences.txt（不再另存 updated_sentences.txt）

编程入口:
  from speech_error_detector.speech_pipeline import run_pipeline

运行整条流水线 + 审核:
  python -m speech_error_detector.test.run_speech_pipeline
未发现 1_转录 目录时，run_pipeline 会自动从目录内视频转录。
"""

import json
import shutil
import time
from pathlib import Path

from speech_error_detector.config import (
    DEFAULT_MODEL,
    DEFAULT_DETECT_REPEAT_MODEL,
    SILENCE_THRESH,
    SPLIT_MODE,
    USE_ORIGINAL_SCRIPT,
    MAX_DET_ROUNDS,
    MAX_LOOP_ROUNDS,
    VIDEO_DURATION,
    LANGUAGE,
    TRANSCRIPT_DIR,
    ANALYSIS_DIR,
    REVIEW_DIR,
    WORDS_JSON,
    VOLC_RESULT_JSON,
    ORIGINAL_SCRIPT_FILE,
    SENTENCES_ORIGIN_FILE,
)


from speech_error_detector.detect_repeat.detect_loop import run_detect_judge_loop
from speech_error_detector.assemble.assemble import run_assemble, generate_report_markdown, generate_updated_sentences
from speech_error_detector.detect_agent.review_loop import run_agent_review_loop_v3
from speech_error_detector.detect_agent.agent_apply import apply_deletions_to_sentences
from speech_error_detector.utils.sentence_io import load_sentences, write_sentences
from speech_error_detector.utils.paths import detect_agent_dir, make_subdirs
from speech_error_detector.assemble.subtitle_generator import (
    generate_readable_txt,
    generate_sentences_txt,
    generate_subtitles_words,
)
from speech_error_detector.utils.audio_extractor import extract_audio, get_video_info
from speech_error_detector.utils.volcengine_transcriber import transcribe


# 视频扩展名（自动探测目录内视频用）
_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".flv", ".m4v")


def _find_video(base_dir: Path) -> Path | None:
    """在 base_dir 下查找真实视频文件（排除 3_审核 与符号链接）。"""
    for ext in _VIDEO_EXTS:
        for p in sorted(base_dir.rglob(f"*{ext}")):
            if REVIEW_DIR in p.parts or p.is_symlink():
                continue
            return p
    return None


def transcribe_video(
    base_dir: str | Path,
    video_file: str | Path | None = None,
    language: str = "zh-CN",
) -> Path:
    """从目录里的视频提取音频 → 火山识别 → 创建 1_转录 转录目录。

    步骤:
      1. 定位视频（video_file 指定，或自动探测 base_dir 下首个视频）
      2. 提取 16k 单声道音频到 1_转录/audio_0.mp3
      3. 调用火山引擎 Seed ASR 2.0 转录 → volcengine_result.json
      4. 转字级时间轴 → subtitles_words.json

    Args:
        base_dir:   项目目录（视频 + optional original_script.txt 所在）
        video_file: 视频路径；为 None 时自动探测 base_dir 下的视频
        language:   识别语言（透传给火山 ASR）

    Returns:
        transcript_dir (Path) = base_dir / TRANSCRIPT_DIR
    """
    base_dir = Path(base_dir)
    transcript_dir = base_dir / TRANSCRIPT_DIR
    transcript_dir.mkdir(parents=True, exist_ok=True)
 
    # 1) 定位视频
    video_path = Path(video_file) if video_file else _find_video(base_dir)
    if not video_path or not video_path.exists():
        raise FileNotFoundError(
            f"未在 {base_dir} 找到视频文件（或指定的 {video_file} 不存在）"
        )
    print("=" * 60)
    print("  转录阶段: 视频 → 音频 → 火山识别 → 1_转录")
    print("=" * 60)
    print(f"   🎬 视频: {video_path.name}")

    # 2) 提取音频
    audio_path = extract_audio(str(video_path), str(transcript_dir), suffix="0")
    print(f"   🎧 音频: {Path(audio_path).name}")

    # 3) 火山识别
    result_json, _ = transcribe(Path(audio_path), transcript_dir, language=language)
    print(f"   📝 转录结果: {result_json.name}")

    # 4) 转字级时间轴
    words_json = generate_subtitles_words(result_json, transcript_dir)
    print(f"   ✅ 转录目录就绪: {transcript_dir}")
    print(f"      {words_json.name} | {result_json.name}")
    return transcript_dir



# ============================================================
#  步骤0: gen_texts
# ============================================================

def gen_texts(base_dir: Path, split_mode: str) -> tuple[Path, Path]:
    """由 subtitles_words.json 生成 readable.txt 和 sentences.txt（使用 subtitle_generator 模块，含序号合并）"""
    words_json = base_dir / TRANSCRIPT_DIR / WORDS_JSON
    volcengine_result = base_dir / TRANSCRIPT_DIR / VOLC_RESULT_JSON
    analysis_dir = base_dir / ANALYSIS_DIR
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
    analysis_dir = base_dir / ANALYSIS_DIR
    analysis_dir.mkdir(parents=True, exist_ok=True)
    make_subdirs(analysis_dir)
    # 非 skip_loop → 清空 loop 子目录（旧循环审查文件会干扰新结果）
    if not skip_loop:
        _loop = detect_agent_dir(analysis_dir)
        for _f in _loop.iterdir():
            if _f.is_file():
                _f.unlink()
            elif _f.is_dir():
                shutil.rmtree(_f)
    return analysis_dir


def _load_original_script(original_script_path: Path) -> str:
    """读取原稿（不存在则返回空串）。原稿放在项目根目录下，不另拷贝。"""
    if not original_script_path.exists():
        return ""
    return original_script_path.read_text(encoding="utf-8")


def _run_review_loop_phase(
    *,
    analysis_dir: Path,
    words_data: list,
    model: str,
    use_original_script: bool,
    original_script: str,
    cleaned: list[dict],
    enable_deepseek_thinking: bool,
    sentences_path: Path,
    original_sentences: list[dict],
    skip_loop: bool,
    max_rounds: int,
) -> tuple[list[dict], list[dict]]:
    """步骤5-7: Agent 循环审查。返回 (最终句子列表, 循环审查决策列表)。"""
    if skip_loop:
        # 跳过 loop agent，但复用磁盘上已存在的 review_loop_decisions.json：
        # 把其中「已确认删除」(confirmed 且 action != "report") 的决策真正投影进 sentences.txt，
        # 使正文与报告（报告恒读该缓存）保持一致；不再像旧逻辑那样只返回 cleaned 导致报告谎报。
        print("-" * 40)
        print("[步骤5-7] 跳过 Agent 循环审查：复用磁盘 review_loop_decisions.json 的已确认删除")
        print("-" * 40)
        current_sentences = cleaned
        # review_loop_decisions.json 写在 loop/ 子目录下（与 run_review_loop / assemble 同路径）；
        # 兼容旧数据曾放在 analysis_dir 根的情况。
        _loop_cache = detect_agent_dir(analysis_dir) / "review_loop_decisions.json"
        if not _loop_cache.exists():
            _loop_cache = analysis_dir / "review_loop_decisions.json"
        _cached_decisions: list[dict] = []
        if _loop_cache.exists():
            try:
                with open(_loop_cache, encoding="utf-8") as _f:
                    _cached_decisions = json.load(_f)
            except Exception as _e:
                print(f"  [警告] 读取 {_loop_cache.name} 失败: {_e}")
        # 与 assemble.py 同口径：confirmed 且 action != "report"（✅ 删除；排除 📌 仅标注 / ❌ 驳回）
        _confirmed_issues = [
            d["detect"] for d in _cached_decisions
            if d.get("decision", {}).get("confirmed") is True
            and d.get("decision", {}).get("action") != "report"
        ]
        if _confirmed_issues:
            current_sentences, _ = apply_deletions_to_sentences(
                sentences=cleaned,
                words=words_data,
                confirmed_issues=_confirmed_issues,
            )
            # 落盘，使磁盘与内存一致（供下游 generate_updated_sentences / 报告使用）
            write_sentences(sentences_path, current_sentences, original=original_sentences)
            print(f"   已复用缓存确认删除: {len(_confirmed_issues)} 项 → 回写 {sentences_path.name}")
        else:
            print(f"   loop 补充删除: 0 项（无可用缓存或缓存无已确认删除）")
        return current_sentences, _cached_decisions

    print("-" * 40)
    print("[步骤5-7] Agent 循环审查")
    print("-" * 40)

    # LLM 循环检测（复用 Judge 清洗后的 cleaned 引用，不再回读磁盘）
    # 固定使用 V3：对照原稿找"说错"（检测 + 确认 双 Agent）。
    _loop_path, loop_decisions, current_sentences = run_agent_review_loop_v3(
        loop_dir=detect_agent_dir(analysis_dir),
        words=words_data,
        model=model,
        use_original_script=use_original_script,
        original_script=original_script,
        sentences=cleaned,
        max_rounds=max_rounds,
        consecutive_empty_to_exit=1,
        enable_thinking=enable_deepseek_thinking,
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


def _generate_updated_sentences_phase(
    analysis_dir: Path, words_json: Path, sentences_path: Path,
    current_sentences: list[dict], original_sentences: list[dict],
) -> None:
    """步骤10: 把修正后句子列表直接写回 sentences.txt（覆盖，不另存 updated_sentences.txt）。"""
    print("-" * 40)
    print("[步骤10] 写回修正后句子列表 → sentences.txt")
    print("-" * 40)

    generate_updated_sentences(
        analysis_dir, words_json, 
        sentences=current_sentences, original=original_sentences,
        out_path=sentences_path,
    )
    print(f"  sentences.txt: 已更新（应用删除后）")
    print()


# ============================================================
#  run_pipeline — 主编排器
# ============================================================

def run_pipeline(
    base_dir: str | Path,
    video_path: str | Path,
    skip_judge: bool = False,
    skip_loop: bool = False,
    silence_thresh: float = SILENCE_THRESH,
    detect_repeat_model: str = DEFAULT_DETECT_REPEAT_MODEL,
    detect_agent_model: str = DEFAULT_MODEL,
    video_duration: float = VIDEO_DURATION,
    enable_deepseek_thinking: bool = False,
    split_mode: str = SPLIT_MODE,
    use_original_script: bool = USE_ORIGINAL_SCRIPT,
    max_det_rounds: int = MAX_DET_ROUNDS,
    max_loop_rounds: int = MAX_LOOP_ROUNDS,
    language: str = LANGUAGE,
) -> dict:
    """
    运行完整的口误检测流水线（编排各步骤子函数）。

    步骤0:  gen_texts()          → readable.txt + sentences.txt
    步骤1-4: run_detect_judge_loop()  → 机械检测 + 研判 + 应用（多轮收敛）
    步骤5-7: _run_review_loop_phase()  → Agent 循环审查
    步骤8:   _run_assemble_and_report() → auto_selected.json + 报告
    步骤9:  _generate_updated_sentences_phase() → 直接写回 sentences.txt

    Args:
        base_dir: 数据根目录 (包含 1_转录/)
        skip_judge: 跳过 LLM 研判 Layer 2（使用磁盘已有的 decisions_*.json）
        skip_loop: 跳过 Agent 循环审查（步骤5-7）
        silence_thresh: 静音删除阈值(秒)
        detect_repeat_model: 机械检测(重复/残句)所用模型，默认 deepseek-v4-pro（DeepSeek pro）
        detect_agent_model:  Agent 循环审查(V3) 所用模型，默认 DEFAULT_MODEL
        video_duration: 视频时长(秒)，用于结尾补尾
        enable_deepseek_thinking: 开启 DeepSeek 思考模式
        video_path: 视频文件路径

    Returns:
        汇总统计字典
    """
    base_dir = Path(base_dir)
    analysis_dir = _prepare_analysis_dir(base_dir, skip_loop=skip_loop)

    # 预检：转录目录（1_转录）不存在才调用火山转录（提取音频→火山识别→1_转录）
    transcript_dir = base_dir / TRANSCRIPT_DIR
    if not transcript_dir.exists():
        print("=" * 60)
        print("  [预检] 未发现 1_转录 目录，自动从视频转录...")
        print("=" * 60)
        _vid = Path(video_path)
        transcribe_video(base_dir, video_file=_vid, language=language)
        if video_duration == 0.0 and _vid and _vid.exists():
            try:
                video_duration = get_video_info(str(_vid)).get("duration", 0.0)
                print(f"   ⏱️  视频时长: {video_duration:.1f}s（用于结尾补尾）")
            except Exception:
                pass

    words_json = base_dir / TRANSCRIPT_DIR / WORDS_JSON
    # 原稿放在项目根目录下
    original_script_path = base_dir / ORIGINAL_SCRIPT_FILE
    transcript_dir = base_dir / TRANSCRIPT_DIR

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

    # ========== 加载基础数据 ==========
    original_script = _load_original_script(original_script_path)
    sentences = load_sentences(sentences_path)

    print(f"\n  原稿长度: {len(original_script)} 字" if original_script else "  (无原稿)")
    print(f"  口播句子数: {len(sentences)}")
    print()

    # ========== 备份最原始句子（detect 之前） ==========
    # 存一份未做任何检测清洗的 sentences_origin.txt，便于后续对照/回溯
    sentences_origin_path = analysis_dir / SENTENCES_ORIGIN_FILE
    write_sentences(sentences_origin_path, sentences)
    print(f"  已备份原始句子: {sentences_origin_path.name} ({len(sentences)} 句)")

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
        model=detect_repeat_model,
        enable_deepseek_thinking=enable_deepseek_thinking,
        skip_judge=skip_judge,
        max_det_rounds=max_det_rounds,
        original_script=original_script,
    )

    # ========== 步骤5-7: Agent 循环审查 ==========
    current_sentences, loop_decisions = _run_review_loop_phase(
        analysis_dir=analysis_dir,
        words_data=words_data,
        model=detect_agent_model,
        use_original_script=use_original_script,
        original_script=original_script,
        cleaned=cleaned,
        enable_deepseek_thinking=enable_deepseek_thinking,
        sentences_path=sentences_path,
        original_sentences=original_sentences,
        skip_loop=skip_loop,
        max_rounds=max_loop_rounds
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

    # ========== 步骤10: 修正后句子列表直接写回 sentences.txt ==========
    _generate_updated_sentences_phase(
        analysis_dir, words_json, sentences_path, current_sentences, original_sentences,
    )

    # ========== 完成 ==========
    elapsed = time.time() - t_start

    print("=" * 60)
    print("  检测完成！")
    print("=" * 60)
    print(f"  耗时: {elapsed:.1f}s")
    print(f"  候选异常: {total_candidates}")
    print(f"  最终删除: {stats['total']} 个 word 索引")

    return {
        "candidates": total_candidates,
        "deletes": stats["total"],
        "elapsed": elapsed,
    }
