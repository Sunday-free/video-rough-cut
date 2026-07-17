"""
subtitle_generator.py — 字幕轴生成模块

对应原 skill 脚本: generate_subtitles.js (已改为纯 Python 实现)
功能: 火山引擎结果 → 字级时间轴 (subtitles_words.json) → 易读格式/句子列表
"""

import json
from pathlib import Path

from speech_error_detector.utils.sentence_io import write_sentences
from speech_error_detector.config import WORDS_JSON, SENTENCES_FILE, READABLE_FILE, SILENCE_GAP_THRESHOLD


def _sentences_to_records(sentences: list[dict]) -> list[dict]:
    """subtitle_generator 内部句子 [{text, startIdx, endIdx}] →
    write_sentences 需要的 [{idx, range, text}]。"""
    records = []
    for i, s in enumerate(sentences):
        a, b = s.get("startIdx"), s.get("endIdx")
        r = f"{a}-{b}" if a is not None and b is not None and a >= 0 else ""
        records.append({"idx": i, "range": r, "text": s.get("text", "")})
    return records


def generate_subtitles_words(volcengine_result: Path, output_dir: Path) -> Path:
    """
    从火山引擎结果生成字级字幕轴。
    
    输出: subtitles_words.json
      格式: [{text, start, end, isGap}, ...]
      - isGap=True 表示静音段
      - 时间单位为秒
    """
    result_path = output_dir / WORDS_JSON

    if result_path.exists() and result_path.stat().st_size > 0:
        print(f"   📂 字幕轴缓存存在: {result_path.name}")
        return result_path

    # 解析火山引擎结果
    with open(volcengine_result, "r", encoding="utf-8") as f:
        data = json.load(f)

    utterances = (
        data.get("utterances")
        or (data.get("result", {}) or {}).get("utterances")
        or []
    )
    if not utterances:
        raise RuntimeError(f"转录结果里没有 utterances: {volcengine_result}")

    # 提取所有字/词
    all_words = []
    for utt in utterances:
        for w in utt.get("words", []):
            text = str(w.get("text", "")).strip()
            st = float(w.get("start_time", -1))
            et = float(w.get("end_time", -1))
            if not text or st < 0 or et < 0:
                continue
            all_words.append({
                "text": text,
                "start": round(st / 1000, 3),
                "end": round(et / 1000, 3),
            })

    if not all_words:
        raise RuntimeError("过滤后无有效字/词")

    # 插入静音标记 (isGap)
    words_with_gaps = []
    last_end = 0.0

    for w in all_words:
        gap_dur = w["start"] - last_end

        if gap_dur > 0.1:
            if gap_dur > 0.5:
                # >0.5s 按 1 秒拆分
                gap_start = last_end
                while gap_start < w["start"]:
                    gap_end = min(gap_start + 1.0, w["start"])
                    words_with_gaps.append({
                        "text": "",
                        "start": round(gap_start, 2),
                        "end": round(gap_end, 2),
                        "isGap": True,
                    })
                    gap_start = gap_end
            else:
                words_with_gaps.append({
                    "text": "",
                    "start": round(last_end, 2),
                    "end": round(w["start"], 2),
                    "isGap": True,
                })

        words_with_gaps.append({**w, "isGap": False})
        last_end = w["end"]

    # 写入文件
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(words_with_gaps, f, ensure_ascii=False, indent=2)

    total = len(words_with_gaps)
    gaps = sum(1 for w in words_with_gaps if w.get("isGap"))
    print(f"   ✅ 字幕轴: {total} 个元素 (文字 {total - gaps}, 静音段 {gaps})")

    return result_path


def generate_readable_txt(subtitles_words: Path, analysis_dir: Path) -> Path:
    """生成 readable.txt (逐字可读稿，含静音段标记)。"""
    output = analysis_dir / READABLE_FILE
    
    with open(subtitles_words, "r") as f:
        words = json.load(f)

    lines = []
    for i, w in enumerate(words):
        if w.get("isGap"):
            dur = w["end"] - w["start"]
            if dur >= 0.2:
                lines.append(f"{i}|[静{dur:.2f}s]|{w['start']:.2f}-{w['end']:.2f}")
        else:
            lines.append(f"{i}|{w['text']}|{w['start']:.2f}-{w['end']:.2f}")

    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"   📄 readable.txt: {len(lines)} 行")
    return output


def generate_sentences_by_utterance(
    subtitles_words: Path,
    analysis_dir: Path,
    volcengine_result: Path,
) -> tuple[Path, list]:
    """
    按 volcengine_result.json 的 utterance 边界切分句子 (sentences.txt)。

    Args:
        subtitles_words:   字幕轴文件 (subtitles_words.json)
        analysis_dir:      输出目录
        volcengine_result: 火山引擎原始结果，用于获取 utterance 边界

    Returns:
        (sentences_path, sentences_list)
        sentences_list: [{text, startIdx, endIdx}, ...]
    """
    output = analysis_dir / SENTENCES_FILE

    # 加载扁平字幕轴
    with open(subtitles_words, "r") as f:
        words = json.load(f)

    # 加载火山引擎原始结果，提取每个 utterance 的 word 时间范围
    with open(volcengine_result, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_utts = (
        data.get("utterances")
        or (data.get("result", {}) or {}).get("utterances")
        or []
    )

    # 构建每个 utterance 的 [start_ms, end_ms) 区间
    utt_intervals = []
    for utt in raw_utts:
        ws = utt.get("words", [])
        if not ws:
            continue
        start_ms = min(float(w.get("start_time", 0)) for w in ws)
        end_ms = max(float(w.get("end_time", 0)) for w in ws)
        utt_intervals.append((start_ms / 1000, end_ms / 1000))  # 转秒

    if not utt_intervals:
        raise RuntimeError(f"volcengine_result 中无有效 utterances: {volcengine_result}")

    # 将 words 按 utterance 边界分组
    # 每个 word 属于哪个 utterance → 取 word 中点时间，找包含它的区间
    sentences = []
    curr = {"text": "", "startIdx": -1, "endIdx": -1}
    curr_utt_idx = -1

    for i, w in enumerate(words):
        if w.get("isGap"):
            continue

        w_mid = (w["start"] + w["end"]) / 2

        # 找这个 word 属于哪个 utterance
        utt_idx = -1
        for idx, (u_start, u_end) in enumerate(utt_intervals):
            if u_start <= w_mid < u_end:
                utt_idx = idx
                break

        # utterance 变了 → 断句
        if utt_idx != curr_utt_idx and curr_utt_idx != -1:
            if curr["text"]:
                sentences.append(curr.copy())
            curr = {"text": "", "startIdx": -1, "endIdx": -1}

        if curr["startIdx"] == -1:
            curr["startIdx"] = i
        curr["text"] += w["text"]
        curr["endIdx"] = i
        curr_utt_idx = utt_idx

    # 最后一句
    if curr["text"]:
        sentences.append(curr.copy())

    # === 后处理：合并极短序号句到相邻句 ===
    ORDINAL_CHARS = set("一二三四五六七八九十第")
    merged = []
    i = 0
    while i < len(sentences):
        s = sentences[i]
        text = s["text"].strip()

        if len(text) <= 2 and text and all(ch in ORDINAL_CHARS for ch in text):
            if i + 1 < len(sentences):
                nxt = sentences[i + 1]
                merged.append({
                    "text": s["text"] + nxt["text"],
                    "startIdx": s["startIdx"],
                    "endIdx": nxt["endIdx"],
                })
                i += 2
                continue

        merged.append(s)
        i += 1

    sentences = merged

    write_sentences(output, _sentences_to_records(sentences))
    print(f"   📄 sentences.txt (utterance): {len(sentences)} 句")
    return output, sentences


def generate_sentences_hybrid(
    subtitles_words: Path,
    analysis_dir: Path,
    volcengine_result: Path,
) -> tuple[Path, list]:
    """
    混合模式：以 utterance 边界为主，内部长静音为辅。

    - 两个不同 utterance 之间 → 强制断句
    - 同一个 utterance 内部 → 遇到 ≥ SILENCE_GAP_THRESHOLD 的静音也断句
    """
    output = analysis_dir / SENTENCES_FILE

    with open(subtitles_words, "r") as f:
        words = json.load(f)

    with open(volcengine_result, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_utts = (
        data.get("utterances")
        or (data.get("result", {}) or {}).get("utterances")
        or []
    )

    # 构建每个 utterance 的 [start_s, end_s) 区间
    utt_intervals = []
    for utt in raw_utts:
        ws = utt.get("words", [])
        if not ws:
            continue
        start_ms = min(float(w.get("start_time", 0)) for w in ws)
        end_ms = max(float(w.get("end_time", 0)) for w in ws)
        utt_intervals.append((start_ms / 1000, end_ms / 1000))

    # 遍历 words：两种断句条件同时生效
    sentences = []
    curr = {"text": "", "startIdx": -1, "endIdx": -1}
    curr_utt_idx = -1

    def _flush():
        nonlocal curr
        if curr["text"]:
            sentences.append(curr.copy())
            curr = {"text": "", "startIdx": -1, "endIdx": -1}

    for i, w in enumerate(words):
        is_gap = w.get("isGap", False)

        if not is_gap:
            w_mid = (w["start"] + w["end"]) / 2

            # 判断 word 属于哪个 utterance
            utt_idx = -1
            for idx, (u_start, u_end) in enumerate(utt_intervals):
                if u_start <= w_mid < u_end:
                    utt_idx = idx
                    break

            # 条件1: utterance 变了 → 断句
            if utt_idx != curr_utt_idx and curr_utt_idx != -1:
                _flush()
                curr_utt_idx = utt_idx
            elif curr_utt_idx == -1:
                curr_utt_idx = utt_idx

        # 条件2: 长静音 → 断句（不管是不是同一个 utterance）
        if is_gap and (w["end"] - w["start"]) >= SILENCE_GAP_THRESHOLD:
            _flush()

        if not is_gap:
            if curr["startIdx"] == -1:
                curr["startIdx"] = i
            curr["text"] += w["text"]
            curr["endIdx"] = i

    _flush()

    # === 后处理：合并极短序号句 ===
    ORDINAL_CHARS = set("一二三四五六七八九十第")
    merged = []
    j = 0
    while j < len(sentences):
        s = sentences[j]
        text = s["text"].strip()

        if len(text) <= 2 and text and all(ch in ORDINAL_CHARS for ch in text):
            if j + 1 < len(sentences):
                nxt = sentences[j + 1]
                merged.append({
                    "text": s["text"] + nxt["text"],
                    "startIdx": s["startIdx"],
                    "endIdx": nxt["endIdx"],
                })
                j += 2
                continue

        merged.append(s)
        j += 1

    sentences = merged

    write_sentences(output, _sentences_to_records(sentences))
    print(f"   📄 sentences.txt (hybrid): {len(sentences)} 句")
    return output, sentences


def generate_sentences_txt(
    subtitles_words: Path,
    analysis_dir: Path,
    split_mode: str,
    volcengine_result: Path | None = None,
) -> tuple[Path, list]:
    """
    生成句子列表 (sentences.txt)。

    Args:
        subtitles_words:   字幕轴文件
        analysis_dir:      输出目录
        split_mode:        "silence" = 按静音 gap 切分（默认）
                           "utterance" = 按 volcengine_result.json 的 utterance 边界切分
                           "hybrid" = utterance 边界为主 + 内部长静音辅助
        volcengine_result: split_mode=utterance/hybrid 时必须提供

    Returns:
        (sentences_path, sentences_list)
    """
    if split_mode in ("utterance", "hybrid"):
        if volcengine_result is None:
            raise ValueError(f"split_mode='{split_mode}' 需要提供 volcengine_result 路径")
        if split_mode == "hybrid":
            return generate_sentences_hybrid(subtitles_words, analysis_dir, volcengine_result)
        return generate_sentences_by_utterance(subtitles_words, analysis_dir, volcengine_result)

    # === 默认：按静音切分 ===
    output = analysis_dir / SENTENCES_FILE

    with open(subtitles_words, "r") as f:
        words = json.load(f)

    sentences = []
    curr = {"text": "", "startIdx": -1, "endIdx": -1}

    for i, w in enumerate(words):
        is_long_gap = w.get("isGap", False) and (w["end"] - w["start"]) >= SILENCE_GAP_THRESHOLD
        
        if is_long_gap:
            if curr["text"]:
                sentences.append(curr.copy())
            curr = {"text": "", "startIdx": -1, "endIdx": -1}
        elif not w.get("isGap"):
            if curr["startIdx"] == -1:
                curr["startIdx"] = i
            curr["text"] += w["text"]
            curr["endIdx"] = i

    # 最后一句
    if curr["text"]:
        sentences.append(curr.copy())

    # === 后处理：合并极短序号句到相邻句 ===
    # 口播者说序号时常停顿思考(如 "一... (停顿) 二... (停顿) 三 (停顿0.6s) 锂矿")
    # 导致序号被切成独立短句。检测并合并：
    ORDINAL_CHARS = set("一二三四五六七八九十第")
    merged = []
    i = 0
    while i < len(sentences):
        s = sentences[i]
        text = s["text"].strip()
        
        # 极短句(≤2字符)且为序号 → 尝试合并到下一句
        if len(text) <= 2 and text and all(ch in ORDINAL_CHARS for ch in text):
            if i + 1 < len(sentences):
                # 合并到下一句
                nxt = sentences[i + 1]
                merged.append({
                    "text": s["text"] + nxt["text"],
                    "startIdx": s["startIdx"],
                    "endIdx": nxt["endIdx"],
                })
                i += 2  # 跳过已合并的下一句
                continue
        
        merged.append(s)
        i += 1
    
    sentences = merged
    
    write_sentences(output, _sentences_to_records(sentences))
    print(f"   📄 sentences.txt: {len(sentences)} 句")
    return output, sentences
