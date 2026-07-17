"""
detect_partial.py — 句间部分删除（保头删尾 / keep_head）机械检测

设计定位（见重构说明）：
- inter / fragment 只做「整句删除」；partial 只做「部分删除（keep_head）」。
- 本模块接管所有「前句 A 与后句 B 重叠、且 A 头部独有 → 保头删尾」的现象，
  统一收口，不再分散在 fragment / inter 两个整句删除器里。

检测策略（全部产出 keep_head 候选，head_text 为 A 独有头部）：
1. 前句尾与后句头重叠（相邻，尾对齐 / 近尾对齐 / 短重叠+长停顿）
2. 头体重叠（跨 ≤2 句，A 尾部 == B 头部）
3. 非前缀子串重叠（B 包含 A 大部分内容但不在开头 → A 头部独有）
4. 尾部重叠（A 尾部 == B 尾部 → A 为残次近重复，头部独有）

关键护栏：若推算出 A 头部为空（前句完全被后句覆盖）→ 那是「整句删除」，
  不属于 partial 职责，直接跳过（交给 inter 处理），绝不在此生成候选。

输出: 每条 finding 含 head_text / head_word_start,end / decision_hint(mode=keep_head)
"""

from pathlib import Path

from speech_error_detector.detect_repeat import CN_DIGIT_MAP, normalize_numerals
from speech_error_detector.utils.fillers import MODAL_CHARS
from speech_error_detector.config import PARTIAL_MIN_OVERLAP


def _norm(t: str) -> str:
    """去掉语气衬字 + 统一数字后的规范化文本（用于近重复匹配）。"""
    return "".join(CN_DIGIT_MAP.get(ch, ch) for ch in t if ch not in MODAL_CHARS)


def _norm_frag(t: str) -> str:
    """去掉语气衬字 + 统一数字后的规范化文本（用于头体重叠匹配）。"""
    return "".join(CN_DIGIT_MAP.get(ch, ch) for ch in t if ch not in MODAL_CHARS)


def _compute_head_word_range(
    sentence: dict, words: list, head_char_len: int,
) -> tuple[int, int] | None:
    """从句子 range 和词表推算 head_text 对应的词索引区间。

    sentence: 含 range (如 "564-585") 的句子 dict
    words: subtitles_words.json 的词列表（按数组索引对应 wordIdx）
    head_char_len: head_text 的字符数

    Returns:
        (start_word_idx, end_word_idx) 或 None（无法计算时）
    """
    if not words or not sentence:
        return None
    range_str = sentence.get("range", "")
    if not range_str or "-" not in range_str:
        return None
    parts = range_str.split("-")
    try:
        start_idx = int(parts[0])
        end_idx = int(parts[1])
    except (ValueError, IndexError):
        return None

    char_count = 0
    word_end = start_idx
    for i in range(start_idx, min(end_idx + 1, len(words))):
        w = words[i]
        if not w.get("isGap"):
            char_count += len(w.get("text", ""))
        word_end = i
        if char_count >= head_char_len:
            break

    return (start_idx, word_end)


def _norm_pos_to_orig_pos(norm_pos: int, original: str) -> int:
    """将规范化文本（_norm 后）中的字符位置映射回原始文本位置（跳过语气衬字）。"""
    norm_idx = 0
    for orig_idx, ch in enumerate(original):
        if ch not in MODAL_CHARS:
            if norm_idx >= norm_pos:
                return orig_idx
            norm_idx += 1
    return len(original)


def detect_partial(sentences: list[dict], original_script: str, words: list | None = None) -> list[dict]:
    """
    执行句间部分删除（保头删尾）检测。

    Returns:
        findings: 检测结果列表（全部为 keep_head 候选，head 独有）
    """
    findings: list[dict] = []

    # ---------------------------------------------------------------
    # 1. 前句尾与后句头重叠（相邻，残句被接续重说 / 口吃重说）
    #    支持三种模式：
    #    a) 精确尾对齐：a[-L:] == b[:L]  (L>=4，无需停顿)
    #    b) 近尾对齐：A尾部有多余字符，从A后半段搜索子串匹配B头部
    #    c) 短重叠(2字) + 句间长停顿(>=0.8s)：口吃重说经典模式
    # ---------------------------------------------------------------
    TAIL_HEAD_K = 2
    TAIL_HEAD_GAP_MIN = 0.8
    for i in range(len(sentences) - 1):
        a, b = sentences[i]["text"], sentences[i + 1]["text"]
        if len(a) < 6 or len(b) < 6:
            continue
        na, nb = normalize_numerals(a), normalize_numerals(b)

        overlap = 0
        best_start = len(a)
        is_stutter = False

        gap_s = None
        if words:
            ra = sentences[i].get("range", "")
            rb = sentences[i + 1].get("range", "")
            if ra and rb:
                try:
                    ra_start, ra_end = map(int, ra.split("-"))
                    rb_start, rb_end = map(int, rb.split("-"))
                    if 0 <= ra_end < len(words) and 0 <= rb_start < len(words):
                        gap_s = words[rb_start].get("start", 0) - words[ra_end].get("end", 0)
                except (ValueError, AttributeError):
                    gap_s = None

        # 模式 a: 精确尾对齐
        for L in range(4, min(8, len(na), len(nb)) + 1):
            if na[-L:] == nb[:L]:
                overlap = L
                best_start = len(a) - L

        # 模式 b: 近尾对齐（A 尾部有额外字符）
        if overlap == 0:
            search_from = max(len(na) // 2, 0)
            search_to = len(na) - 4
            for start in range(search_from, search_to + 1):
                remaining = na[start:]
                for L in range(min(8, len(remaining), len(nb)), 3, -1):
                    if remaining[:L] == nb[:L]:
                        if L > overlap:
                            overlap = L
                            best_start = start
                        break

        # 模式 c: 短重叠 + 长停顿 → 口吃重说
        if overlap == 0 and words is not None:
            if len(na) >= TAIL_HEAD_K and len(nb) >= TAIL_HEAD_K and na[-TAIL_HEAD_K:] == nb[:TAIL_HEAD_K]:
                if gap_s is not None and gap_s >= TAIL_HEAD_GAP_MIN:
                    overlap = TAIL_HEAD_K
                    best_start = len(a) - TAIL_HEAD_K
                    is_stutter = True

        if overlap >= 2:
            a_head = a[:best_start]
            if len(a_head) == 0:
                # 头部为空 → 整句完整包含于后句 → 归 inter 整句删，partial 不产出
                continue
            _hw = _compute_head_word_range(sentences[i], words, len(a_head)) if words else None
            _hint_extra = ""
            if _hw:
                _hint_extra = f" keep_head 词索引区间: [{_hw[0]}, {_hw[1]}](请直接使用此区间, 不要自行推算)"
            decision_hint = (
                f"删前保后→默认保头删尾; 句{sentences[i]['idx']}头部『{a_head}』为独有内容"
                f"(后句『{b}』不含该内容), 必须 mode=keep_head 保留头部、只删尾部重叠的"
                f"『{a[best_start:]}』; 仅当头部被后句完整包含时才允许 mode=full 整句删"
                f"{_hint_extra}"
            )
            _subtype = ("句尾句头重叠+长停顿(口吃重说)" if is_stutter
                        else "残句(被后句接续重说)")
            _fnd: dict = {
                "type": "partial",
                "subtype": _subtype,
                "sent_idx": sentences[i]["idx"],
                "range": sentences[i]["range"],
                "text": a,
                "next_sent_idx": sentences[i + 1]["idx"],
                "next_sent_text": b,
                "overlap_len": overlap,
                "head_text": a_head,
                "head_char_len": len(a_head),
                "decision_hint": decision_hint,
            }
            if is_stutter:
                _fnd["gap_seconds"] = round(gap_s, 2) if gap_s is not None else None
            if _hw:
                _fnd["head_word_start"] = _hw[0]
                _fnd["head_word_end"] = _hw[1]
            findings.append(_fnd)

    # ---------------------------------------------------------------
    # 2. 头体重叠（跨 ≤2 句）：A 尾部 == B 头部 → 保头删尾
    # ---------------------------------------------------------------
    _HEAD_MIN = 5
    _GAP_MAX = 3
    _TAIL_RATIO = 0.5  # B 头部须落在 A 后 50% 才算"尾部"重叠

    def _longest_head_in_tail(b_norm: str, a_norm: str) -> tuple[int, int]:
        target = b_norm[:_HEAD_MIN]
        if target not in a_norm:
            return (0, -1)
        tail_start = int(len(a_norm) * _TAIL_RATIO)
        best_len = 0
        best_pos = -1
        start = 0
        while True:
            pos = a_norm.find(target, start)
            if pos == -1:
                break
            if pos == 0:
                start = pos + 1
                continue
            if pos < tail_start:
                start = pos + 1
                continue
            L = _HEAD_MIN
            while pos + L < len(a_norm) and L < len(b_norm) and a_norm[pos + L] == b_norm[L]:
                L += 1
            if L > best_len:
                best_len = L
                best_pos = pos
            start = pos + 1
        return (best_len, best_pos)

    for j in range(1, len(sentences)):
        b = sentences[j]["text"]
        if len(b) < _HEAD_MIN + 1:
            continue
        nb = _norm_frag(b)
        for offset in range(1, min(j, _GAP_MAX) + 1):
            i = j - offset
            a = sentences[i]["text"]
            if len(a) < _HEAD_MIN:
                continue
            na = _norm_frag(a)
            match_len, pos = _longest_head_in_tail(nb, na)
            if match_len >= _HEAD_MIN and pos > 0:
                split_orig = _norm_pos_to_orig_pos(pos, a)
                a_head = a[:split_orig]
                a_tail = a[split_orig:]
                if len(a_head) == 0 or len(a_tail) == 0:
                    continue  # 头部为空 → 整句删, 归 inter
                _hw = _compute_head_word_range(sentences[i], words, len(a_head)) if words else None
                _hint_extra = ""
                if _hw:
                    _hint_extra = (
                        f" 保头词区间(可直接用): [{_hw[0]}, {_hw[1]}]"
                        f"(保留『{a_head}』, 删除尾部『{a_tail}』)"
                    )
                decision_hint = (
                    f"保头删尾(keep_head)→前句{sentences[i]['idx']}尾部『{a_tail}』"
                    f"与后句{sentences[j]['idx']}头部重叠({match_len}字), "
                    f"重叠应删前面的, 故需 mode=keep_head 保留前句独有头部『{a_head}』、"
                    f"删除冗余尾部『{a_tail}』; 后句{sentences[j]['idx']}完整保留"
                    f"{_hint_extra}"
                )
                _fnd = {
                    "type": "partial",
                    "subtype": f"前句尾与后句头重叠(头体重叠,跳过{offset}句)",
                    "sent_idx": sentences[i]["idx"],
                    "range": sentences[i]["range"],
                    "text": a,
                    "next_sent_idx": sentences[j]["idx"],
                    "next_sent_text": b,
                    "overlap_len": match_len,
                    "head_text": a_head,
                    "head_char_len": len(a_head),
                    "tail_text": a_tail,
                    "decision_hint": decision_hint,
                }
                if _hw:
                    _fnd["head_word_start"] = _hw[0]
                    _fnd["head_word_end"] = _hw[1]
                findings.append(_fnd)
                break

    # ---------------------------------------------------------------
    # 3. 非前缀子串重叠：B 包含 A 大部分内容但不在开头 → A 头部独有 → 保头删尾
    # ---------------------------------------------------------------
    for i in range(len(sentences) - 1):
        a_text, b_text = sentences[i]["text"], sentences[i + 1]["text"]
        na, nb = _norm(a_text), _norm(b_text)
        if len(na) < 8 or len(nb) < 12:
            continue

        max_overlap = 0
        best_start = 0
        for start in range(1, min(len(na) - 8, 12)):
            remaining = na[start:]
            for win_len in range(min(len(remaining), 30), 7, -1):
                sub = remaining[:win_len]
                if sub in nb:
                    if win_len > max_overlap:
                        max_overlap = win_len
                        best_start = start
                    break

        if max_overlap >= PARTIAL_MIN_OVERLAP and len(a_text) > 0:
            a_head = a_text[:best_start]
            if len(a_head) == 0:
                continue  # 头部被完全覆盖 → 整句删, 归 inter
            _hw = _compute_head_word_range(sentences[i], words, len(a_head)) if words else None
            _hint_extra = ""
            if _hw:
                _hint_extra = f" 保头词区间(可直接用): [{_hw[0]}, {_hw[1]}](保留『{a_head}』)"
            decision_hint = (
                f"保头删尾(keep_head)→句{sentences[i]['idx']}非头部内容被后句{sentences[i+1]['idx']}"
                f"包含(重叠{max_overlap}字), 前句头部『{a_head}』独有, 须 mode=keep_head 保留头部、删尾部"
                f"{_hint_extra}"
            )
            _fnd = {
                "type": "partial",
                "subtype": "非前缀子串重叠(保头删尾)",
                "sent_idx": sentences[i]["idx"],
                "range": sentences[i]["range"],
                "text": a_text,
                "next_sent_idx": sentences[i + 1]["idx"],
                "next_sent_text": b_text,
                "overlap_len": max_overlap,
                "head_text": a_head,
                "head_char_len": len(a_head),
                "decision_hint": decision_hint,
            }
            if _hw:
                _fnd["head_word_start"] = _hw[0]
                _fnd["head_word_end"] = _hw[1]
            findings.append(_fnd)

    # ---------------------------------------------------------------
    # 4. 尾部重叠：A 尾部 == B 尾部 → A 为残次近重复, 头部独有 → 保头删尾
    # ---------------------------------------------------------------
    TAIL_K = 4
    for i in range(len(sentences) - 1):
        a_orig = sentences[i]["text"]
        a_norm = _norm(a_orig)
        if len(a_norm) < 5:
            continue
        for j in range(i + 1, min(i + 3, len(sentences))):
            b_norm = _norm(sentences[j]["text"])
            k = min(TAIL_K, len(a_norm), len(b_norm))
            if k < TAIL_K:
                continue
            if len(a_norm) < len(b_norm) and a_norm[-k:] == b_norm[-k:]:
                split_orig = _norm_pos_to_orig_pos(len(a_norm) - k, a_orig)
                a_head = a_orig[:split_orig]
                a_tail = a_orig[split_orig:]
                if len(a_head) == 0 or len(a_tail) == 0:
                    continue  # 头部为空 → 整句删, 归 inter
                _hw = _compute_head_word_range(sentences[i], words, len(a_head)) if words else None
                _hint_extra = ""
                if _hw:
                    _hint_extra = (
                        f" 保头词区间(可直接用): [{_hw[0]}, {_hw[1]}]"
                        f"(保留『{a_head}』, 删除尾部『{a_tail}』)"
                    )
                decision_hint = (
                    f"保头删尾(keep_head)→句{sentences[i]['idx']}尾部『{a_tail}』与句{sentences[j]['idx']}"
                    f"尾部重叠{k}字(去语气词), 前句头部『{a_head}』独有, 须 mode=keep_head 保留头部、删尾部"
                    f"{_hint_extra}"
                )
                _fnd = {
                    "type": "partial",
                    "subtype": "尾部重叠(保头删尾)",
                    "sent_idx": sentences[i]["idx"],
                    "range": sentences[i]["range"],
                    "text": a_orig,
                    "next_sent_idx": sentences[j]["idx"],
                    "next_sent_text": sentences[j]["text"],
                    "overlap_len": k,
                    "head_text": a_head,
                    "head_char_len": len(a_head),
                    "tail_text": a_tail,
                    "decision_hint": decision_hint,
                }
                if _hw:
                    _fnd["head_word_start"] = _hw[0]
                    _fnd["head_word_end"] = _hw[1]
                findings.append(_fnd)
                break

    return findings


def run_detect_partial(
    sentences: list[dict],
    output_dir: Path,
    words: list[dict],
    original_script: str,
) -> list[dict]:
    """运行句间部分删除检测（不写文件，仅返回结果）"""
    findings = detect_partial(sentences, words=words, original_script=original_script)
    print(f"   [detect_partial] 句间部分删除(保头删尾)发现: {len(findings)} 处")
    for fnd in findings:
        print(f"      - {fnd['subtype']} 句{fnd['sent_idx']}: {fnd['text'][:24]}")
    if not findings:
        print(f"      (无句间部分删除)")
    return findings
