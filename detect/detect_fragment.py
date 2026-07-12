"""
detect_fragment.py — 残句机械检测 + 原稿对齐提示

检测策略:
1. 孤立编号句 (如 "三" 单独成句)
2. 极短孤立句 (<=2字符)
3. 前句尾与后句头重叠 (残句被后句接续重说)

输出: detect_fragment.json
"""

from pathlib import Path

from .script_window import build_org_window
from . import normalize_numerals


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


def detect_fragment(sentences: list[dict], words: list | None = None, original_script: str = "") -> list[dict]:
    """
    执行残句检测。
    
    Returns:
        findings: 检测结果列表
    """
    findings = []
    
    # 中文数字编号集合（有序）
    numbers = {"一", "二", "三", "四", "五", "六", "七", "八", "九", "十"}
    
    def _is_ordinal_sequence(sent_idx: int) -> bool:
        """检查孤立编号是否为有序列的一部分"""
        t = sentences[sent_idx]["text"]
        
        # 查找前后是否有其他序号句（形成序列证据）
        has_ordinal_neighbor = False
        for other in sentences:
            if other["idx"] == sent_idx:
                continue
            ot = other["text"]
            # 其他句子以中文数字或"第X"开头 → 说明这是编号体系
            if ot and (ot[0] in numbers or ot.startswith("第")):
                has_ordinal_neighbor = True
                break
        
        if not has_ordinal_neighbor:
            return False
        
        # 检查后一句是否像内容延续（非同样极短）
        next_sent = None
        for s in sentences:
            if s["idx"] == sent_idx + 1:
                next_sent = s
                break
        
        if next_sent and len(next_sent["text"]) > 5:
            # 若后句以同一中文数字开头 → 说明本句是 fragment 残留，不是合法序号
            # 例：句40 "二" / 句42 "二人工智能..." → 句40 是残句，不应放过
            # if next_sent["text"][0] == t:
            #     return False
            return True
        
        return False
    
    # 1. 极短句检测 (孤立编号 / 极短孤立)
    for sent in sentences:
        t = sent["text"]
        if len(t) <= 2:
            if t in numbers:
                # 检查是否为有序列中的合法序号
                if _is_ordinal_sequence(sent["idx"]):
                    continue  # 是合法序号，跳过
                findings.append({
                    "type": "fragment",
                    "subtype": "孤立编号",
                    "sent_idx": sent["idx"],
                    "range": sent["range"],
                    "text": t,
                    "decision_hint": "孤立编号(如'三')→删整句, 内容在后续句",
                })
            else:
                findings.append({
                    "type": "fragment",
                    "subtype": "极短孤立句",
                    "sent_idx": sent["idx"],
                    "range": sent["range"],
                    "text": t,
                    "decision_hint": "疑似残句/孤立→删整句",
                })
    
    # 2. 前句尾与后句头重叠（残句被接续重说 / 口吃重说）
    #    支持三种模式：
    #    a) 精确尾对齐：a[-L:] == b[:L]  (L>=4，无需停顿)
    #    b) 近尾对齐：A尾部有多余字符，从A后半段搜索子串匹配B头部
    #    c) 短重叠(2字) + 句间长停顿(>=0.8s)：口吃重说经典模式
    #       （原 inter 检测器策略7，统一收归此处，复用 keep_head 部分删除）
    TAIL_HEAD_K = 2
    TAIL_HEAD_GAP_MIN = 0.8
    for i in range(len(sentences) - 1):
        a, b = sentences[i]["text"], sentences[i + 1]["text"]
        if len(a) < 6 or len(b) < 6:
            continue
        # 归一化文本（数字统一为阿拉伯数字）用于字面比较
        na, nb = normalize_numerals(a), normalize_numerals(b)

        overlap = 0
        best_start = len(a)
        is_stutter = False  # 模式 c 标记

        # 句间停顿（供模式 c 判定）
        gap_s = None
        if words:
            ra = sentences[i].get("range", "")
            rb = sentences[i + 1].get("range", "")
            if ra and rb:
                try:
                    ra_start, ra_end = map(int, ra.split("-"))
                    rb_start, rb_end = map(int, rb.split("-"))
                    if (0 <= ra_end < len(words) and 0 <= rb_start < len(words)):
                        gap_s = words[rb_start].get("start", 0) - words[ra_end].get("end", 0)
                except (ValueError, AttributeError):
                    gap_s = None

        # 模式 a: 精确尾对齐 (a[-L:] == b[:L])
        for L in range(4, min(8, len(na), len(nb)) + 1):
            if na[-L:] == nb[:L]:
                overlap = L
                best_start = len(a) - L

        # 模式 b: 非精确对齐，从A后半段搜索子串匹配B头部
        # 处理A尾部有额外字符的情况（如ASR切句位置不准）
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

        # 模式 c: 短重叠(2字) + 长停顿 → 口吃重说
        if overlap == 0 and words is not None:
            if (len(na) >= TAIL_HEAD_K and len(nb) >= TAIL_HEAD_K
                    and na[-TAIL_HEAD_K:] == nb[:TAIL_HEAD_K]):
                if gap_s is not None and gap_s >= TAIL_HEAD_GAP_MIN:
                    overlap = TAIL_HEAD_K
                    best_start = len(a) - TAIL_HEAD_K
                    is_stutter = True

        if overlap >= 2:
            a_head = a[:best_start]
            # 推算 head_text 对应的词索引区间（供 LLM judge 精准设置 keep_head）
            _hw = _compute_head_word_range(sentences[i], words, len(a_head)) if words else None
            # 头部为空 → 整句完整包含于后句，无独有内容，应整句删除
            if len(a_head) == 0:
                decision_hint = (
                    f"删前保后→句{sentences[i]['idx']}『{a}』完整包含于后句『{b}』头部"
                    f"(重叠{overlap}字), head_text 为空无独有内容, 必须 mode=full 整句删除"
                )
            else:
                _hint_extra = ""
                if _hw:
                    _hint_extra = f" keep_head 词索引区间: [{_hw[0]}, {_hw[1]}]（请直接使用此区间，不要自行推算）"
                decision_hint = (
                    f"删前保后→默认保头删尾; 句{sentences[i]['idx']}头部『{a_head}』为独有内容"
                    f"(后句『{b}』不含该内容), 必须 mode=keep_head 保留头部、只删尾部重叠的"
                    f"『{a[best_start:]}』; 仅当头部被后句完整包含时才允许 mode=full 整句删"
                    f"{_hint_extra}"
                )
            _subtype = ("句尾句头重叠+长停顿(口吃重说)" if is_stutter
                        else "残句(被后句接续重说)")
            _fnd: dict = {
                "type": "fragment",
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
            
            # 仅模式 c（口吃重说）嵌入原文稿对照片段（供 LLM 交叉验证）
            if is_stutter:
                _fnd["gap_seconds"] = round(gap_s, 2) if gap_s is not None else None
                window = build_org_window(original_script, [
                    (sentences[i]["idx"], a),
                    (sentences[i + 1]["idx"], b),
                ])
                if window:
                    _fnd["org_script_window"] = window
            if _hw:
                _fnd["head_word_start"] = _hw[0]
                _fnd["head_word_end"] = _hw[1]
            findings.append(_fnd)
    
    return findings


def run_detect_fragment(
    sentences: list[dict],
    output_dir: Path,
    words: list[dict],
    original_script: str,
) -> list[dict]:
    """运行残句检测（不写文件，仅返回结果）"""
    
    findings = detect_fragment(sentences, words=words, original_script=original_script)
    
    print(f"   [detect_fragment] 残句发现: {len(findings)} 处")
    for fnd in findings:
        print(f"      - {fnd['subtype']} 句{fnd['sent_idx']}: {fnd['text'][:24]}")
    if not findings:
        print(f"      (无残句)")
    
    return findings
