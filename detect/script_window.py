"""
script_window.py — 原文稿窗口匹配与截取

用于检测模块在发现疑似口吃/切句错误时，从原文稿中截取相关片段，
嵌入 finding 供 LLM 交叉验证。
"""

import difflib


def match_script_position(script: str, text: str, min_ratio: float = 0.4) -> int:
    """在原文稿中查找 text 的最佳匹配位置，返回字符偏移，-1 未找到。"""
    pos = script.find(text)
    if pos >= 0:
        return pos
    clean = text.strip('，。！？、；：""''（）…— \t\n\r')
    if clean and clean != text:
        pos = script.find(clean)
        if pos >= 0:
            return pos
    s = difflib.SequenceMatcher(None, script, text)
    blocks = s.get_matching_blocks()
    total_matched = sum(b.size for b in blocks)
    if total_matched < len(text) * min_ratio:
        return -1
    for b in blocks:
        if b.size > 0:
            return b.a
    return -1


def build_org_window(
    script: str,
    involved: list[tuple[int, str]],
) -> str | None:
    """构建原文稿对照片段（从匹配位向两侧对齐标点边界，限制扩展范围与总长，合并覆盖所有匹配位）。"""
    positions: list[int] = []
    for _, text in involved:
        pos = match_script_position(script, text)
        if pos >= 0:
            positions.append(pos)
    if not positions:
        return None

    min_pos = min(positions)
    max_pos = max(positions)
    max_len = max(len(t) for _, t in involved)

    # 从匹配位置本身向两侧对齐标点边界（而非从 ctx 切点，避免无界扩展）
    _SENT_END = set("。！？，,")
    MAX_ALIGN = 40   # 单侧最多向边界扩展的字符数
    MAX_WINDOW = 80  # 窗口总长度硬上限

    # 向后对齐：从 min_pos 向前找最近的标点边界
    start = min_pos
    while start > 0 and start > min_pos - MAX_ALIGN and script[start - 1] not in _SENT_END:
        start -= 1

    # 向前对齐：从 max_pos + max_len 向后找最近的标点边界
    end = max_pos + max_len
    while end < len(script) and end < max_pos + max_len + MAX_ALIGN and script[end] not in _SENT_END:
        end += 1
    if end < len(script) and script[end] in _SENT_END:
        end += 1  # 包含句末标点

    # 硬上限：超出则截断
    if end - start > MAX_WINDOW:
        end = start + MAX_WINDOW

    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(script) else ""
    return f"{prefix}{script[start:end]}{suffix}"
