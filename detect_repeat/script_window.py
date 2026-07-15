"""
script_window.py — 原文稿窗口匹配与截取

用于检测模块在发现疑似口吃/切句错误时，从原文稿中截取相关片段，
嵌入 finding 供 LLM 交叉验证。
"""

import difflib


def match_script_position(
    script: str,
    text: str,
    min_ratio: float = 0.4,
    expected_pos: float | None = None,
    lower: int | None = None,
) -> int:
    """在原文稿中查找 text 的最佳匹配位置，返回字符偏移，-1 未找到。

    expected_pos: 期望匹配中心（字符偏移）。当存在多个候选（文本在原文稿中
        重复出现，或模糊匹配到多处）时，优先选择与 expected_pos 最接近者。
        利用句子在原稿中的先后顺序消歧（第 idx 句大致出现在 idx/总句数 处）。
    lower: 软下界（字符偏移）。候选位置不得小于 lower，用于保证「后句必在前句
        之后」。若满足该约束的候选为空，则放宽回全部候选，避免彻底定位不到。
    """
    candidates: list[int] = []

    # 1) 精确子串匹配：收集所有出现位置（文本可能重复出现）
    pos = script.find(text)
    while pos >= 0:
        candidates.append(pos)
        pos = script.find(text, pos + 1)

    # 2) 去标点再匹配（应对 ASR/口播稿标点差异）
    if not candidates:
        clean = text.strip('，。！？、；：""\'\'（）…— \t\n\r')
        if clean and clean != text:
            p = script.find(clean)
            while p >= 0:
                candidates.append(p)
                p = script.find(clean, p + 1)

    # 3) 模糊匹配：用 difflib 收集所有匹配块
    if not candidates:
        s = difflib.SequenceMatcher(None, script, text)
        blocks = s.get_matching_blocks()
        total_matched = sum(b.size for b in blocks)
        if total_matched < len(text) * min_ratio:
            return -1
        candidates = [b.a for b in blocks if b.size > 0]

    if not candidates:
        return -1

    # 顺序约束：优先满足「在后句之前已定位句之后」
    if lower is not None:
        after = [p for p in candidates if p >= lower]
        if after:
            candidates = after

    # 多候选就近消歧（依赖句子顺序的期望位置）
    if expected_pos is not None and len(candidates) > 1:
        return min(candidates, key=lambda p: abs(p - expected_pos))

    return candidates[0]


def build_org_window(
    script: str,
    involved: list[tuple[int, str]],
    n_sentences: int | None = None,
) -> str | None:
    """构建原文稿对照片段（从匹配位向两侧对齐标点边界，限制扩展范围与总长，合并覆盖所有匹配位）。

    n_sentences: 原稿句子总数。给定后用于估算每句的「期望位置」
        (idx/总句数 × 原稿长度)，在文本重复/模糊多匹配时就近消歧；并按句序号
        排序逐句定位，使后句的匹配不得早于前句结束（顺序约束）。
    """
    positions: list[int] = []
    # 按句序号排序，利用句子在原稿中的先后顺序逐句定位（后句必在前句之后）
    lower = None
    matched: list[tuple[int, str, int]] = []  # (idx, text, pos)
    for idx, text in sorted(involved, key=lambda x: x[0]):
        expected = (idx / n_sentences * len(script)) if n_sentences else None
        pos = match_script_position(script, text, expected_pos=expected, lower=lower)
        if pos >= 0:
            matched.append((idx, text, pos))
            lower = pos + len(text)  # 下一句不得早于本句结束
    if not matched:
        return None

    min_pos = min(p for _, _, p in matched)
    max_pos = max(p for _, _, p in matched)

    # 从匹配位置本身向两侧对齐标点边界（而非从 ctx 切点，避免无界扩展）
    # 对齐边界只用句末标点（不含逗号）——否则遇到逗号即停，截不出完整句子；
    # 去掉逗号后窗口可跨越逗号、覆盖整句，给 LLM 更充分的原稿上下文。
    # 注意：右对齐必须从「最右匹配句起点 max_pos」向前找第一个句号，而不能用
    # pos+len(text) 这类易越界的值——一旦越过句号，就会从句号之后继续扫到更后文。
    _SENT_END = set("。！？")
    MAX_ALIGN = 80   # 单侧最多向边界扩展的字符数
    MAX_WINDOW = 200 # 窗口总长度硬上限

    # 向后对齐：从 min_pos 向前找最近的标点边界
    start = min_pos
    while start > 0 and start > min_pos - MAX_ALIGN and script[start - 1] not in _SENT_END:
        start -= 1

    # 向前对齐：从 max_pos 向后找最近的句末标点（即最右匹配句自身的句号）
    end = max_pos
    while end < len(script) and end < max_pos + MAX_ALIGN and script[end] not in _SENT_END:
        end += 1
    if end < len(script) and script[end] in _SENT_END:
        end += 1  # 包含句末标点

    # 右侧尾随上下文：句号之后若立即收尾，LLM 看不到片段之后的叙事延续。
    # 再向后补充一小段（到下个句号为止，最多 RIGHT_TRAIL 字），让窗口右缘更自然；
    # 仍以 MAX_WINDOW 封顶，不会一路涨回更后的段落（如「赚七成收益」）。
    RIGHT_TRAIL = 45
    trail = end
    while trail < len(script) and trail < end + RIGHT_TRAIL and script[trail] not in _SENT_END:
        trail += 1
    if trail < len(script) and script[trail] in _SENT_END:
        trail += 1
    if trail - start <= MAX_WINDOW:
        end = trail

    # 硬上限：超出则截断
    if end - start > MAX_WINDOW:
        end = start + MAX_WINDOW

    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(script) else ""
    return f"{prefix}{script[start:end]}{suffix}"


def build_short_org_window(
    script: str,
    sentences: list,
    sent_idx: int,
) -> str | None:
    """为极短句(≤2字)构建原文稿窗口。

    极短句自身文本难以在长原稿中可靠定位（易误匹配到别处的同字），故改用其
    **前后长句**夹出窗口。在待研判句左右两侧各取「最近的可靠长邻居」(长度>2)
    一同交给 build_org_window 定位——窗口被前后邻居同时界定，避免只靠单侧锚点
    时另一侧飘到很远（如跨过整个段落直到下一个句号）。

    例：句31「寄到」/ 句32「县」夹在句30「…琢磨明白」与句33「…记到现在」之间，
    窗口应聚焦在「…震住了，记到现在。」而非延伸到更后的「复盘…赚七成收益」。
    """
    if not script:
        return None
    # 左右各取最近可靠长邻居（>2字）；极短邻居本身不可靠，跳过。
    # left：在 sent_idx 左侧，idx 越大越近；right：在 sent_idx 右侧，idx 越小越近。
    left = right = None
    for s in sorted(sentences, key=lambda x: x["idx"]):
        t = (s["text"] or "").strip()
        if not t or len(t) <= 2:
            continue
        if s["idx"] < sent_idx:
            left = (s["idx"], t)        # 不断刷新为更近者
        elif s["idx"] > sent_idx:
            right = (s["idx"], t)       # 第一个遇到的即最近的，直接收尾
            break
    involved = [n for n in (left, right) if n is not None]
    if not involved:
        return None
    return build_org_window(script, involved, n_sentences=len(sentences))


# finding 中可能携带的 (idx 字段, 文本字段) 对，用于自动抽取原稿定位句。
_INVOLVED_PAIRS = (
    ("sent_a_idx", "text_a"),
    ("sent_b_idx", "text_b"),
    ("sent_c_idx", "text_c"),
    ("mid_sent_idx", "mid_text"),
    ("sent_idx", "text"),
    ("next_sent_idx", "next_sent_text"),
)


def _involved_from_finding(finding: dict) -> list[tuple[int, str]]:
    """从 finding 的常见字段自动抽取 (idx, text) 列表，供 build_org_window 定位原稿。"""
    out: list[tuple[int, str]] = []
    for idx_key, text_key in _INVOLVED_PAIRS:
        if idx_key in finding and text_key in finding and finding[text_key]:
            out.append((finding[idx_key], finding[text_key]))
    return out


def attach_org_window(
    finding: dict, script: str, sentences: list, short: bool | None = None,
) -> dict:
    """给 finding 追加 `org_script_window` 字段（原稿窗口的唯一挂载入口）。

    统一在 detect_loop 里对送研判的 findings 批量调用本函数，各 detect 模块
    本身不再挂窗口，也不再直接调用 build_org_window / build_short_org_window。

    short:
      - None（默认）：自动判断——当 finding 只有单个定位句且其文本 ≤2 字
        （极短孤立句 / 孤立编号）时走 short 逻辑，否则走全窗口。
      - True：强制用 build_short_org_window（前后长邻居夹窗口，适合极短单句）。
      - False：强制用 build_org_window 全窗口（句间/跨句重叠类 finding）。

    定位失败（原稿为空 / 文本无匹配）则不挂载字段，保持上游不变。
    """
    if not script:
        return finding
    if short is None:
        # 单句 finding（仅 sent_idx、无 next/pair 定位字段）且文本极短(≤2字，含空)
        # → 自身难定位，走 short（前后长邻居夹窗口）；否则走全窗口。
        # 注意：极短句 text 可能为空串，不能依赖 _involved_from_finding（空串被
        # 视为无效定位句而丢弃），故此处直接看 text 长度与是否为单句。
        _multi = any(
            k in finding for k in
            ("next_sent_idx", "sent_a_idx", "sent_b_idx", "sent_c_idx", "mid_sent_idx")
        )
        short = (not _multi) and len(finding.get("text", "")) <= 2
    if short:
        sent_idx = finding.get("sent_idx")
        if sent_idx is None:
            return finding
        window = build_short_org_window(script, sentences, sent_idx)
    else:
        involved = _involved_from_finding(finding)
        if not involved:
            return finding
        window = build_org_window(script, involved, n_sentences=len(sentences))
    if window:
        finding["org_script_window"] = window
    return finding
