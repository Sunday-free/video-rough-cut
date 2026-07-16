"""
script_window.py — 原文稿窗口匹配与截取

用于检测模块在发现疑似口吃/切句错误时，从原文稿中截取相关片段，
嵌入 finding 供 LLM 交叉验证。
"""

import difflib


_PUNCT = set("，。！？、；：\"'（）…— \t\n\r")


def _depunctuate(s: str) -> str:
    """去除标点/空白，仅保留语义字符。"""
    return "".join(ch for ch in s if ch not in _PUNCT)


def _depunctuate_map(s: str) -> tuple[str, list[int]]:
    """返回 (去标点字符串, mapping)，mapping[i] 为 clean[i] 在原稿中的偏移。"""
    out: list[str] = []
    mapping: list[int] = []
    for i, ch in enumerate(s):
        if ch not in _PUNCT:
            out.append(ch)
            mapping.append(i)
    return "".join(out), mapping


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

    # 2) 候选去标点再匹配（应对 ASR/口播稿标点差异）
    if not candidates:
        clean = _depunctuate(text)
        p = script.find(clean)
        while p >= 0:
            candidates.append(p)
            p = script.find(clean, p + 1)

    # 2.5) 双向去标点匹配：原稿也去标点后再找，应对【原稿内部标点差异】
    #      （如原稿「二、人工智能、新能源车」vs 候选「二人工智能新能源车」——
    #      仅候选去标点时因内部 、 对不上会退化成模糊匹配、从中间开始，导致定位右移）。
    if not candidates:
        clean = _depunctuate(text)
        if clean:
            clean_script, mapping = _depunctuate_map(script)
            p = clean_script.find(clean)
            while p >= 0:
                orig = mapping[p] if p < len(mapping) else -1
                if orig >= 0:
                    candidates.append(orig)
                p = clean_script.find(clean, p + 1)

    # 2.6) 候选【前缀】匹配：口播/残句常与原文稿仅【前缀】一致（后半改写或截断，
    #      如候选「…也是接下来的」vs 原稿「…也是接下来我们要…」），整串匹配不到。
    #      用去标点后的候选前缀在原稿去标点文本中定位，取最长可匹配前缀 → 拿到候选句首。
    if not candidates:
        clean = _depunctuate(text)
        if clean and len(clean) >= 6:
            clean_script, mapping = _depunctuate_map(script)
            for L in range(len(clean), 5, -1):
                pref = clean[:L]
                p = clean_script.find(pref)
                if p >= 0:
                    while p >= 0:
                        orig = mapping[p] if p < len(mapping) else -1
                        if orig >= 0:
                            candidates.append(orig)
                        p = clean_script.find(pref, p + 1)
                    break

    # 3) 模糊匹配：用 difflib 找「最大连续匹配块」作为锚点
    #    口语与原稿有差异时长句会散出多块，必须取【最大块】而非「最接近估算点」，
    #    否则会误锚到凑巧靠近的无关片段，导致窗口左/右边界错位（左邻居被截断）。
    if not candidates:
        s = difflib.SequenceMatcher(None, script, text)
        blocks = s.get_matching_blocks()
        good = [b for b in blocks if b.size >= max(2, int(len(text) * min_ratio))]
        if not good:
            return -1
        # 顺序约束：优先满足「后句必在前句之后」（仅当存在满足约束的块时生效）
        if lower is not None:
            after = [b for b in good if b.a >= lower]
            if after:
                good = after
        # 主选最大连续匹配块；并列时取最接近 expected_pos 者
        best = max(
            good,
            key=lambda b: (b.size, -abs(b.a - expected_pos) if expected_pos is not None else 0),
        )
        return best.a

    if not candidates:
        return -1

    # 顺序约束：优先满足「在后句之前已定位句之后」（精确/去标点匹配走此路径）
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
    max_window: int = 100,
    focus_idx: int | None = None,
) -> str | None:
    """构建原文稿对照片段（从匹配位向两侧对齐标点边界，限制扩展范围与总长，合并覆盖所有匹配位）。

    n_sentences: 原稿句子总数。给定后用于估算每句的「期望位置」
        (idx/总句数 × 原稿长度)，在文本重复/模糊多匹配时就近消歧；并按句序号
        排序逐句定位，使后句的匹配不得早于前句结束（顺序约束）。
    """
    # 按句序号排序，利用句子在原稿中的先后顺序逐句定位（后句必在前句之后）
    lower = None
    matched: list[tuple[int, str, int, bool]] = []  # (idx, text, pos, is_real)
    for idx, text in sorted(involved, key=lambda x: x[0]):
        expected = (idx / n_sentences * len(script)) if n_sentences else None
        pos = match_script_position(script, text, expected_pos=expected, lower=lower)
        if pos >= 0:
            matched.append((idx, text, pos, True))
            lower = pos + len(text)  # 下一句不得早于本句结束
        elif expected is not None and expected > 0:
            # 文本匹配失败（ASR 噪音/口误），先用 idx 估算原稿位置占位；
            # 真正定位交给下方 focus 兜底：用前后已匹配邻居句插值。
            est = int(expected)
            if lower is not None and est < lower:
                est = lower  # 不早于已定位的前句
            matched.append((idx, text, est, False))
            lower = est + len(text)
    if not matched:
        return None

    min_pos = min(p for _, _, p, _ in matched)
    max_pos = max(p for _, _, p, _ in matched)

    # ── 聚焦位置（问题句中心）────────────────────────────────────────────
    # 优先用 focus_idx 对应句的直接匹配位置；若问题句在原稿匹配不到
    # （ASR 噪音/口误），则用其前后【已成功匹配】的邻居句位置插值估计——
    # 即「问题句定位不到原稿 → 由前后句定位」，最终仍保证它落在窗口中间。
    # 缺省（未指定 focus_idx，如多句比对）取匹配跨度的几何中点。
    focus = None
    p_start: int = 0
    p_end: int = len(script)
    prob_len = 0
    if focus_idx is not None:
        real_left = real_right = None
        for idx, text, pos, is_real in matched:
            if idx == focus_idx:
                if is_real:
                    p_start, p_end = pos, pos + len(text)
                    focus = (p_start + p_end) // 2
                prob_len = len(text)
            elif is_real:
                if idx < focus_idx and (real_left is None or idx > real_left[0]):
                    real_left = (idx, pos, text)
                elif idx > focus_idx and (real_right is None or idx < real_right[0]):
                    real_right = (idx, pos, text)
        if focus is None:
            # 问题句未直接匹配：用前后已匹配邻居句位置插值估计。
            # 注意用「左邻居句尾 → 右邻居句首」之间插值（而非两句首），
            # 否则长左邻居会把估计点拉进其内部，导致问题句不居中。
            if real_left and real_right:
                li, lp, lt = real_left
                ri, rp, _ = real_right
                left_end = lp + len(lt)
                ratio = (focus_idx - li) / (ri - li)
                fpos = int(left_end + (rp - left_end) * ratio)
            elif real_left:
                fpos = real_left[1] + len(real_left[2])
            elif real_right:
                fpos = real_right[1]
            else:
                fpos = (min_pos + max_pos) // 2
            p_start, p_end = fpos, fpos + max(prob_len, 1)
            focus = (p_start + p_end) // 2
    if focus is None:
        focus = (min_pos + max_pos) // 2

    # 是否需要对「问题句本体」做保护（不切入本句）：仅当明确传了 focus_idx 时。
    # focus_idx=None（如多句比对 / build_short_org_window 借邻居夹窗）时，
    # p_start/p_end 只是 0/len(script) 的占位默认值，绝不能据此钳制。
    protect = focus_idx is not None

    # ── 以问题句为中心、左右均衡截取 ─────────────────────────────────────
    # 窗口以 focus 为中心各取半窗（half = max_window//2），并向句末标点对齐；
    # 对齐只允许在「半窗范围内」扩展、绝不越过 focus±half，从而保证问题句居中；
    # 最后强制平衡（两侧扩展等长、上限 half），左右用「…」收尾。
    _SENT_END = set("。！？")
    half = max_window // 2

    # 左边界：以 focus-half 为基准，向左（最多一整个 half）对齐到最近句末标点；
    # 若半窗内无句末，则直接截断在 focus-half（不越过，保证居中）。
    lo = max(0, focus - 2 * half)
    ls = focus - half
    while ls > lo and script[ls - 1] not in _SENT_END:
        ls -= 1
    start = ls if (ls > lo and script[ls - 1] in _SENT_END) else focus - half

    # 右边界：以 focus+half 为基准，向右（最多一整个 half）对齐到最近句末标点（含该标点）；
    # 若半窗内无句末，则直接截断在 focus+half。
    hi = min(len(script), focus + 2 * half)
    rs = focus + half
    while rs < hi and script[rs] not in _SENT_END:
        rs += 1
    end = (rs + 1) if (rs < len(script) and script[rs] in _SENT_END) else focus + half

    # 平衡 + 保护问题句完整可见：窗口必须覆盖整个 [p_start, p_end]（问题句本体），
    # 但两侧扩展以较长侧为基准、上限 half，保证问题句居中、左右均衡。
    left_ext = focus - start
    right_ext = end - focus
    target = min(max(left_ext, right_ext), half)
    ns = max(0, focus - target)
    if protect and ns > p_start:           # 不得切入问题句左侧
        ns = p_start
    ne = min(len(script), focus + target)
    if protect and ne < p_end:             # 不得切入问题句右侧
        ne = p_end
    start, end = ns, ne

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


def _strip_nl(s: str | None) -> str | None:
    """去掉字符串中的换行符，None 透传。"""
    if s is None:
        return None
    return s.replace("\n", "").replace("\r", "")


def get_org_script_window(
    script: str,
    sentences: list,
    finding: dict,
    short: bool | None = None,
    focus_idx: int | None = None,
) -> str | None:
    """计算原稿窗口并直接返回（原稿窗口的唯一入口）。

    统一在 detect_loop 里对送研判的 findings 批量调用本函数，各 detect 模块
    本身不再挂窗口，也不再直接调用 build_org_window / build_short_org_window。

    short:
      - None（默认）：自动判断——当 finding 只有单个定位句且其文本 ≤2 字
        （极短孤立句 / 孤立编号）时走 short 逻辑，否则走全窗口。
      - True：强制用 build_short_org_window（前后长邻居夹窗口，适合极短单句）。
      - False：强制用 build_org_window 全窗口（句间/跨句重叠类 finding）。

    max_window 内部自适应：至少 100，如果定位句较长则按 2 倍扩展。
    定位失败（原稿为空 / 文本无匹配）返回 None。
    """
    if not script:
        return None
    if short is None:
        _multi = any(
            k in finding for k in
            ("next_sent_idx", "sent_a_idx", "sent_b_idx", "sent_c_idx", "mid_sent_idx")
        )
        short = (not _multi) and len(finding.get("text", "")) <= 2
    if short:
        sent_idx = finding.get("sent_idx")
        if sent_idx is None:
            return None
        return _strip_nl(build_short_org_window(script, sentences, sent_idx))
    else:
        involved = _involved_from_finding(finding)
        if not involved:
            return None
        # 补充前后邻居句辅助定位：当前句文本可能为 ASR 噪音无法匹配原稿，
        # 但邻居句可以匹配，通过邻居句在原稿中的位置间接定位当前句。
        involve_set = {i for i, _ in involved}
        sent_idx_map = {s["idx"]: (s.get("text", "") or "").strip() for s in sentences}
        _sorted = sorted(involved, key=lambda x: x[0])
        min_i, max_i = _sorted[0][0], _sorted[-1][0]
        max_sid = max(sent_idx_map.keys()) if sent_idx_map else max_i
        # 向左取最近的非空邻居（>2字）
        i = min_i - 1
        while i >= 0:
            t = sent_idx_map.get(i, "")
            if t and i not in involve_set and len(t) > 2:
                involved.append((i, t))
                involve_set.add(i)
                break
            i -= 1
        # 向右取最近的非空邻居（>2字）
        i = max_i + 1
        while i <= max_sid:
            t = sent_idx_map.get(i, "")
            if t and i not in involve_set and len(t) > 2:
                involved.append((i, t))
                involve_set.add(i)
                break
            i += 1
        dyn_max = max(50, max(len(t) for _, t in involved) * 2)
        return _strip_nl(build_org_window(
            script, involved, n_sentences=len(sentences), max_window=dyn_max,
            focus_idx=focus_idx,
        ))
