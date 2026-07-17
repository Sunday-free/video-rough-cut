"""
script_window.py — 原文稿窗口匹配与截取

用于检测模块在发现疑似口吃/切句错误时，从原文稿中截取相关片段，
嵌入 finding 供 LLM 交叉验证。
"""

import difflib

from speech_error_detector.config import MIN_RATIO, MIN_SCRIPT_DYN_MAX


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
    min_ratio: float = MIN_RATIO,
    expected_pos: float | None = None,
    lower: int | None = None,
    debug: bool = False,
):
    """在原文稿中查找 text 的最佳匹配位置，返回字符偏移，-1 未找到。

    expected_pos: 期望匹配中心（字符偏移）。当存在多个候选（文本在原文稿中
        重复出现，或模糊匹配到多处）时，优先选择与 expected_pos 最接近者。
        利用句子在原稿中的先后顺序消歧（第 idx 句大致出现在 idx/总句数 处）。
    lower: 软下界（字符偏移）。候选位置不得小于 lower，用于保证「后句必在前句
        之后」。若满足该约束的候选为空，则放宽回全部候选，避免彻底定位不到。
    debug: 为 True 时返回 (pos, info_dict) 而非仅 int，便于诊断每层匹配结果。
    """
    info: dict = {"layers": [], "chosen": None}
    candidates: list[int] = []

    def _ret(pos):
        if debug:
            return pos, info
        return pos

    # 1) 精确子串匹配：收集所有出现位置（文本可能重复出现）
    pos = script.find(text)
    while pos >= 0:
        candidates.append(pos)
        pos = script.find(text, pos + 1)
    info["layers"].append(("1_exact", list(candidates)))

    # 2) 候选去标点再匹配（应对 ASR/口播稿标点差异）
    if not candidates:
        clean = _depunctuate(text)
        p = script.find(clean)
        while p >= 0:
            candidates.append(p)
            p = script.find(clean, p + 1)
        info["layers"].append(("2_depunct", list(candidates)))

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
        info["layers"].append(("2.5_both_depunct", list(candidates)))

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
                    info["2.6_prefix"] = pref
                    break
        info["layers"].append(("2.6_prefix", list(candidates)))

    # 3) 模糊匹配：锚点投票 + 对齐相似度打分（通用，无需任何字符级特殊映射）
    #    原理：difflib 的每个匹配块 (a, b, size) 表示 script[a:a+size]==text[b:b+size]。
    #    若整句对齐到原稿起点 S，则该块满足 a-b≈S。字符「替换」（异体字/口误/谐音，
    #    如口播「响」↔原稿「喺」）是等长替换，不改变 a-b 偏移；口语填充词造成的「插入」
    #    只会让 a-b 轻微漂移。因此：
    #      (1) 把每个块隐含的句首 S=a-b 收作候选起点；
    #      (2) 对每个候选起点，用 script[S:S+len(text)] 与 text 的真实对齐相似度打分；
    #      (3) 取相似度最高者为锚点（并列时用顺序约束 lower 与期望位 expected_pos 消歧）。
    #    这样即使句中若干字被替换/漏读，剩余匹配块仍会共同指向同一真实起点，
    #    天然容忍粤语异体字与口误，而无需维护「响→喺」之类的硬编码词表。
    if not candidates:
        # autojunk=False：关闭「高频字符当 junk」启发式。原稿 >200 字符时 difflib
        # 默认会把出现 >1% 的常用字（的/一/不/是…）判为 junk 不参与匹配，导致对齐
        # 相似度被系统性压低、含大量常用字的句子更容易被误判为匹配不足而丢锚。中文
        # 场景下必须关掉，否则锚点召回率下降、窗口退化成整篇原稿。
        sm = difflib.SequenceMatcher(None, script, text, autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size >= 2]
        if not blocks:
            return _ret(-1)
        cand_starts: set[int] = set()
        for b in blocks:
            s0 = b.a - b.b
            cand_starts.add(s0 if s0 >= 0 else b.a)

        def _score(S: int) -> float:
            seg = script[S:S + len(text)]
            return difflib.SequenceMatcher(None, seg, text, autojunk=False).ratio()

        scored = [(S, _score(S)) for S in cand_starts if 0 <= S < len(script)]
        if not scored:
            return _ret(-1)

        def _key(item: tuple[int, float]):
            S, r = item
            after = 1 if (lower is None or S >= lower) else 0
            near = -abs(S - expected_pos) if expected_pos is not None else 0
            # 相似度优先（量化到 3 位小数使近似并列可比），再看顺序约束，最后靠期望位
            return (round(r, 3), after, near)

        best_S, best_r = max(scored, key=_key)
        info["layers"].append(
            ("3_fuzzy", sorted(scored, key=lambda x: -x[1])[:5])
        )
        # 可靠性：最佳对齐相似度达到阈值即认为起点可信（供锚点/邻居筛选使用）
        info["fuzzy_is_prefix"] = best_r >= min_ratio
        info["chosen"] = ("3_fuzzy", best_S)
        return _ret(best_S)

    if not candidates:
        return _ret(-1)

    # 顺序约束：优先满足「在后句之前已定位句之后」（精确/去标点匹配走此路径）
    if lower is not None:
        after = [p for p in candidates if p >= lower]
        if after:
            candidates = after
        else:
            # lower 约束下无候选：不静默返回最早候选（会跳回前段），
            # 改为交给下方「多候选就近消歧」按 expected_pos 选。
            info["lower_failed"] = True

    # 多候选就近消歧（依赖句子顺序的期望位置）
    if expected_pos is not None and len(candidates) > 1:
        chosen = min(candidates, key=lambda p: abs(p - expected_pos))
        info["chosen"] = ("nearest_expected", chosen)
        return _ret(chosen)

    info["chosen"] = ("first", candidates[0])
    return _ret(candidates[0])


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
        # 匹配失败（ASR 噪音 / 口误 / 口播跑题）：不再用 idx/总句数 退化估算位置
        # （旧逻辑会把窗口锚到与原稿毫不相干的「同序号附近」片段，误导 LLM）。
        # 该句直接放弃定位，交由下方「无任何真实匹配 → 返回完整原稿」兜底。
    if not matched:
        # 所有 involved 句均未能在原稿定位（五层匹配全失败）：不猜测位置，
        # 直接返回完整原稿，交由 LLM 自行对照全文。
        return script

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
    return _window_around_center(
        script, focus, max_window=max_window,
        p_start=p_start, p_end=p_end, protect=protect,
    )


def _window_around_center(
    script: str,
    focus: int,
    max_window: int = 100,
    p_start: int = 0,
    p_end: int = 0,
    protect: bool = False,
) -> str:
    """以 focus 为中心向两侧均衡截取原文稿窗口，对齐句末标点，加 … 收尾。

    窗口以 focus 为中心各取半窗（half = max_window//2），并向句末标点对齐；
    对齐只允许在「半窗范围内」扩展、绝不越过 focus±half，从而保证焦点居中；
    最后强制平衡（两侧扩展等长、上限 half），左右用「…」收尾。

    protect / p_start / p_end：当保护问题句本体（focus_idx 明确传入）时，窗口
    两侧不得切入 [p_start, p_end]（问题句本体）。
    """
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
    # 降级用左右长邻居同时定位时，故意**不受 dyn_max 控制**（不传 max_window）：
    # 该路径旨在用两侧邻居把问题句夹出来，窗口应自由覆盖到左、右邻居的真实落点，
    # 而非被 full 路径的 dyn_max=max(50, 最长句*2) 截断。
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


def _build_anchors(script: str, sentences: list) -> dict:
    """为每句在原稿中做高置信匹配，返回 {idx: pos}（pos = 该句在原稿中的起点）。

    仅接受「起点可靠」的匹配：
      - 精确 / 去标点 / 双向去标点 / 前缀匹配：均以句首为锚，起点即真实起点；
      - 模糊匹配中「从口播句开头命中」(b.b == 0, size>=4) 的块：起点可靠；
      - 模糊匹配中仅中段片段命中 (b.b > 0) 的块：起点不对应句子开头，不可信，拒绝。
    最后做单调性过滤：按 idx 升序要求 pos 非递减（原稿保持句序），剔除乱序错锚。
    """
    anchors: dict = {}
    for s in sentences:
        idx = s.get("idx")
        t = (s.get("text") or "").strip()
        if not t or len(t) <= 2:
            continue
        # expected_pos 按句序估算（idx/总句数 × 原稿长度），用于在文本于原稿中
        # 多处重复出现时就近消歧（如口头禅同时出现在句 5 与句 80），避免错锚到
        # 最早出现处；同时作为模糊层并列时的次序 tiebreaker。
        expected = (idx / len(sentences) * len(script)) if sentences else None
        pos, info = match_script_position(
            script, t, expected_pos=expected, debug=True,
        )
        if pos < 0:
            continue
        chosen = info.get("chosen")
        layer = chosen[0] if chosen else None
        # 模糊层接受条件：最佳对齐相似度达到阈值即认为起点可信（fuzzy_is_prefix 现
        # 语义= best_r >= min_ratio，而非旧版的「必须从句首命中 b.b==0」）。即便命中
        # 块位于句中（b.b>0），只要整段对齐相似度够高，投票+打分已把候选起点收敛到
        # 真实的句首 S，起点依然可靠，故此处仅以相似度阈值把关，不再额外要求前缀块。
        if layer == "3_fuzzy" and not info.get("fuzzy_is_prefix"):
            continue
        # 其余（精确 / 去标点 / 双向去标点 / 前缀匹配 → chosen 标签为 'first'，
        # 或模糊匹配中从句首命中的前缀块）起点均可靠，作为锚点。
        anchors[idx] = pos

    # 单调性过滤：原稿保持句序，故 pos 应随 idx 非递减。用「最长非降子序列(LIS)」
    # 取最大的一致锚点链，剔除离群错锚——关键：单个残句错锚（如短残句「哎呀呢度」的
    # 「呢度」凑巧命中原稿结尾）只是链外离群点，会被 LIS 排除，而不会像贪心过滤那样
    # 污染 last 值、把它之后所有正确锚点连带丢弃。
    items = sorted(anchors.items())  # [(idx, pos), ...] 按 idx 升序
    n = len(items)
    if n == 0:
        return {}
    dp = [1] * n          # dp[i]: 以 i 结尾的最长非降链长度
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if items[j][1] <= items[i][1] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                prev[i] = j
    best_i = max(range(n), key=lambda i: dp[i])
    keep: set[int] = set()
    k = best_i
    while k != -1:
        keep.add(k)
        k = prev[k]
    return {items[i][0]: items[i][1] for i in range(n) if i in keep}


def _window_for_focus(
    script: str,
    anchors: dict,
    focus_idx: int,
    focus_text: str = "",
) -> str:
    """用已匹配的最近左/右邻居把焦点句夹在中间，取中心截窗口。

    焦点句自身能高置信匹配 → 以自身匹配点为窗口中心；
    否则取「最近左邻居起点」与「最近右邻居起点」之间中点为中心。
    窗口半径按句长自适应。
    """
    focus = anchors.get(focus_idx)
    left_idx = max((i for i in anchors if i < focus_idx), default=None)
    right_idx = min((i for i in anchors if i > focus_idx), default=None)
    if focus is not None:
        center = focus
    elif left_idx is not None and right_idx is not None:
        center = (anchors[left_idx] + anchors[right_idx]) // 2
    elif left_idx is not None:
        center = anchors[left_idx] + max(30, len(focus_text) // 2)
    elif right_idx is not None:
        center = anchors[right_idx] - max(30, len(focus_text) // 2)
    else:
        return script  # 无任何锚点，退回整篇
    max_window = max(90, len(focus_text) * 2) if focus_text else 150
    return _window_around_center(script, center, max_window=max_window)


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
        # 单句焦点（明确给了 focus_idx）：用「全体句子高置信锚点 + 最近左/右邻居夹窗」
        # 定位——稳健且不受焦点句自身能否匹配影响（即使焦点句是残句/口误，只要左右邻居
        # 正确匹配，窗口就落在正确段落）。多句比对（未给 focus_idx）才走下方旧逻辑。
        if focus_idx is not None:
            anchors = _build_anchors(script, sentences)
            focus_text = (finding.get("text") or "").strip()
            return _strip_nl(_window_for_focus(script, anchors, focus_idx, focus_text))
        # 多句比对：补充前后邻居句辅助定位。当前句文本可能为 ASR 噪音无法匹配原稿，
        # 但（未被改写的）邻居句可以匹配，通过它的原稿位置间接定位当前句。
        # 邻居若被污染（五层匹配全失败，或非前缀模糊错锚）则跳过，继续向外找最近的可匹配邻居；
        # 每侧最多找 NEIGHBOR_DEPTH 层。若两侧都找不到可匹配邻居，
        # 则该 finding 在 build_org_window 中 matched 为空 → 退化返回完整原稿。
        involve_set = {i for i, _ in involved}
        sent_idx_map = {s["idx"]: (s.get("text", "") or "").strip() for s in sentences}
        _sorted = sorted(involved, key=lambda x: x[0])
        min_i, max_i = _sorted[0][0], _sorted[-1][0]
        max_sid = max(sent_idx_map.keys()) if sent_idx_map else max_i
        NEIGHBOR_DEPTH = 3
        # 焦点期望位置：用于过滤「匹配到了但定位离焦点过远」的误锚邻居
        # （如被污染的短句 fuzzy 命中到原稿另一端），避免把窗口拽到毫不相干的位置。
        focus_expected = (min_i / len(sentences) * len(script)) if sentences else None
        pos_tol = int(len(script) * 0.3)

        def _pick_matching_neighbor(start: int, step: int) -> tuple[int, str] | None:
            """从 start 沿 step 方向向外找最近一个「能匹配原稿且定位邻近焦点」的邻居（>2字）。
            被污染的邻居（匹配不到 / 非前缀模糊错锚 / 定位离焦点过远）直接跳过，继续向外；
            最多找 NEIGHBOR_DEPTH 层。
            """
            for _ in range(NEIGHBOR_DEPTH):
                start += step
                if start < 0 or start > max_sid:
                    return None
                t = sent_idx_map.get(start, "")
                if t and start not in involve_set and len(t) > 2:
                    pos, info = match_script_position(script, t, debug=True)
                    if pos < 0:
                        continue  # 邻居被污染（匹配不到）：跳过，向外找下一层
                    if info.get("chosen") and info["chosen"][0] == "3_fuzzy" \
                            and not info.get("fuzzy_is_prefix"):
                        continue  # 非前缀模糊块起点不可信：跳过
                    if focus_expected is not None and abs(pos - focus_expected) > pos_tol:
                        # 匹配到了，但定位离焦点期望位置过远 → 视为误锚，跳过
                        continue
                    return start, t
            return None

        pick = _pick_matching_neighbor(min_i, -1)
        if pick:
            involved.append(pick)
            involve_set.add(pick[0])
        pick = _pick_matching_neighbor(max_i, 1)
        if pick:
            involved.append(pick)
            involve_set.add(pick[0])
        dyn_max = max(MIN_SCRIPT_DYN_MAX, max(len(t) for _, t in involved) * 2)
        return _strip_nl(build_org_window(
            script, involved, n_sentences=len(sentences), max_window=dyn_max,
            focus_idx=focus_idx,
        ))
