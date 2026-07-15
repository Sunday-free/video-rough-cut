"""
script_faithful.py — 原稿窗口内重复排除（忠于原稿/排比的共享 helper）

用途：在检测/审查阶段，把「待删除段」与原稿（作者原稿）做比对，
      若原稿里也存在等价的重复（如序号枚举、并列排比），说明该删除段
      「忠于原稿」，并非口误赘余，应排除（不删 / 不自动确认）。

设计要点：
  1. 只活在 Python 侧比对，不改任何 sentences 数据、不进 LLM prompt。
  2. 数字对齐用增强版 normalize_numerals_full（十八==18、两==2、十幾==10几），
     旧 normalize_numerals（单字版）不动，故 detector / prompt 零影响。
  3. 计数分两侧（均为**局部**比对，避免全篇多算/稀释）：
       - ASR 域：只数本 finding 涉及的句子/句对（intra/fragment=整句 text，
         inter=sent_a+sent_b），而非全篇口播。
       - 原稿域：优先用本句对应的原稿局部窗口 org_script_window（≤200 字），
         不再数整篇原稿；fragment/inter 用「边界感知」计数（待删段后须跟标点/句末
         才算一次忠于原稿的命中，解决 false-start『二人工智能』后跟『新』→不计数→照删）；
         intra 用朴素计数（句内重复原稿多为合法并列）。
       - 局部比对的意义：原稿别处的高频重复不会被「多算」进来稀释本句的口误，
         从而避免把真口误误判为「忠于原稿/排比」而 skip。
  4. 字段兼容：同时支持 LLM 检测器字段（text/head_char_len/text_a/hits）
     与 mechanical_seed 字段（delete_text/delete_sentence_idx，inter 类 delete_text 为空）。
       机械种子缺 text 字段时 local_asr 为空，n_asr 回退全篇 domain_text；n_org 严格只用
       org_script_window（≤200 字），窗口缺失则直接不跳（保守，交 LLM 研判），不再回退整篇
       original_script（否则原稿别处高频重复被多算、稀释本句口误→误 skip）。

排除条件：n_org(局部窗口) >= n_asr(局部句子/句对) >= 1  → 忠于原稿/排比，排除。
"""

from speech_error_detector.detect_repeat import normalize_numerals_full

# 边界感知计数用的「句读/边界」字符集（命中后跟这些才算一次忠于原稿）
_BOUNDARY_CHARS = set("，。！？、；：…—,.!?;: \t\n\r（）()《》<>“”‘’\"'")


# ============================================================
#  计数原语
# ============================================================

def _count_nonoverlap(text: str, sub: str) -> int:
    """统计 sub 在 text 中的非重叠出现次数。"""
    if not sub or not text:
        return 0
    cnt = 0
    start = 0
    L = len(sub)
    while True:
        idx = text.find(sub, start)
        if idx == -1:
            break
        cnt += 1
        start = idx + L
    return cnt


def _count_boundary(original: str, sub: str) -> int:
    """边界感知计数：sub 在 original 中每次出现，其结尾须紧跟句读/边界/句末
    才算一次『忠于原稿』命中。用于 fragment/inter（待删段应为独立成读的短语）。"""
    if not sub or not original:
        return 0
    cnt = 0
    start = 0
    L = len(sub)
    n = len(original)
    while True:
        idx = original.find(sub, start)
        if idx == -1:
            break
        end = idx + L
        if end >= n or original[end] in _BOUNDARY_CHARS:
            cnt += 1
        start = idx + L
    return cnt


# ============================================================
#  待删段抽取（字段兼容）
# ============================================================

def _extract_deleted_segment(f: dict, sentences: list[dict]) -> tuple:
    """从 finding/issue 抽取待删段 p 及计数模式。

    Returns:
        (seg, use_boundary)
          seg: str（单段）或 list[str]（intra 逐 hit）或 None
          use_boundary: 原稿侧是否用边界感知计数
                         （fragment/inter/mechanical 非 intra = True；intra = False）
    """
    # ---- mechanical_seed（dimension 字段，带 mechanical 标记）----
    if f.get("mechanical"):
        dim = f.get("dimension")
        delete_text = f.get("delete_text") or ""
        if delete_text:
            # intra_repeat 类：delete_text 是重叠段，用朴素计数
            # fragment/inter 类：delete_text 是整句/短句，用边界感知
            return delete_text, (dim != "intra_repeat")
        # inter 包含类 delete_text 为空 → 用 delete_sentence_idx 取整句文本当 p
        didx = f.get("delete_sentence_idx")
        if didx is not None:
            for s in sentences:
                if s.get("idx") == didx:
                    return s.get("text", ""), True
        return None, True

    # ---- LLM 检测器（type 字段）----
    typ = f.get("type") or ""
    if typ == "fragment":
        head = f.get("head_char_len", 0) or 0
        txt = f.get("text", "") or ""
        if head > 0:
            return txt[head:], True       # keep_head 删尾部
        return txt, True                  # 整句删
    if typ == "inter_repeat":
        return (f.get("text_a") or ""), True
    if typ == "intra_repeat":
        hits = f.get("hits") or []
        phrases = [h.get("phrase", "") for h in hits if h.get("phrase")]
        return phrases, False
    return None, True


# ============================================================
#  单段判定
# ============================================================

def _local_asr_text(f: dict) -> str:
    """抽取 finding 对应的**局部** ASR 文本，供 n_asr 做本地化计数（避免数全篇口播
    被无关句子稀释/放大）。

      - inter_repeat：sent_a + sent_b（+sent_c）文本（重复发生在句对之间）
      - intra_repeat / fragment：整句 text
      - 其它（如机械种子缺 text 字段）：返回空串，调用方回退到全篇 domain_text
    """
    typ = f.get("type") or ""
    if typ == "inter_repeat":
        return "".join(
            str(f.get(k) or "") for k in ("text_a", "text_b", "text_c")
        )
    return f.get("text") or ""


def _check_one(p: str, domain_text: str,
               use_boundary: bool, f: dict, local_asr: str | None = None) -> str | None:
    """单段原稿窗口内重复判定。忠于原稿返回 reason，否则 None。

    比较粒度：**本地 vs 本地**。
      - n_asr：默认只数出问题的句子/句对（local_asr），而非全篇口播；local_asr 为空
        时才回退到全篇 domain_text（兼容机械种子等无 text 字段的 finding）。
      - n_org：严格只用本句对应的原稿局部窗口 org_script_window（≤200 字），不再数
        整篇原稿、也不回退 original_script。这样原稿别处的重复不会被「多算」进来稀释
        本句的口误，避免误 skip；合法排比在原稿局部窗口里同样重复，仍会被正确排除。
        窗口缺失（如未能定位）则直接不跳，交 LLM 研判（保守）。
    """
    p_n = normalize_numerals_full(p)
    if not p_n:
        return None
    # n_asr：本地化计数（无本地文本时回退全篇）
    _asr_scope = local_asr or domain_text or ""
    n_asr = _count_nonoverlap(normalize_numerals_full(_asr_scope), p_n)
    if n_asr < 1:
        return None
    # n_org：严格只用本句对应原稿局部窗口 org_script_window（≤200 字）；
    # 窗口缺失不回退整篇 original_script（否则原稿别处高频重复被多算→误 skip）。
    org_text = (f.get("org_script_window") if isinstance(f, dict) else "") or ""
    if not org_text:
        return None
    org_n = normalize_numerals_full(org_text)
    if use_boundary:
        n_org = _count_boundary(org_n, p_n)
    else:
        n_org = _count_nonoverlap(org_n, p_n)
    if n_org >= n_asr >= 1:
        return f"待删段『{p}』原稿窗口亦重复({n_org}≥{n_asr})→忠于原稿/排比,排除"
    return None


def is_repeated_in_org_window(f: dict, sentences: list[dict],
                              domain_text: str | None = None) -> str | None:
    """单条 finding/issue 的原稿窗口内重复检查（忠于原稿/排比 → 返回 reason，否则 None）。"""
    if domain_text is None:
        domain_text = "\n".join(s.get("text", "") for s in sentences)
    seg, use_boundary = _extract_deleted_segment(f, sentences)
    if seg is None:
        return None
    # 局部 ASR 文本：只取本 finding 涉及的句子/句对，供 n_asr 本地化计数
    local_asr = _local_asr_text(f)
    if isinstance(seg, list):
        if not seg:
            return None
        all_faithful = all(
            _check_one(p, domain_text, use_boundary, f, local_asr) is not None
            for p in seg
        )
        return "句内所有重复段均原稿窗口重复,排除" if all_faithful else None
    return _check_one(seg, domain_text, use_boundary, f, local_asr)


# ============================================================
#  批量过滤（供 detect_loop 挂载）
# ============================================================

def filter_repeat_in_org_window(findings_by_cat: dict,
                                sentences: list[dict], domain_text: str | None = None) -> list[dict]:
    """对 {cat: [findings]} 做原稿窗口内重复排除（忠于原稿/排比的共享过滤）。

    **原地**在 findings 上打 skip 标记，不新建 skipped 对象：
        f["skip"] = {"stage": "filter_repeat_in_org_window", "reason": "..."}
    已被 _prepare_findings_for_judge 打过 skip 标记的不再二次判定。

    Returns:
        newly_skipped: 本次新打标记（filter_repeat_in_org_window）的 finding 列表，
                       供调用方单独打印「原稿窗口内重复」排除明细。
    """
    if domain_text is None:
        domain_text = "\n".join(s.get("text", "") for s in sentences)
    newly: list[dict] = []
    for cat, fnds in findings_by_cat.items():
        for f in fnds:
            if f.get("skip"):   # 已被去重/级联跳过的不再二次判定
                continue
            reason = is_repeated_in_org_window(f, sentences, domain_text)
            if reason:
                f["skip"] = {"stage": "filter_repeat_in_org_window", "reason": reason}
                newly.append(f)
    return newly
