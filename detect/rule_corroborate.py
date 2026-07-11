"""
rule_corroborate.py — 规则兜底过滤（复用 detect_intra / detect_inter）

职责：对 LLM 输出的「重复类」 issue（intra_repeat / inter_repeat）做机械兜底核对：
  - 若机械检测（detect_intra / detect_inter）也找到对应的真实重复 → 佐证通过，
    保留并交给 verify agent 判语义（是否同一角色/修辞/自然叠词等）。
  - 若机械检测判定为「不存在真实重复」→ 视为 LLM 编造的重复，直接过滤（驳回）。

分工原则：
  - 本模块只做「机械存在性」兜底，避免 LLM/verify 双重失误导致误删。
  - 语义正确性（是否同一角色、是否修辞、是否自然叠词）→ 仍由 verify agent 负责。
  - 保守偏向：当规则无法证伪时一律 KEEP（交由 verify 决定），绝不在不确定时删除。
  - fragment / 其他维度不进本过滤，保留交由 verify 处理。

intra_repeat 兜底判定（复用 detect_intra）：
  1) delete_text 在目标句中完全不存在 → 编造（如把 5500 当成 5000）→ REJ
  2) 句内真实出现 ≥2 次：
       - 单字（如主语「你…你」）→ KEEP
       - 多字 → detect_intra 命中重叠片段则 KEEP；否则 REJ
         （detect_intra 已排除自然叠词/修辞/语气词，故未命中=不同角色或修辞重复）
  3) 句内仅 1 次（模式B 跨句尾→头候选）：
       - detect_intra 命中重叠 → KEEP
       - 或 detect_inter 命中「目标句 与 后续某句 长共同前缀」（跨句头重叠）→ KEEP
       - 否则 KEEP（模式B 无法被规则证伪，交给 verify，不冒险过滤）

inter_repeat 兜底判定（复用 detect_inter）：
  detect_inter 在 (delete_sentence_idx, sentence_idx) 这对句子上命中句间重复 → KEEP；否则 REJ。
"""

from ..detect.detect_intra import detect_intra
from ..detect.detect_inter import detect_inter


def _common_prefix_len(a: str, b: str, n: int = 8) -> int:
    k = 0
    while k < n and k < len(a) and k < len(b) and a[k] == b[k]:
        k += 1
    return k


def _find_sentence(sentences: list[dict], idx) -> dict | None:
    for s in sentences:
        if s.get("idx") == idx:
            return s
    return None


def _intra_hit_overlaps(fnd: dict | None, delete_text: str) -> bool:
    if not fnd:
        return False
    for h in fnd.get("hits", []):
        ph = h.get("phrase", "")
        if ph and (ph in delete_text or delete_text in ph):
            return True
    return False


def _inter_modeB(inter_findings: list[dict], sid, sentences) -> bool:
    """目标句 sid 是否与某后续句存在足够长的共同前缀（detect_inter 跨句头重叠）。"""
    target = _find_sentence(sentences, sid)
    if target is None:
        return False
    for fnd in inter_findings:
        a = fnd.get("sent_a_idx")
        b = fnd.get("sent_b_idx") or fnd.get("sent_c_idx")
        if a is None or b is None:
            continue
        if sid not in (a, b):
            continue
        other = b if a == sid else a
        # 只接受「后续句」（idx 更大）作为完整版参照
        if other <= sid:
            continue
        cpl = fnd.get("common_prefix_len", 0)
        if cpl >= 5:
            return True
    return False


def _truly_absent(delete_text: str, txt: str) -> bool:
    """delete_text 是否真的在句中完全不存在（连 2 字子串都找不到）→ 编造。

    用 2 字子串兜底，避免 LLM 把 rough 版写得与原文差一两个字时误判为'不存在'。
    """
    if delete_text in txt:
        return False
    if len(delete_text) <= 2:
        return True
    for i in range(0, len(delete_text) - 1):
        if delete_text[i:i + 2] in txt:
            return False
    return True


def corroborate_intra(
    iss: dict,
    sentences: list[dict],
    intra_findings_by_sent: dict[int, dict],
    inter_findings: list[dict],
) -> tuple[bool, str]:
    """核对 intra_repeat：句内重复 / 跨句尾→头裁剪（模式B）。

    安全偏向：仅在「delete_text 物理上不存在于目标句」时驳回（LLM 编造，如把
    5500 当成 5000 笔误）。其余情况一律保留，交给 verify 判语义——因为 detect_intra
    会把相邻整词双写（「我先我先」）判为自然叠词跳过，直接用作负信号会误杀真实口误。
    detect_intra / detect_inter 在此仅作**正向**佐证信号（命中即保留）。
    """
    delete_text = (iss.get("delete_text") or iss.get("error_text") or "").strip()
    if not delete_text:
        return False, "intra_repeat 但无 delete_text，无法机械核对"
    sid = iss.get("sentence_idx")
    target = _find_sentence(sentences, sid)
    if target is None:
        return False, f"找不到目标句 句{sid}"
    txt = target.get("text", "")

    # 1) delete_text 物理上不存在于目标句 → 编造（如凭空把 5500 当成 5000）→ 驳回
    if _truly_absent(delete_text, txt):
        return False, f"delete_text「{delete_text}」在目标句句{sid}中完全不存在→编造"

    # 2) 句内真实出现 ≥2 次（含单字主语重复如「你…你」、多字口误重说如「我先我先」）
    #    → 真实重复，保留交由 verify 判是否冗余/修辞/不同角色
    if txt.count(delete_text) >= 2:
        return True, f"句内「{delete_text}」真实出现≥2次→真实重复"

    # 3) 句内仅 1 次 → 模式B（跨句尾→头）候选，用 detect_intra/detect_inter 正向佐证
    if _intra_hit_overlaps(intra_findings_by_sent.get(sid), delete_text):
        return True, f"detect_intra 句内命中重叠片段「{delete_text}」"
    if _inter_modeB(inter_findings, sid, sentences):
        return True, f"detect_inter 命中句{sid}与后续句的长共同前缀（跨句尾→头模式B）"
    # 无法被规则证伪 → 保守保留，交给 verify 判语义（不冒险过滤）
    return True, f"句内1次且规则无法证伪→保守保留（交由 verify）"


def _share_substring(a: str, b: str, min_len: int = 4) -> bool:
    """两字符串是否共享长度 >= min_len 的连续子串。"""
    if not a or not b:
        return False
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    for i in range(0, len(shorter) - min_len + 1):
        if shorter[i:i + min_len] in longer:
            return True
    return False


def corroborate_inter(
    iss: dict,
    sentences: list[dict],
    inter_findings: list[dict],
) -> tuple[bool, str]:
    """核对 inter_repeat：句间重复。

    通过条件：detect_inter 在 (delete_sentence_idx, sentence_idx) 这对句子上命中了
    句间重复；或两句话确实共享 ≥4 字连续子串（真实重叠，交给 verify 判语义）。
    仅当两句话几乎无共享内容时，才视为编造 → 驳回（保守，避免误杀真实句间重复）。
    """
    delete_sid = iss.get("delete_sentence_idx", iss.get("sentence_idx"))
    ref_sid = iss.get("sentence_idx")
    for fnd in inter_findings:
        a = fnd.get("sent_a_idx")
        b = fnd.get("sent_b_idx") or fnd.get("sent_c_idx")
        if a is None or b is None:
            continue
        if {a, b} == {delete_sid, ref_sid}:
            return True, f"detect_inter 命中句{a}与句{b}的句间重复"

    sa = _find_sentence(sentences, delete_sid)
    sb = _find_sentence(sentences, ref_sid)
    if sa and sb and _share_substring(sa.get("text", ""), sb.get("text", ""), 4):
        return True, f"句{delete_sid}与句{ref_sid}共享≥4字子串→真实重叠"
    return (
        False,
        f"句{delete_sid}与句{ref_sid}几乎无共享内容→疑似编造的句间重复",
    )


def rule_corroborate(
    issues: list[dict],
    sentences: list[dict],
) -> tuple[list[dict], list[tuple[dict, str]]]:
    """对 issues 做规则兜底过滤。

    Returns:
        kept:    通过机械兜底核对的 issues（交给 verify / 应用）
        rejected: [(iss, reason)] 被规则过滤掉的 issues（疑似编造的重复）
    """
    intra_findings = detect_intra(sentences)
    intra_findings_by_sent: dict[int, dict] = {
        f["sent_idx"]: f for f in intra_findings
    }
    inter_findings = detect_inter(sentences)

    kept: list[dict] = []
    rejected: list[tuple[dict, str]] = []
    for iss in issues:
        dim = iss.get("dimension")
        if dim == "intra_repeat":
            ok, reason = corroborate_intra(
                iss, sentences, intra_findings_by_sent, inter_findings
            )
        elif dim == "inter_repeat":
            ok, reason = corroborate_inter(iss, sentences, inter_findings)
        else:
            # fragment / 其他维度：不进规则过滤，保留交由 verify
            ok, reason = True, "非重复维度，不过滤"
        if ok:
            kept.append(iss)
        else:
            rejected.append((iss, reason))
    return kept, rejected
