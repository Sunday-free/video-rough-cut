"""agent_apply.py — Agent 循环审查系统的删除应用与摘要模块。

提供:
  - build_sent_range_map(): 句子编号 -> 字级 range 映射
  - apply_deletions_to_sentences(): 应用已确认删除（整句删 / 片段裁剪）
  - build_processed_summary(): 汇总已处理项供 detect agent 参考

被 review_loop.py 复用。
"""

from speech_error_detector.utils.fillers import MODAL_CHARS


def is_modal_only_delete(delete_text: str) -> bool:
    """delete_text 非空且全部由语气词/叹词(MODAL_CHARS)组成 → 纯语气词删除，应过滤。

    如「呃」重复出现，属自然语流，不在系统处理范围内。
    """
    t = (delete_text or "").strip()
    return bool(t) and all(ch in MODAL_CHARS for ch in t)


def build_sent_range_map(sentences: list[dict]) -> dict[int, tuple[int, int]]:
    """构建 {句子编号: (startIdx, endIdx)} 映射"""
    m = {}
    for s in sentences:
        idx = s.get("idx")
        if idx is None:
            continue
        if "startIdx" in s and "endIdx" in s:
            m[idx] = (s["startIdx"], s["endIdx"])
        else:
            rng = s.get("range", "0-0")
            parts = rng.split("-")
            m[idx] = (int(parts[0]), int(parts[1]))
    return m


def _extract_quoted(text: str) -> str:
    """从描述性文字中提取 LLM 引用的片段（最后一个单/双引号包裹的子串）。

    例: "句内重复：'全部卖掉8000' 中的 '8000' 与后文重复…" → '8000'
    用于 delete_text 缺失/非子串时，兜底定位要删的片段。
    """
    if not text:
        return ""
    import re
    qs = re.findall(r"['\"]([^'\"]+)['\"]", text)
    for q in reversed(qs):
        if q:
            return q
    return ""



def apply_deletions_to_sentences(
    sentences: list[dict],
    words: list[dict],
    confirmed_issues: list[dict],
) -> list[dict]:
    """
    应用已确认的删除操作到句子列表。
    
    - inter_repeat / fragment（整句删）→ 直接从列表移除该条目
    - intra_repeat（片段删）→ 保留句子但修剪 text
    
    Returns:
        修正后的句子列表（已删条目移除，剩余句子 idx 保持不变）
    """
    range_map = build_sent_range_map(sentences)
    
    # 收集要整句删除的 sentence idx 集合
    full_delete_sids: set[int] = set()
    # 收集要精确删除的 word 索引
    partial_delete_indices: set[int] = set()
    
    for iss in confirmed_issues:
        dim = iss.get("dimension", "")
        sid = iss.get("sentence_idx")
        delete_sid = iss.get("delete_sentence_idx", sid)
        
        if dim in ("inter_repeat", "fragment", "misread"):
            # misread 整删用 delete_sentence_idx（错的句/重说的残句）；fragment 用 sentence_idx
            target_sid = delete_sid if dim != "fragment" else sid
            if target_sid is not None and target_sid in range_map:
                full_delete_sids.add(target_sid)
                    
        elif dim == "intra_repeat":
            target_sid = sid
            if target_sid is not None and target_sid in range_map:
                a, b = range_map[target_sid]
                delete_text = iss.get("delete_text") or iss.get("error_text", "")
                char_offset = iss.get("char_offset")
                
                if delete_text and a is not None:
                    sent_text = ""
                    word_map: dict[int, int] = {}
                    char_pos = 0
                    for wi in range(a, b + 1):
                        if wi >= len(words):
                            break
                        w = words[wi]
                        if w.get("isGap"):
                            continue
                        word_map[char_pos] = wi
                        sent_text += w["text"]
                        char_pos += len(w["text"])
                    
                    del_start = None
                    if char_offset is not None and 0 <= char_offset < len(sent_text):
                        if sent_text[char_offset:char_offset + len(delete_text)] == delete_text:
                            del_start = char_offset
                    if del_start is None:
                        del_start = sent_text.rfind(delete_text)

                    # 兜底：delete_text 缺失或不是句子子串（如 LLM 只给了描述性文字），
                    # 从 error_text 提取其引用的片段（引号内）作为实际待删文本。
                    if del_start is None or del_start < 0:
                        quoted = _extract_quoted(iss.get("error_text", "") or delete_text)
                        if quoted and quoted in sent_text:
                            del_start = sent_text.rfind(quoted)

                    if del_start is not None and del_start >= 0:
                        del_end = del_start + len(delete_text)
                        for cp in sorted(word_map.keys()):
                            if del_start <= cp < del_end:
                                partial_delete_indices.add(word_map[cp])
    
    # 构建新列表：跳过被整句删除的，保留并修剪片段删除的
    cleaned = []
    removed_count = 0
    
    for s in sentences:
        idx = s.get("idx")
        
        # 整句删除 → 直接跳过
        if idx in full_delete_sids:
            removed_count += 1
            continue
        
        # 检查是否有片段删除需要修剪
        needs_trim = False
        if idx is not None and idx in range_map:
            a, b = range_map[idx]
            for wi in range(a, b + 1):
                if wi in partial_delete_indices:
                    needs_trim = True
                    break
        
        if needs_trim:
            a, b = range_map[idx]
            result_text = ""
            for wi in range(a, b + 1):
                if wi >= len(words):
                    break
                w = words[wi]
                if w.get("isGap"):
                    continue
                if wi not in partial_delete_indices:
                    result_text += w["text"]
            text = result_text
        else:
            text = s.get("text", "")
        
        cleaned.append({
            "idx": idx,
            "range": s.get("range", ""),
            "text": text,
        })
    
    if removed_count > 0:
        print(f"      移除了 {removed_count} 个整句，剩 {len(cleaned)} 句")
    
    return cleaned



def build_processed_summary(all_decisions: list[dict]) -> str:
    """基于已有 decisions 构建"已处理摘要"，传给 detect agent（含已应用 + 已驳回）"""
    if not all_decisions:
        return "**（第一轮扫描，尚无已处理项）**"
    
    lines_applied = []
    lines_rejected = []
    for d in all_decisions:
        det = d.get("detect", {})
        dec = d.get("decision", {})
        dim = det.get("type", det.get("dimension", "?"))
        sid = det.get("sent_idx", det.get("sentence_idx", "?"))
        err = (det.get("text") or det.get("error_text") or det.get("delete_text", ""))[:40]
        reason = dec.get("llm_reason", dec.get("reason", ""))[:50]
        rnd = d.get("round", "?")

        entry = f"- [Round {rnd}] 句{sid}「{err}」({dim}): {reason}"

        if dec.get("confirmed"):
            lines_applied.append(entry)
        else:
            lines_rejected.append(entry)
    
    parts = []
    if lines_applied:
        parts.append(f"### ✅ 已确认并删除（{len(lines_applied)} 个）：\n" + "\n".join(lines_applied))
    if lines_rejected:
        parts.append(f"### ❌ 已验证驳回（{len(lines_rejected)} 个，不要再报）：\n" + "\n".join(lines_rejected))
    
    if not parts:
        return "**（尚无已处理的修改）**"
    
    header = f"【以下共 {len(lines_applied) + len(lines_rejected)} 个问题已处理完毕，不要重复报告】：\n\n"
    return header + "\n\n".join(parts)

