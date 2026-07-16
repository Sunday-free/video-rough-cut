"""agent_apply.py — Agent 循环审查系统的删除应用与摘要模块。

提供:
  - build_sent_range_map(): 句子编号 -> 字级 range 映射
  - apply_deletions_to_sentences(): 应用已确认删除（整句删 / 片段裁剪）
  - build_rejected_summary(): 汇总已驳回项供 detect agent 参考

被 review_loop.py 复用。
"""

import re

from speech_error_detector.utils.fillers import MODAL_CHARS, FILLER_CHARS


def is_modal_only_delete(delete_text: str) -> bool:
    """delete_text 非空且全部由语气词/叹词(MODAL_CHARS)组成 → 纯语气词删除，应过滤。

    如「呃」重复出现，属自然语流，不在系统处理范围内。
    """
    t = (delete_text or "").strip()
    return bool(t) and all(ch in MODAL_CHARS for ch in t)


# 归一化时剔除的标点/空白集合（中英文）
_PUNCT = set("，。！？；、：\"'…—-()[]【】《》〈〉,.!?;:()[]{}<>~`@#%^&*+=|\\/ \t\n\r")


def _norm_text(s: str) -> tuple[str, list[int]]:
    """去标点/空白/语气词/虚词归一化，返回 (归一串, 原字符索引映射)。

    剥离语气词(呢/啊/呀…)与虚词(的/了/把…)后比对，避免口语语气词干扰
    resay 判定（如「〔甲内容〕呢是…」与「〔甲内容〕是…」应视为同一内容）。
    """
    out_chars: list[str] = []
    orig_idx: list[int] = []
    for i, ch in enumerate(s):
        if ch in _PUNCT or ch in FILLER_CHARS:
            continue
        out_chars.append(ch)
        orig_idx.append(i)
    return "".join(out_chars), orig_idx

def _pick_occurrence(starts: list[int], reason: str) -> int:
    """delete_text 在句中出现多次时，二选一确定要删的位置。

    口播场景里「残句重说 / 片段裁剪 / 尾部残片」几乎都位于句尾，故默认删【最后一个】
    匹配；仅当 confirm 理由显式指明「开头 / 前半 / 前面」时才删【第一个】匹配。

    返回选定的起始下标；starts 为空返回 -1。
    """
    if not starts:
        return -1
    if len(starts) == 1:
        return starts[0]
    r = (reason or "")
    if any(k in r for k in ("开头", "前半", "前面", "前置", "首句")):
        return starts[0]
    # 默认尾部优先
    return starts[-1]


def _locate_delete_words(
    sid: int,
    delete_text: str,
    words: list[dict],
    range_map: dict[int, tuple[int, int]],
    reason: str = "",
) -> tuple[set[int] | None, str]:
    """在句 sid 中定位 delete_text 对应的【词索引】集合（确定性，不依赖 LLM）。

    流程：
      1) 用 range_map 取出该句覆盖的词区间，构建 原始句文本 + (归一字符下标 → 词索引)
         映射（静音 gap 词跳过，不参与对齐）。
      2) 对句文本与 delete_text 各做一次归一化（去标点/语气词），在归一文本上用子串匹配
         定位 delete_text，再映射回原始字符下标，得到要删的词索引集合。
      3) 归一匹配失败则兜底：直接在原始句文本里 find（兼容 confirm 给出的含标点片段）。
    返回 (to_delete, to_delete_text)：
      - to_delete: 词索引集合；None 表示无法定位（文本不匹配 / delete_text 为空）→ 调用方据此安全不删
      - to_delete_text: 这些词索引拼接出的实际删除文字（按词序），供报告/控制台精确展示
    """
    if sid not in range_map or not delete_text:
        return None, ""
    a, b = range_map[sid]
    raw = ""
    pos2wi: dict[int, int] = {}
    cp = 0
    for wi in range(a, b + 1):
        if wi >= len(words):
            break
        w = words[wi]
        if w.get("isGap"):
            continue
        pos2wi[cp] = wi
        raw += w.get("text", "")
        cp += len(w.get("text", ""))
    if not raw:
        return None, ""

    norm_raw, orig_idx = _norm_text(raw)
    norm_dt, _ = _norm_text(delete_text)
    if norm_dt:
        starts = [m.start() for m in re.finditer(re.escape(norm_dt), norm_raw)]
        start = _pick_occurrence(starts, reason)
        if start >= 0:
            end = start + len(norm_dt)
            o_start = orig_idx[start]
            o_end = orig_idx[end - 1] + 1
            to_delete = {wi for cpos, wi in pos2wi.items() if o_start <= cpos < o_end}
            to_delete_text = "".join(words[wi].get("text", "") for wi in sorted(to_delete))
            return (to_delete if to_delete else None), to_delete_text

    # 兜底：不经归一化直接在原始句文本里找
    raw_starts = [m.start() for m in re.finditer(re.escape(delete_text), raw)]
    raw_start = _pick_occurrence(raw_starts, reason)
    if raw_start < 0:
        return None, ""
    raw_end = raw_start + len(delete_text)
    to_delete = {wi for cpos, wi in pos2wi.items() if raw_start <= cpos < raw_end}
    to_delete_text = "".join(words[wi].get("text", "") for wi in sorted(to_delete))
    return (to_delete if to_delete else None), to_delete_text

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
) -> tuple[list[dict], list[dict]]:
    """应用已确认的删除操作到句子列表（delete_text 驱动，确定性，不二次研判）。

    - delete_text 覆盖整句所有非静音词 → 整句移除（full）
    - delete_text 仅为句中片段 → 修剪该片段（partial）
    删除范围完全由 confirm 给出的 delete_text 决定，不再要求"整句被后文重说包含"。

    Returns:
        (修正后的句子列表, applied_records)
        - 句子列表：已删条目移除，剩余句子 idx 保持不变
        - applied_records: 与 confirmed_issues 平行，每条记录实际删除的文字/词索引/模式（full/partial/None）
    """
    range_map = build_sent_range_map(sentences)

    # 收集要整句删除的 sentence idx 集合
    full_delete_sids: set[int] = set()
    # 收集要精确删除的 word 索引
    partial_delete_indices: set[int] = set()
    # 与 confirmed_issues 平行，记录每条实际删除内容，供报告/控制台精确展示
    applied_records: list[dict] = []

    for iss in confirmed_issues:
        dim = iss.get("dimension", "")
        if dim not in ("misread", "intra_repeat", "inter_repeat", "fragment"):
            applied_records.append({"sid": None, "deleted_text": "", "indices": [], "mode": None})
            continue
        # 目标句：resay/misread 候选即该句
        target_sid = iss.get("sentence_idx")
        if dim in ("fragment", "intra_repeat"):
            target_sid = iss.get("sentence_idx")
        if target_sid is None or target_sid not in range_map:
            applied_records.append({"sid": target_sid, "deleted_text": "", "indices": [], "mode": None})
            continue

        delete_text = (iss.get("delete_text") or "").strip()
        reason = iss.get("delete_reason", iss.get("reason", ""))
        to_delete, deleted_text = _locate_delete_words(target_sid, delete_text, words, range_map, reason)
        rec = {
            "sid": target_sid,
            "deleted_text": deleted_text,
            "indices": sorted(to_delete) if to_delete else [],
            "mode": None,
        }
        applied_records.append(rec)
        if not to_delete:
            # 无法定位（文本不匹配 / delete_text 为空）→ 安全不删
            continue

        a, b = range_map[target_sid]
        all_non_gap = {
            wi for wi in range(a, b + 1)
            if wi < len(words) and not words[wi].get("isGap")
        }
        if to_delete >= all_non_gap:
            # 删除范围覆盖整句 → 整句移除
            full_delete_sids.add(target_sid)
            rec["mode"] = "full"
        else:
            partial_delete_indices.update(to_delete)
            rec["mode"] = "partial"

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

    # 兜底清理：移除归一化后仅含语气词/虚词的残留残句（如单字「在/诶/啊/嗱」）。
    # 成因：多轮循环里 detect 每轮基于 origin 全句检测，对已 partial 删除的句重复报
    # 「主要残句部分」，删后永远剩一个纯语气词，下一轮又基于 origin 报同样内容 → 死循环。
    # 这里确定性地把「归一后为空」的句子整句移除（正常短句如『吸筹』归一非空，不误删）。
    final_cleaned = []
    residual_removed = 0
    for s in cleaned:
        t = (s.get("text") or "").strip()
        if not t or not _norm_text(t)[0]:
            residual_removed += 1
            continue
        final_cleaned.append(s)
    if residual_removed:
        print(f"      兜底清理移除 {residual_removed} 个纯语气词残留句，剩 {len(final_cleaned)} 句")

    return final_cleaned, applied_records



def build_rejected_summary(all_decisions: list[dict]) -> str:
    """基于已有 decisions 构建"已驳回摘要"，传给 detect agent。
    
    只包含已驳回（confirmed=false）的项目 + 驳回原因；已确认并删除的句子已不在当前文本中，无需提醒。
    """
    if not all_decisions:
        return "**（第一轮扫描，尚无已处理项）**"

    lines_rejected = []
    for d in all_decisions:
        dec = d.get("decision", {})
        if dec.get("confirmed"):
            continue  # 已确认删除的不需要提醒，句子已不在文本中

        det = d.get("detect", {})
        dim = det.get("type", det.get("dimension", "?"))
        sid = det.get("sent_idx", det.get("sentence_idx", "?"))
        err = (det.get("text") or det.get("error_text") or det.get("delete_text", ""))[:40]
        reason = dec.get("llm_reason", dec.get("reason", ""))[:50]
        rnd = d.get("round", "?")

        entry = f"### ❌ 已验证驳回 - [Round {rnd}] 句{sid}「{err}」({dim}): {reason}"
        lines_rejected.append(entry)

    if not lines_rejected:
        return "**（尚无已驳回的项目）**"

    header = f"【以下共 {len(lines_rejected)} 个问题已验证驳回，不要重复报告】：\n\n"
    return header + "\n".join(lines_rejected)

