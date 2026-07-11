"""
annotated_subtitle.py — 生成带口误标注的字幕 txt

功能:
- 输出完整口播文本
- 用【删除】/【重复】/【残句】等标记标注口误位置
- 静音部分不标注（正常停顿保留）
- 循环审查发现的问题按轮次单独列出

输出: 口误标注字幕.txt
"""

import json
from pathlib import Path
from speech_error_detector.assemble.assemble import _sentence_range
from speech_error_detector.base.paths import detect_dir


def generate_annotated_subtitle(
    analysis_dir: Path,
    words_json_path: Path,
    sentences: list[dict],
) -> str:
    """
    生成带口误标注的字幕文本
    
    Returns:
        标注后的完整文本
    """
    
    # === 加载数据 ===
    with open(words_json_path, encoding="utf-8") as f:
        words = json.load(f)
    
    # 待删除的 word 索引
    auto_path = analysis_dir / "auto_selected.json"
    with open(auto_path, encoding="utf-8") as f:
        delete_indices = set(json.load(f))
    
    # 加载 decisions（合并格式，兼容旧格式）
    decisions = {}
    for name in ["inter", "intra", "fragment"]:
        p = detect_dir(analysis_dir) / f"judge_decisions_{name}.json"
        if not p.exists():
            p = detect_dir(analysis_dir) / f"decisions_{name}.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                raw = json.load(f)
            # 新格式: [{detect, decision, status}] → 提取 decision
            if raw and isinstance(raw[0], dict) and "detect" in raw[0]:
                decisions[name] = [item["decision"] for item in raw]
            else:
                decisions[name] = raw
        else:
            decisions[name] = []
    
    loop_decisions_path = analysis_dir / "review_loop_decisions.json"
    loop_decisions = []
    if loop_decisions_path.exists():
        with open(loop_decisions_path, encoding="utf-8") as f:
            loop_decisions = json.load(f)
    
    # === 构建每个 word 的标注类型 ===
    # 先判断哪些 delete_indices 是静音（不标注），哪些是真正的口误
    speech_deletes = set()  # 非静音的删除（真正要标注的）
    gap_deletes = set()     # 静音删除（不标注）
    
    for idx in delete_indices:
        if idx < len(words):
            if words[idx].get("isGap"):
                gap_deletes.add(idx)
            else:
                speech_deletes.add(idx)
    
    # === 为每个句子构建标注信息 ===
    sentence_annotations = []  # [(sentence_text_with_marks, annotation_list), ...]
    
    for sent in sentences:
        sent_idx = sent["idx"]
        r = _sentence_range(sent)
        if not r:
            sentence_annotations.append((sent["text"], [], sent_idx))
            continue
        start_idx, end_idx = r

        # 找出这个句子中需要标记的字索引
        sent_delete_idxs = [i for i in range(start_idx, end_idx + 1) if i in speech_deletes]
        
        if not sent_delete_idxs:
            # 这个句子没有口误标注
            sentence_annotations.append((sent["text"], [], sent_idx))
            continue
        
        # 构建标注文本：在要删除的位置插入标记
        annotated_chars = []
        current_mark_start = None
        mark_chars = []
        
        for i in range(start_idx, end_idx + 1):
            char = words[i]["text"] if i < len(words) else ""
            
            if i in speech_deletes:
                # 要删除的字
                if current_mark_start is None:
                    current_mark_start = i
                mark_chars.append(char)
            else:
                # 正常的字
                if current_mark_start is not None and mark_chars:
                    # 结束一个标记段
                    annotated_chars.append(f"【删除:'{''.join(mark_chars)}'】")
                    mark_chars = []
                    current_mark_start = None
                annotated_chars.append(char)
        
        # 处理末尾的标记
        if current_mark_start is not None and mark_chars:
            annotated_chars.append(f"【删除:'{''.join(mark_chars)}'】")
        
        annotated_text = "".join(annotated_chars)
        
        # 确定这个句子的错误类型
        error_types = []
        
        # 句间重复
        for d in decisions.get("inter", []):
            if sent_idx in d.get("delete_sentences", []):
                error_types.append(f"🔁 句间重复(被后句覆盖)")
        
        # 句内重复
        for d in decisions.get("intra", []):
            if d.get("sentence") == sent_idx and d.get("delete_ranges"):
                error_types.append(f"🔄 句内重复")
        
        # 残句
        for d in decisions.get("fragment", []):
            if d.get("sentence") == sent_idx:
                mode = d.get("mode", "")
                if mode == "full":
                    error_types.append(f"🔶 残句(整句删): {d.get('llm_reason', '')[:30]}")
                elif mode == "keep_head":
                    error_types.append(f"🔶 残句(保头删尾): {d.get('llm_reason', '')[:30]}")
        
        sentence_annotations.append((annotated_text, error_types, sent_idx))
    
    # === 组装输出文本 ===
    output_lines = []
    output_lines.append("=" * 60)
    output_lines.append("   口误标注字幕（静音部分已隐藏，仅标注口误位置）")
    output_lines.append("=" * 60)
    output_lines.append("")
    
    # ---- 第一部分：标注版（带错误标记）----
    output_lines.append("┌" + "─" * 56 + "┐")
    output_lines.append("│  📝 第一部分：原始字幕（口误已标注）" + " " * 21 + "│")
    output_lines.append("└" + "─" * 56 + "┘")
    output_lines.append("")
    
    has_errors = False
    for annotated_text, error_types, sent_idx in sentence_annotations:
        prefix = ""
        suffix = ""
        
        if error_types:
            has_errors = True
            prefix = "❌ "
            suffix = f"  ← {'; '.join(error_types)}"
        
        output_lines.append(f"{prefix}{annotated_text}{suffix}")
        output_lines.append("")  # 空行分隔
    
    # ---- 第二部分：修正后（干净文本）----
    output_lines.append("")
    output_lines.append("┌" + "─" * 56 + "┐")
    output_lines.append("│  ✅ 第二部分：修正后字幕（口误已删除）" + " " * 19 + "│")
    output_lines.append("└" + "─" * 56 + "┘")
    output_lines.append("")
    
    clean_lines = []
    for annotated_text, error_types, sent_idx in sentence_annotations:
        # 从标注文本中提取干净内容（去掉【删除:xxx】标记）
        import re
        clean_text = re.sub(r"【[^】]+】", "", annotated_text).strip()
        if clean_text:
            clean_lines.append(clean_text)
    
    clean_full = "\n".join(clean_lines)
    output_lines.append(clean_full)
    output_lines.append("")
    
    # === 循环审查发现的问题（按轮次列出） ===
    applied_loop_issues = [d for d in loop_decisions if d.get("status") == "applied"]
    if applied_loop_issues:
        # 按轮次分组
        rounds: dict[int, list] = {}
        for d in applied_loop_issues:
            r = d.get("round", 0)
            rounds.setdefault(r, []).append(d)
        
        output_lines.append("")
        output_lines.append("=" * 60)
        output_lines.append("   🔄 循环审查发现的问题（按轮次）")
        output_lines.append(f"   共 {len(applied_loop_issues)} 项已确认删除，分布在 {len(rounds)} 轮")
        output_lines.append("=" * 60)
        output_lines.append("")
        
        dim_label = {
            "inter_repeat": "句间重复(整句删)",
            "intra_repeat": "句内/跨句裁剪",
            "fragment": "残缺句子",
        }
        severity_map = {
            "critical": "🔴 致命",
            "major": "🟠 严重",
            "minor": "🟡 轻微",
        }
        
        for rnd in sorted(rounds.keys()):
            issues = rounds[rnd]
            output_lines.append(f"--- Round {rnd} ({len(issues)} 项) ---")
            
            for iss in issues:
                detect = iss.get("detect", {})
                decision = iss.get("decision", {})
                dim = dim_label.get(detect.get("dimension", ""), detect.get("dimension", "?"))
                sev = severity_map.get(detect.get("severity", ""), "")
                sid = detect.get("sentence_idx", "?")
                err_text = (detect.get("error_text") or detect.get("delete_text", ""))[:50]
                reason = decision.get("reason", "")[:60]
                
                tag = f"[{sev}] " if sev else ""
                output_lines.append(f"  {tag}[{dim}] 句{sid}「{err_text}」")
                if reason:
                    output_lines.append(f"      → {reason}")
            output_lines.append("")
    
    # === 统计 & 对比摘要 ===
    output_lines.append("-" * 60)
    total_speech_errors = len(speech_deletes)
    total_gap_removed = len(gap_deletes)
    
    # 从 words 数据直接计算字数（而非 sentences 文本，避免不一致）
    original_chars = 0
    deleted_chars = 0
    for sent in sentences:
        r = _sentence_range(sent)
        if not r:
            continue
        s_start, s_end = r
        for wi in range(s_start, s_end + 1):
            if wi >= len(words):
                break
            w = words[wi]
            if not w.get("isGap"):
                original_chars += len(w["text"])
                if wi in speech_deletes:
                    deleted_chars += len(w["text"])
    
    clean_chars = original_chars - deleted_chars
    
    output_lines.append("📊 删除前后对比:")
    output_lines.append(f"  ┌────────────┬────────────┐")
    output_lines.append(f"  │   指标      │    数量     │")
    output_lines.append(f"  ├────────────┼────────────┤")
    output_lines.append(f"  │ 原始字数    │  {original_chars:>8}  │")
    output_lines.append(f"  │ 修正后字数  │  {clean_chars:>8}  │")
    output_lines.append(f"  │ 删除口误    │  {deleted_chars:>8}  │ (字符数)")
    output_lines.append(f"  │ 静音隐藏    │  {total_gap_removed:>8}  │")
    output_lines.append(f"  └────────────┴────────────┘")
    output_lines.append(f"  • 精简率: {(1 - clean_chars/max(original_chars,1))*100:.1f}%")
    output_lines.append(f"  • 循环审查发现: {len(applied_loop_issues)} 项已确认删除")
    
    # 按维度统计
    dim_counts = {}
    for d in applied_loop_issues:
        dim = (d.get("detect") or {}).get("dimension", "unknown")
        dim_counts[dim] = dim_counts.get(dim, 0) + 1
    for dim, cnt in sorted(dim_counts.items()):
        label = {"inter_repeat": "句间重复", "intra_repeat": "句内裁剪", "fragment": "残句"}.get(dim, dim)
        output_lines.append(f"    - {label}: {cnt} 项")
    output_lines.append("")
    
    if not has_errors and not applied_loop_issues:
        output_lines.append("✅ 未检测到口误问题！")
    
    return "\n".join(output_lines)

