"""
assemble.py — 步骤5: 装配 auto_selected.json + 口误分析报告

合并 decisions_inter/intra/fragment + 静音/语气词规则 → 最终删除 idx 列表
"""

import json
import os
from pathlib import Path
from speech_error_detector.base.paths import detect_dir
from speech_error_detector.base.sentence_io import write_sentences


# ============================================================
#  决策删除：对已应用决策的 sentences 做 diff（不再重新投影决策 JSON）
# ============================================================

def _sentence_range(s: dict) -> tuple[int, int] | None:
    """从句子 dict 解析 word_idx 范围（兼容 range 字符串与 startIdx/endIdx）"""
    rng = s.get("range")
    if rng:
        a, b = rng.split("-")
        return int(a), int(b)
    if "startIdx" in s and "endIdx" in s:
        return int(s["startIdx"]), int(s["endIdx"])
    return None


def _find_kept_words(a: int, b: int, words: list[dict], cleaned: str) -> set[int]:
    """在 [a, b] 内找出保留的非 gap 词，使其文本拼接 == cleaned。

    保留词是 range 词的一个子序列；用 DP 求解，回溯时优先 keep（删除尽可能少）。
    """
    seq = [
        (wi, words[wi]["text"])
        for wi in range(a, b + 1)
        if wi < len(words) and not words[wi].get("isGap")
    ]
    m, n = len(seq), len(cleaned)
    reach = [[False] * (n + 1) for _ in range(m + 1)]
    reach[0][0] = True
    for i in range(m + 1):
        for c in range(n + 1):
            if not reach[i][c]:
                continue
            if i < m:
                _, wt = seq[i]
                if c + len(wt) <= n and cleaned[c:c + len(wt)] == wt:
                    reach[i + 1][c + len(wt)] = True
                reach[i + 1][c] = True
    if not reach[m][n]:
        # 兜底（理论上不会触发）：顺序贪心
        kept: set[int] = set()
        cp = 0
        for wi, wt in seq:
            if cleaned[cp:].startswith(wt):
                kept.add(wi)
                cp += len(wt)
        return kept
    kept: set[int] = set()
    i = c = 0
    while i < m:
        wi, wt = seq[i]
        if c + len(wt) <= n and cleaned[c:c + len(wt)] == wt and reach[i + 1][c + len(wt)]:
            kept.add(wi)
            i += 1
            c += len(wt)
        else:
            i += 1
    return kept


def _compute_decision_delete_set(
    words: list[dict],
    current_sentences: list[dict],
    original_sentences: list[dict],
) -> set[int]:
    """由已应用决策的 current_sentences 与原始 sentences 做 diff，得到决策类待删 word 集合。

    - current_sentences 中缺失 idx 的句 → 整句删除（删其 [a,b] 全部词）。
    - 其余句 → 用 _find_kept_words 找出被修剪掉的词并删除。
    """
    sel: set[int] = set()
    final_text = {s["idx"]: s.get("text", "") for s in current_sentences}
    for s in original_sentences:
        r = _sentence_range(s)
        if not r:
            continue
        a, b = r
        if s["idx"] not in final_text:
            # 整句删除：范围内所有词（含 gap），与旧逻辑一致
            for wi in range(a, b + 1):
                if wi < len(words):
                    sel.add(wi)
        else:
            cleaned = final_text[s["idx"]]
            kept = _find_kept_words(a, b, words, cleaned)
            for wi in range(a, b + 1):
                if wi >= len(words):
                    break
                if words[wi].get("isGap"):
                    continue
                if wi not in kept:
                    sel.add(wi)
    return sel


def iter_gap_runs(words: list) -> list:
    """返回连续 gap 块列表，每项 (start_idx, end_idx_exclusive, combined_duration)。

    一段长停顿常被拆成多条 gap 条目；单条可能都低于阈值，但合计超过阈值，
    应作为一整段静音处理（删除时由剪辑端统一保留 SILENCE_KEEP_DURATION 呼吸感）。
    """
    runs = []
    i = 0
    n = len(words)
    while i < n:
        if words[i].get("isGap"):
            j = i
            while j < n and words[j].get("isGap"):
                j += 1
            combined = sum(words[k]["end"] - words[k]["start"] for k in range(i, j))
            runs.append((i, j, combined))
            i = j
        else:
            i += 1
    return runs


# ============================================================
#  核心: 合并所有 decisions → auto_selected.json
# ============================================================

def run_assemble(
    analysis_dir: Path,
    words_json_path: Path,
    sentences: list[dict],
    original_sentences: list[dict],
    silence_thresh: float = 0.3,
    video_duration: float = 0.0,
) -> tuple[Path, dict]:
    """
    装配 auto_selected.json。

    设计要点：Judge 与 Agent 循环审查已经把决策投影进 current_sentences（即传入的
    sentences）。这里**不再重新读取决策 JSON 投影 word 索引**，而是直接对
    current_sentences 与原始 sentences 做 diff 得到删除集合，静音/夹缝规则另行计算。
    决策投影只发生一次，避免重复与格式漂移。

    Args:
        sentences:          最终句子列表（已应用 Judge + Loop 删除）。必传，
                            不再从 sentences.txt 读取。
        original_sentences: 投影前的原始全量句子（含被整句删除的句），用于定位整句
                            删除的 word 范围。必传。

    Returns:
        (auto_selected_path, stats_dict)
    """
    auto_path = analysis_dir / "auto_selected.json"

    # 加载数据
    with open(words_json_path, encoding="utf-8") as f:
        words = json.load(f)

    sel = set()  # 待删除的 word idx 集合
    stats = {
        "sil_open": 0,
        "sil_body": 0,
        "sil_tail": 0,
        "sil_internal": 0,
        "sil_after_deleted": 0,
        "filler": 0,
        "total": 0,
    }
    
    # --- 结尾补尾 ---
    vdur = video_duration or float(os.environ.get("VDUR", "0"))
    last = words[-1]
    if vdur and vdur - last["end"] > 0.3:
        words.append({
            "text": "", "start": last["end"], "end": vdur,
            "isGap": True, "reason": "结尾未转录(杂音/收尾)"
        })
    
    first_speech = next(i for i, e in enumerate(words) if not e.get("isGap"))
    last_speech = len(words) - 1 - next(
        i for i, e in enumerate(reversed(words)) if not e.get("isGap")
    )

    # --- 规则1: 静音删除（连续 gap 合并判断） ---
    # 开头(first_speech 之前)与结尾(last_speech 之后)的静音无条件删除，不论多长；
    # 片中静音：一整段连续 gap 合计时长 ≥ silence_thresh 才删除
    # （单条低于阈值、但合计超过阈值的长停顿也一并删，剪辑时保留
    #  SILENCE_KEEP_DURATION 呼吸感）。
    for gi, gj, combined in iter_gap_runs(words):
        is_opening = gj <= first_speech       # 整段在首句之前
        is_closing = gi > last_speech         # 整段在末句之后
        if is_opening or is_closing:
            for k in range(gi, gj):
                sel.add(k)
            if is_opening:
                stats["sil_open"] += (gj - gi)
            else:
                stats["sil_tail"] += (gj - gi)
        elif combined >= silence_thresh:
            for k in range(gi, gj):
                sel.add(k)
            stats["sil_body"] += (gj - gi)
        # 否则：片中短停顿（合计也低于阈值）保留，不删
    
    # --- 决策删除：由已应用决策的 sentences 与原始 sentences 做 diff ---
    # 决策（句间/句内/残句/循环审查）已在 Judge 与 Loop 阶段投影进 sentences，
    # 这里不再重新读取决策 JSON，直接 diff 得到删除集合（单一事实来源，避免重复投影）。
    sel |= _compute_decision_delete_set(words, sentences, original_sentences)

    # --- 规则补充: 夹在被删词之间的静音 gap 一并删除 ---
    # 否则删掉两侧说错内容后，中间会留下一段孤立停顿。
    for i, e in enumerate(words):
        if e.get("isGap") and i not in sel:
            l = i - 1
            while l >= 0 and words[l].get("isGap"):
                l -= 1
            r = i + 1
            while r < len(words) and words[r].get("isGap"):
                r += 1
            if l >= 0 and r < len(words) and l in sel and r in sel:
                sel.add(i)
                stats["sil_internal"] = stats.get("sil_internal", 0) + 1

    # --- 规则补充: 被整句删除的句子，它前后的静音全部删光 ---
    # 当一句话被整句删除，它和前一句、后一句之间的所有静音（不限于紧邻的）
    # 都应删除，不留呼吸感。如果中间有多段不连续的静音也全部删除。
    final_text = {s["idx"]: s.get("text", "") for s in sentences}
    # 预计算每个句子的 word 区间
    orig_range_map: dict[int, tuple[int, int]] = {}
    for os_ in original_sentences:
        rr = _sentence_range(os_)
        if rr:
            orig_range_map[os_["idx"]] = rr

    for s in original_sentences:
        r = _sentence_range(s)
        if not r:
            continue
        a, b = r
        # 兼容「整句删除后仍保留空 text 行」：空文本等同于整句删除
        if final_text.get(s["idx"], "").strip() != "":
            continue

        # 找到前一个存活的句子
        prev_end: int | None = None
        for os_ in original_sentences:
            if os_["idx"] >= s["idx"]:
                break
            if final_text.get(os_["idx"], "").strip() != "":
                pr = orig_range_map.get(os_["idx"])
                if pr:
                    prev_end = pr[1]

        # 找到后一个存活的句子
        next_start: int | None = None
        for os_ in reversed(original_sentences):
            if os_["idx"] <= s["idx"]:
                break
            if final_text.get(os_["idx"], "").strip() != "":
                nr = orig_range_map.get(os_["idx"])
                if nr:
                    next_start = nr[0]

        # 删除 [prev_end+1, next_start) 之间的所有静音，
        # 覆盖句间短停顿、多段不连续静音等。
        gap_start = (prev_end + 1) if prev_end is not None else 0
        gap_end = next_start if next_start is not None else len(words)
        for wi in range(gap_start, gap_end):
            if words[wi].get("isGap") and wi not in sel:
                sel.add(wi)
                stats["sil_after_deleted"] += 1

    # --- 规则补充: 句尾部分删词后露出的静音也删除 ---
    # 当一句的尾部词被删（keep_head 模式），尾词后的静音会暴露为
    # 句子间的不自然停顿，应一并删除。
    for s in original_sentences:
        r = _sentence_range(s)
        if not r:
            continue
        a, b = r

        # 跳过整句删除的（前面已处理）
        if final_text.get(s["idx"], "").strip() == "":
            continue

        # 跳过整句未变的
        if s.get("text", "") == final_text.get(s["idx"], ""):
            continue

        # 找到句尾最后一个被删的非 gap 词
        # 若 b 本身是 gap，往前找到该句实际的最后一个词
        _tail_idx = b
        while _tail_idx >= a and words[_tail_idx].get("isGap"):
            _tail_idx -= 1
        if _tail_idx < a:
            continue

        # 尾部词未被删 → 跳过
        if _tail_idx not in sel:
            continue

        # 尾部词被删了：删除从 b+1 到下一个句子第一个非 gap 词之间的所有静音
        gs = b + 1
        while gs < len(words) and words[gs].get("isGap"):
            if gs not in sel:
                sel.add(gs)
                stats["sil_after_deleted"] += 1
            gs += 1

    # --- 排序 & 写入 ---
    final = sorted(sel)
    stats["total"] = len(final)
    
    with open(auto_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=1)
    
    return auto_path, stats


# ============================================================
#  报告生成
# ============================================================

def generate_report_markdown(
    analysis_dir: Path,
    stats: dict,
    silence_thresh: float = 0.3,
) -> str:
    """生成口误分析 Markdown 报告（含循环审查详情）"""
    
    # 加载 judge decisions（合并格式）做摘要 + 计数
    summaries = {}
    judge_counts = {"inter": 0, "intra": 0, "fragment": 0}
    for name in ["inter", "intra", "fragment"]:
        p = detect_dir(analysis_dir) / f"judge_decisions_{name}.json"
        if not p.exists():
            p = detect_dir(analysis_dir) / f"decisions_{name}.json"  # 兼容旧格式
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            # 新格式: 提取 decision 部分，并按 status 计 applied 项数
            if data and isinstance(data[0], dict) and "detect" in data[0]:
                summaries[name] = [item["decision"] for item in data]
                judge_counts[name] = sum(
                    1 for item in data if item.get("status") == "applied"
                )
            else:
                summaries[name] = data
                judge_counts[name] = len(data)  # 旧格式默认全 applied
        else:
            summaries[name] = []

    loop_path = analysis_dir / "review_loop_decisions.json"
    loop_decisions = []
    if loop_path.exists():
        with open(loop_path, encoding="utf-8") as f:
            loop_decisions = json.load(f)
    # confirmed 位于 decision 子对象内
    applied_loop = [d for d in loop_decisions if d.get("decision", {}).get("confirmed") is True]
    
    md = f"""# 口误分析报告

> 静音阈值：开头/结尾静音必删（不限长度），其余 ≥{silence_thresh}s；被整句删除的句子前后所有静音也删（不留呼吸感）。

## 一、总览
| 类别 | 删除数 | 说明 |
|------|--------|------|
| 开头静音 | {stats['sil_open']} | 开头静音必删（不限长度） |
| 结尾静音 | {stats['sil_tail']} | 结尾静音必删（不限长度） |
| 其他静音 ≥{silence_thresh}s | {stats['sil_body']} | {silence_thresh}s 以内保留 |
| 夹在待删词间的静音 | {stats['sil_internal']} | 两侧说错内容都删时，中间停顿一并删 |
| 整句删除后的静音 | {stats.get('sil_after_deleted', 0)} | 被整句删除的句子前后所有静音也删 |
| 语气词（呃/哎） | {stats['filler']} | 明显犹豫；啊/呀保留 |
| 句间重复 | {judge_counts['inter']} 项 | 被后句完整重说覆盖 |
| 句内重复 | {judge_counts['intra']} 项 | 精确片段删除 |
| 残句 | {judge_counts['fragment']} 项 | 保头删尾 / 整句删 |
| 循环审查补充 | {len(applied_loop)} 项 | Detect+Verify 循环确认删除 |

## 二、关键判断（LLM）
"""
    
    # inter 摘要
    if summaries.get("inter"):
        md += "- **句间重复**：\n"
        for d in summaries["inter"]:
            md += f"  - {d.get('finding', '')}: {d.get('llm_reason', '')}\n"
    
    # intra 摘要
    if summaries.get("intra"):
        md += "\n- **句内重复**：\n"
        for d in summaries["intra"]:
            reason = d.get("llm_reason", "")
            md += f"  - 句{d.get('sentence', '')}: {reason}\n"
    
    # fragment 摘要
    if summaries.get("fragment"):
        md += "\n- **残句**：\n"
        for d in summaries["fragment"]:
            reason = d.get("llm_reason", "")
            mode = d.get("mode", "")
            sent = d.get("sentence", "")
            md += f"  - 句{sent} ({mode}): {reason}\n"
    
    # ====== 循环审查详情（按轮次） ======
    md += _build_loop_section(loop_decisions)

    # ====== 跨轮检测去重记录 ======
    md += _build_dedup_section(analysis_dir)

    md += f"""
## 五、统计汇总

- **总计删除**: {stats['total']} 个 word 索引
- **输出文件**: `auto_selected.json`（整数升序数组）
"""
    
    return md


def _fmt_dedup_finding(det_type: str, fnd: dict) -> str:
    """把一条检测结果格式化为可读身份串（用于报告里展示跳过了/研判了什么）。"""
    if det_type == "inter":
        a = fnd.get("sent_a_idx")
        b = fnd.get("sent_b_idx") or fnd.get("sent_c_idx")
        sub = fnd.get("subtype", "")
        cpl = fnd.get("common_prefix_len")
        extra = f"（同前缀{cpl}字）" if cpl else ""
        return f"句{a}↔句{b} [{sub}]{extra}"
    if det_type == "intra":
        return f"句{fnd.get('sent_idx', '?')}"
    if det_type == "fragment":
        return f"句{fnd.get('sent_idx', '?')} [{fnd.get('subtype', '')}]"
    return str(fnd)


def _build_dedup_section(analysis_dir: Path) -> str:
    """从 detect_history.json 构建「跨轮检测去重记录」章节。

    展示每轮：候选数 / 送研判数 / 跳过(重复或级联)数，并逐条列出被跳过的问题
    及其原因（精确去重 = 前轮已研判过；级联抑制 = 前轮整句删除造成的新邻接句重叠）。
    """
    hist_path = detect_dir(analysis_dir) / "detect_history.json"
    if not hist_path.exists():
        return ""
    try:
        history = json.loads(hist_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not history:
        return ""

    total_cand = sum(
        sum(len(v) for v in r.get("candidates", {}).values()) for r in history
    )
    total_judged = sum(
        sum(len(v) for v in r.get("judged", {}).values()) for r in history
    )
    total_skipped = sum(
        sum(len(v) for v in r.get("skipped", {}).values()) for r in history
    )

    section = (
        f"\n\n## 四、跨轮检测去重记录\n\n"
        f"> 多轮检测中，凡「前轮已研判过的问题」会被精确去重（不再重复送 LLM / 重复应用）；"
        f"前轮整句删除后相邻句变成新邻接对而触发的重叠（级联）也会被抑制。\n\n"
        f"- **检测轮次**: {len(history)} 轮\n"
        f"- **累计候选异常**: {total_cand} 项\n"
        f"- **送 LLM 研判**: {total_judged} 项\n"
        f"- **跳过(重复/级联)**: {total_skipped} 项\n"
    )

    if total_skipped == 0:
        section += "\n> 本轮数据未触发跨轮重复/级联，所有候选均送研判。\n"
        return section

    for r in history:
        rnd = r.get("round", "?")
        cand = r.get("candidates", {})
        judged = r.get("judged", {})
        skipped = r.get("skipped", {})
        ncand = sum(len(v) for v in cand.values())
        njudged = sum(len(v) for v in judged.values())
        nskipped = sum(len(v) for v in skipped.values())
        if nskipped == 0:
            continue  # 只有发生跳过的轮次才展开明细，避免刷屏
        section += (
            f"\n### 第 {rnd} 轮（候选 {ncand} / 送研判 {njudged} / 跳过 {nskipped}）\n\n"
        )
        for det_type in ("inter", "intra", "fragment"):
            items = skipped.get(det_type, [])
            if not items:
                continue
            label = {
                "inter": "句间重复",
                "intra": "句内重复",
                "fragment": "残句",
            }.get(det_type, det_type)
            section += f"- **{label}**（跳过 {len(items)} 项）：\n"
            for it in items:
                fnd = it.get("finding", {})
                reason = it.get("reason", "")
                section += f"  - {_fmt_dedup_finding(det_type, fnd)} → {reason}\n"

    return section


def _load_issues_json(path: Path) -> list:
    """安全加载 issues JSON，失败返回空列表"""
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("issues", []) if isinstance(data, dict) else []
    except Exception:
        return []


def _get_field(obj, key: str, default: str = ""):
    """统一从 dict 或 dataclass 对象中取字段"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _build_loop_section(decisions: list) -> str:
    """从 review_loop_decisions.json 构建循环审查详情（按轮次，含 detect + 确认/驳回）"""
    if not decisions:
        return ""

    def _confirmed(d: dict) -> bool:
        return d.get("decision", {}).get("confirmed") is True

    applied = [d for d in decisions if _confirmed(d)]
    rejected = [d for d in decisions if not _confirmed(d)]

    dim_label = {
        "inter_repeat": "句间重复",
        "intra_repeat": "句内/跨句裁剪",
        "fragment": "残缺句子",
    }
    sev_label = {"critical": "🔴", "major": "🟠", "minor": "🟡"}

    # 按轮次分组全部决策（同时含确认 + 驳回）
    rounds: dict[int, list] = {}
    for d in decisions:
        rounds.setdefault(d.get("round", 0), []).append(d)

    section = (
        f"\n\n## 三、循环审查详情"
        f"（共 {len(decisions)} 项检出：✅ 确认删除 {len(applied)} / ❌ 驳回 {len(rejected)}，"
        f"{len(rounds)} 轮）\n\n"
    )

    for rnd in sorted(rounds.keys()):
        items = rounds[rnd]
        rnd_applied = [d for d in items if _confirmed(d)]
        rnd_rejected = [d for d in items if not _confirmed(d)]
        section += (
            f"### Round {rnd}"
            f"（检出 {len(items)} 项：确认 {len(rnd_applied)} / 驳回 {len(rnd_rejected)}）\n\n"
        )

        for iss in items:
            detect = iss.get("detect", {})
            decision = iss.get("decision", {})
            dim = dim_label.get(detect.get("dimension", ""), detect.get("dimension", "?"))
            sev = sev_label.get(detect.get("severity", ""), "")
            sid = detect.get("sentence_idx", "?")
            err_text = (detect.get("error_text") or detect.get("delete_text", ""))
            reason = (decision.get("reason") or decision.get("llm_reason", ""))
            mark = "✅" if _confirmed(iss) else "❌"

            section += f"- {mark} **[{sev}] [{dim}] 句{sid}** 「{err_text}」\n"
            if reason:
                label = "确认理由" if _confirmed(iss) else "驳回理由"
                section += f"  - {label}: {reason}\n"

        section += "\n"

    return section


# ============================================================
#  updated_sentences.txt 生成
# ============================================================

def generate_updated_sentences(
    analysis_dir: Path,
    words_json_path: Path,
    sentences: list[dict],
    *,
    original: list[dict],
    out_path: Path,
) -> str:
    """
    生成修正后的句子列表文件（应用所有删除后）。

    格式与 sentences.txt 一致：idx|start-end|cleaned_text。直接调用
    write_sentences 写盘，因此：
    - 文本被删空的句子保留为空行（不跳过）；
    - idx 断档（被整句删除）处用 original 的 range 补全空行，保持行号对齐。

    Args:
        sentences: 最终句子列表（已应用 Judge + Loop 删除）。必传，不再从
                   sentences.txt 读取。
        original:  可选，删除前全量句子，用于为断档空行补全 range。
        out_path:  输出文件路径；为 None 时写 updated_sentences.txt（默认）。
                   传 sentences.txt 路径即可直接覆盖原句子文件。

    Returns:
        输出文件的完整文本内容
    """
    auto_path = analysis_dir / "auto_selected.json"

    # 加载删除索引
    with open(auto_path, encoding="utf-8") as f:
        delete_indices = set(json.load(f))

    # 加载 words
    with open(words_json_path, encoding="utf-8") as f:
        words = json.load(f)

    out_sentences: list[dict] = []
    for sent in sentences:
        r = _sentence_range(sent)
        if not r:
            continue
        a, b = r
        # 从 words 中取出该句子的 word，跳过被删除的词
        chars = []
        for wi in range(a, b + 1):
            if wi >= len(words):
                break
            if wi in delete_indices:
                continue
            w = words[wi]
            if not w.get("isGap"):
                chars.append(w["text"])

        cleaned_text = "".join(chars)
        # 保留空行：文本被删空或整句删除后的空行都写出，行号保持对齐
        # （write_sentences 会处理 idx 断档处的空行补全）
        out_sentences.append({
            "idx": sent["idx"],
            "range": f"{a}-{b}",
            "text": cleaned_text,
        })

    write_sentences(out_path, out_sentences, original=original)
    return out_path.read_text(encoding="utf-8")
