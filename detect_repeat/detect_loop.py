"""
detect_loop.py — 机械检测 + LLM 研判 + 应用 多轮循环。

从 pipeline.py 抽离，包含：
  - Judge 决策投影 (_build_delete_indices_from_judge, apply_judge_decisions)
  - 跨轮去重/级联抑制/挂载原稿窗口/忠诚过滤 (_issue_key, _inter_pair, _update_suspect_from_deletions, _prepare_findings_for_judge)
  - 检测主循环 (_run_detect_judge_loop)
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from speech_error_detector.detect_repeat.detect_inter import run_detect_inter
from speech_error_detector.detect_repeat.detect_intra import run_detect_intra
from speech_error_detector.detect_repeat.detect_fragment import run_detect_fragment
from speech_error_detector.detect_repeat.detect_partial import run_detect_partial
from speech_error_detector.detect_repeat.llm_judge import run_all_judges
from speech_error_detector.detect_repeat.script_window import get_org_script_window
from speech_error_detector.detect_repeat.script_faithful import filter_repeat_in_org_window
from speech_error_detector.utils.sentence_io import write_sentences
from speech_error_detector.utils.paths import detect_repeat_dir
from speech_error_detector.config import MAX_DET_ROUNDS


# ============================================================
#  Judge 决策投影
# ============================================================

def _build_delete_indices_from_judge(
    sentences: list[dict],
    analysis_dir: Path,
) -> set[int]:
    """
    从 Layer 2 Judge 的 judge_decisions_*.json 构建要删除的 word 索引集合。
    兼容旧 decisions_*.json 格式（fallback）。

    Args:
        sentences:   句子列表（需含 idx + range 字段）
        analysis_dir: 分析目录（含 judge_decisions_*.json）

    Returns:
        delete_indices: 要删除的 word index 集合
    """
    # 构建句子编号 → word索引范围 映射
    range_map = {}
    for s in sentences:
        idx = s.get("idx")
        if idx is None:
            continue
        rng = s.get("range", "0-0")
        parts = rng.split("-")
        range_map[idx] = (int(parts[0]), int(parts[1]))

    delete_indices: set[int] = set()

    # 辅助：加载决策数据（优先新格式 judge_decisions_*, fallback 旧 decisions_*）
    def _load_decisions(name: str) -> list[dict]:
        path = detect_repeat_dir(analysis_dir) / f"judge_decisions_{name}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return []

    # 只采纳「已确认(applied)」的删除；rejected 项跳过（与 assemble 的
    # _extract_applied_items 保持一致）。
    def _is_applied(d: dict) -> bool:
        return d.get("status") != "rejected"

    # 1) inter_repeat: 删除整句所有 word
    for d in _load_decisions("inter"):
        if not _is_applied(d):
            continue
        # 新格式: {decision: {delete_sentences: [...]}}
        # 旧格式: {delete_sentences: [...]})
        decision = d.get("decision", d)
        for sid in decision.get("delete_sentences", []):
            if sid in range_map:
                a, b = range_map[sid]
                for wi in range(a, b + 1):
                    delete_indices.add(wi)

    # 2) intra_repeat: 删除指定 word 范围
    for d in _load_decisions("intra"):
        if not _is_applied(d):
            continue
        decision = d.get("decision", d)
        for r in decision.get("delete_ranges", []):
            if isinstance(r, list) and len(r) >= 2:
                for wi in range(int(r[0]), int(r[1]) + 1):
                    delete_indices.add(wi)

    # 3) fragment: 整句删除（mode == "full"）或保头删尾（mode == "keep_head"）
    for d in _load_decisions("fragment"):
        if not _is_applied(d):
            continue
        decision = d.get("decision", d)
        sid = decision.get("sentence")
        mode = decision.get("mode", "")
        if sid is None or sid not in range_map:
            continue
        a, b = range_map[sid]
        if mode == "full":
            for wi in range(a, b + 1):
                delete_indices.add(wi)
        elif mode == "keep_head":
            # keep_head: 绝对 word_idx [start, boundary]，保留 boundary 及之前，
            # 删除 boundary+1 起到句末的词。
            kh = decision.get("keep_head")
            if kh:
                if isinstance(kh[0], list):
                    boundary = kh[0][1]          # 二维格式 [[479, 493]]
                elif len(kh) >= 2:
                    boundary = kh[1]             # 一维格式 [479, 493]
                else:
                    boundary = a                 # 回退：保留头部不删
                if boundary < a:                 # 相对索引 → 转绝对
                    boundary = a + boundary
                for wi in range(boundary + 1, b + 1):
                    delete_indices.add(wi)

    # 3b) partial: 句间部分删除（保头删尾 keep_head，或头部完全冗余时 full 整句删）
    for d in _load_decisions("partial"):
        if not _is_applied(d):
            continue
        decision = d.get("decision", d)
        sid = decision.get("sentence")
        mode = decision.get("mode", "")
        if sid is None or sid not in range_map:
            continue
        a, b = range_map[sid]
        if mode == "full":
            for wi in range(a, b + 1):
                delete_indices.add(wi)
        elif mode == "keep_head":
            kh = decision.get("keep_head")
            if kh:
                if isinstance(kh[0], list):
                    boundary = kh[0][1]
                elif len(kh) >= 2:
                    boundary = kh[1]
                else:
                    boundary = a
                if boundary < a:
                    boundary = a + boundary
                for wi in range(boundary + 1, b + 1):
                    delete_indices.add(wi)
   
    return delete_indices


def apply_judge_decisions(
    sentences: list[dict],
    words: list,
    analysis_dir: Path,
    delete_indices: set[int] | None = None,
) -> list[dict]:
    """
    将 Layer 2 Judge 的 decisions_*.json 应用到句子列表上，
    返回修改后的句子副本（text 已去除被删 word）。

    Judge 决策格式：
    - decisions_inter.json:   [{"delete_sentences": [22], ...}]
    - decisions_intra.json:   [{"sentence": 10, "delete_ranges": [[8,12]], ...}]
    - decisions_fragment.json: [{"sentence": 20, "mode": "full", ...}]

    delete_indices: 若提供则直接使用（避免重复计算），否则内部从 judge_decisions_*.json 重建。
    """
    if delete_indices is None:
        delete_indices = _build_delete_indices_from_judge(sentences, analysis_dir)
    
    # 构建修正后句子
    cleaned = []
    for s in sentences:
        idx = s.get("idx")
        rng = s.get("range", "")
        
        # 解析 range
        if rng:
            parts = rng.split("-")
            a, b = int(parts[0]), int(parts[1])
        else:
            cleaned.append({"idx": idx, "range": rng, "text": s.get("text", "")})
            continue
        
        result_text = ""
        for wi in range(a, b + 1):
            if wi >= len(words):
                break
            w = words[wi]
            if w.get("isGap"):
                continue
            if wi not in delete_indices:
                result_text += w["text"]
        
        cleaned.append({
            "idx": idx,
            "range": rng,
            "text": result_text,
        })
    
    return cleaned


# ============================================================
#  跨轮去重 / 级联抑制
# ============================================================

def _issue_key(fnd: dict, det_type: str) -> tuple:
    """跨轮稳定的「问题身份」标识：用于在多轮检测循环中去重，避免对同一问题重复研判/重复应用。

    - inter:  (type, frozenset(句A, 句B))   —— 句间重复以「涉及的两句编号」为身份（同对不再按 subtype 拆多条）
    - intra:  (type, 句编号)                 —— 句内重复以「句子」为身份
    - fragment:(type, 句编号, subtype)       —— 残句以「句子+子类型」为身份
    """
    if det_type == "inter":
        a = fnd.get("sent_a_idx")
        b = fnd.get("sent_b_idx") or fnd.get("sent_c_idx")
        pair = tuple(sorted({a, b}))
        return ("inter", pair)
    if det_type == "intra":
        return ("intra", fnd.get("sent_idx"))
    if det_type == "fragment":
        return ("fragment", fnd.get("sent_idx"), fnd.get("subtype"))
    if det_type == "partial":
        return ("partial", fnd.get("sent_idx"), fnd.get("subtype"))
    return (det_type,)


def _inter_pair(fnd: dict) -> set:
    """返回句间检测涉及的两句编号集合。"""
    a = fnd.get("sent_a_idx")
    b = fnd.get("sent_b_idx") or fnd.get("sent_c_idx")
    return {a, b}


def _update_suspect_from_deletions(
    sentences: list[dict], round_delete: set[int], suspect_idxs: set[int]
) -> None:
    """本轮回被整句删除的句子的相邻句，下一轮会变成新的邻接对，可能触发级联重复检测。

    把相邻句编号加入 suspect_idxs，供下一轮跳过其 inter 检测（避免「删了中间句后又
    冒出相邻两句重叠」这类级联误判）。
    """
    for s in sentences:
        idx = s.get("idx")
        rng = s.get("range", "0-0")
        try:
            a, b = (int(x) for x in rng.split("-"))
        except (ValueError, AttributeError):
            continue
        if b < a:
            continue
        if all(wi in round_delete for wi in range(a, b + 1)):
            # 本轮回整句删除 → 相邻句成为新邻接对，标记可疑
            suspect_idxs.add(idx - 1)
            suspect_idxs.add(idx + 1)


def _suppress_fragment_same_sentence_conflict(findings: dict[str, list]) -> int:
    """Fragment 同句多命中冲突抑制：同一 sent_idx 若同时有「full 候选」与「keep_head 候选」，
    统一保留 full 候选、把 keep_head 候选在送 LLM 前 skip（stage=same_sentence_prefer_full）。

    - "full 候选"：decision_hint 含 "mode=full"（或 subtype 为跨句头-头口误重说）
    - "keep_head 候选"：decision_hint 含 "mode=keep_head"
    仅当同句组内至少存在一个 full 候选时才抑制 keep_head 候选；纯 keep_head 多命中不动。
    已被其它阶段 skip 的 finding 不计入组（如 full 候选本身被忠诚过滤跳过，则不抑制 keep_head）。
    返回被抑制的条数。
    """
    from collections import defaultdict

    _frags = findings.get("fragment", [])
    if len(_frags) < 2:
        return 0

    def _hint_mode(f: dict) -> str:
        _hint = f.get("decision_hint") or ""
        if "mode=full" in _hint:
            return "full"
        if "mode=keep_head" in _hint:
            return "keep_head"
        _sub = f.get("subtype", "")
        if "跨句头-头" in _sub or "口误重说" in _sub:
            return "full"
        return "keep_head"

    _groups: dict[int, list[int]] = defaultdict(list)
    for _i, _f in enumerate(_frags):
        _si = _f.get("sent_idx")
        if isinstance(_si, int) and not _f.get("skip"):
            _groups[_si].append(_i)

    _suppressed = 0
    for _si, _idxs in _groups.items():
        if len(_idxs) < 2:
            continue
        _has_full = any(_hint_mode(_frags[_i]) == "full" for _i in _idxs)
        if not _has_full:
            continue
        for _i in _idxs:
            if _hint_mode(_frags[_i]) == "keep_head":
                _frags[_i]["skip"] = {
                    "stage": "same_sentence_prefer_full",
                    "reason": "同 sent_idx 存在 full 整句删候选，本 keep_head 命中抑制为 skip",
                }
                _suppressed += 1
    return _suppressed


def _suppress_partial_same_sentence(findings: dict[str, list]) -> int:
    """Partial 同句多命中合并：同一 sent_idx 可能命中多个 keep_head 候选
    （如"接续重说"+"尾部重叠"都指向同一前句 A 的尾部）。
    保留「删除最少」的那条（head_char_len 最大 = 保头最多），其余在送 LLM 前 skip，
    避免同一句被多次送审 / 决策冲突。

    返回被抑制的条数。
    """
    from collections import defaultdict

    _parts = findings.get("partial", [])
    if len(_parts) < 2:
        return 0

    _groups: dict[int, list[int]] = defaultdict(list)
    for _i, _f in enumerate(_parts):
        _si = _f.get("sent_idx")
        if isinstance(_si, int) and not _f.get("skip"):
            _groups[_si].append(_i)

    _suppressed = 0
    for _si, _idxs in _groups.items():
        if len(_idxs) < 2:
            continue
        # 选 head_char_len 最大的（保头最多、删得最少）
        _best_i = max(_idxs, key=lambda i: _parts[i].get("head_char_len", 0))
        for _i in _idxs:
            if _i != _best_i:
                _parts[_i]["skip"] = {
                    "stage": "same_sentence_keep_most_head",
                    "reason": "同 sent_idx 存在更保守(保头更多)的 keep_head 候选，本命中抑制为 skip",
                }
                _suppressed += 1
    return _suppressed


def _suppress_full_delete_over_partial(findings: dict[str, list]) -> int:
    """跨检测器同句冲突抑制（优先「删得多」）：同一句若同时被「整句删」检测器
    （inter 删 sent_a_idx 句、fragment 整句删 sent_idx 句，二者皆整句删）与
    「保头删尾」检测器（partial 的 sent_idx 句）命中，则优先整句删（删得多），
    把 partial 候选在送 LLM 前 skip。

    例：句41 同时被 inter(子串完全包含→整句删) 与 partial(非前缀子串重叠→保头删尾) 命中，
    应保留 inter 整句删、抑制 partial，避免 partial 保留的头部与 inter 整句删意图冲突
    （整句删已涵盖该句全部冗余，partial 再保头反而留下应删内容）。

    返回被抑制条数。
    """
    # 1) 收集所有「整句删」目标句索引（inter 删 sent_a；fragment 整句删 sent_idx）
    _full_targets: set[int] = set()
    for _f in findings.get("inter", []):
        _si = _f.get("sent_a_idx")
        if isinstance(_si, int) and not _f.get("skip"):
            _full_targets.add(_si)
    for _f in findings.get("fragment", []):
        _si = _f.get("sent_idx")
        if isinstance(_si, int) and not _f.get("skip"):
            _full_targets.add(_si)

    # 2) partial 命中且目标句已属整句删目标 → 抑制 partial（优先删得多）
    _suppressed = 0
    for _f in findings.get("partial", []):
        if _f.get("skip"):
            continue
        _si = _f.get("sent_idx")
        if isinstance(_si, int) and _si in _full_targets:
            _f["skip"] = {
                "stage": "same_sentence_prefer_full_delete",
                "reason": "同句已被 inter/fragment 整句删命中，partial 保头删尾删得更少，抑制为 skip（优先删得多）",
            }
            _suppressed += 1
    return _suppressed


def _prepare_findings_for_judge(
    findings: dict[str, list],
    decided_keys: set,
    suspect_idxs: set,
    *,
    original_script: str,
    cur_sentences: list[dict],
) -> tuple[set, set, dict, dict]:
    """跨轮去重 + 级联抑制 + 统一挂载原稿对照片段 + 原稿忠诚度过滤。

    已研判/跳过过的问题不再送 LLM；前轮整句删除造成的新邻接句间重叠（级联）也跳过
    （但「子串完全包含」不跳过：A 整句在 B 中说明 A 本身就是残句，并非级联假象，
    仅「头部/尾部重叠」可能是前轮删中间句后碰出的假重叠）。

    同时：
    - get_org_script_window：只对未跳过（去重/抑制）的 finding 处理，避免在各 detect 模块
      里分散重复计算；short=None 由 get_org_script_window 自动判断（极短单句走前后长邻居
      夹窗口）。
    - filter_repeat_in_org_window：原稿窗口内重复（忠于原稿/排比）的重复（序号枚举、并列排比）排除，不送研判；数字对齐
      用增强版 normalize_numerals_full；fragment/inter 用边界感知计数，解决 false-start
      『二人工智能』后跟『新』→不计数→照删，而排比『应唔应该加仓』后跟『，』→计数→排除。
      detector / LLM prompt 零影响。

    **原地**在 findings 上打 skip 标记，不新建 skipped 对象：
        f["skip"] = {"stage": "dedupe_suppress"|"filter_repeat_in_org_window", "reason": "..."}
    调用方据 f 是否含 "skip" 决定送研判，并可据 skip.stage 区分跳过来源。

    Returns: (updated_decided_keys, updated_suspect_idxs, filtered, skipped)
      filtered: 各 category 未 skip 的 finding（送研判）
      skipped:  各 category 已 skip 的 finding
    """
    for _cat, _fnds in findings.items():
        for _fnd in _fnds:
            _key = _issue_key(_fnd, _cat)
            _reason = None
            if _key in decided_keys:
                _reason = "已在前轮研判过(去重)"
            # [DISABLED] 级联抑制逻辑 — 暂时关闭，避免误杀真实问题
            # 原因：common_prefix_len ≥ N 的豁免只绑了"相邻句头重复" subtype，
            # "非前缀子串重叠"等 subtype 即使前缀很长也会被误跳过。
            # TODO: 后续改为通用前缀长度阈值后重新启用
            # elif _cat == "inter" and (_inter_pair(_fnd) & suspect_idxs):
            #     # 「子串完全包含」：A 全部内容在 B 中 → A 本身就是残句，删除中间句
            #     # 只暴露了本就存在的问题，不是级联假象。
            #     # 「相邻句头重复 ≥ 5 字」：共同前缀越长，越不可能是级联碰出的假重叠。
            #     subtype = _fnd.get("subtype", "")
            #     if subtype == "子串完全包含(前句为残句)":
            #         pass  # 不跳过，继续送研判
            #     elif subtype == "相邻句头重复" and _fnd.get("common_prefix_len", 0) >= 5:
            #         pass  # 5 字以上头部重叠不可能由级联碰出
            #     elif subtype.startswith("跨句"):
            #         pass  # 跨句策略跳过≥2 句，不是级联假象
            #     else:
            #         _reason = "涉及前轮整句删除后的新邻接句(级联,跳过)"
            if _reason:
                # 直接在原 candidate 上打标记，不新建 skipped 对象
                _fnd["skip"] = {"stage": "dedupe_suppress", "reason": _reason}
            # 无论本轮是送研判还是被跳过，都记一笔，杜绝下一轮重复检测
            decided_keys.add(_key)

    # 统一挂载原稿对照片段（org_script_window）：只对未跳过（去重/抑制）的
    # finding 处理，避免在各 detect 模块里分散重复计算。short=None 由
    # get_org_script_window 自动判断（极短单句走前后长邻居夹窗口）。
    for _cat_findings in findings.values():
        for _f in _cat_findings:
            if not _f.get("skip"):
                _f["org_script_window"] = get_org_script_window(
                    original_script, cur_sentences, _f,
                )

    # 原稿窗口内重复排除：忠于原稿/排比的重复（序号枚举、并列排比）排除，不送研判
    # 数字对齐用增强版 normalize_numerals_full；fragment/inter 用边界感知计数，
    # 解决 false-start『二人工智能』后跟『新』→不计数→照删，而排比『应唔应该加仓』
    # 后跟『，』→计数→排除。detector / LLM prompt 零影响。
    # 同样直接在原 findings 上打 skip 标记（stage=filter_repeat_in_org_window）。
    filter_repeat_in_org_window(findings, cur_sentences)
    # 同句多命中冲突抑制：同一 sent_idx 同时存在 full 与 keep_head 候选时，抑制 keep_head 候选
    # （full 从根因消除重复，避免保留头部再次与后续句冲突触发跨轮补刀）。送 LLM 前执行。
    _suppress_fragment_same_sentence_conflict(findings)
    _suppress_partial_same_sentence(findings)
    # 跨检测器同句冲突：同一句若被 inter/fragment 整句删 与 partial 保头删尾 同时命中，
    # 优先整句删（删得多），抑制 partial。
    _suppress_full_delete_over_partial(findings)
    # 从已打标的 findings 拆出送研判 / 跳过两个视图（findings 本身保留 skip 标记）
    _filtered = {c: [f for f in findings[c] if not f.get("skip")]
                 for c in ("inter", "intra", "fragment", "partial")}
    _skipped = {c: [f for f in findings[c] if f.get("skip")]
                for c in ("inter", "intra", "fragment", "partial")}

    return decided_keys, suspect_idxs, _filtered, _skipped


def _validate_judge_decisions(
    sentences: list[dict],
    analysis_dir: Path,
) -> list[str]:
    """加载 detect 目录下的 judge_decisions_*.json 并检测格式/语义错误。

    检查项：
    - 文件是否存在、JSON 是否可解析
    - 引用的 sentence ID 是否存在于当前句子列表中
    - delete_ranges / keep_head 范围是否在句子 word 范围内
    - 各字段是否缺失必需键

    Returns:
        错误描述列表（空表示全部正常）
    """
    _det = detect_repeat_dir(analysis_dir)
    errors: list[str] = []

    # 构建句子 idx → range 映射
    _range_map: dict[int, tuple[int, int]] = {}
    for s in sentences:
        idx = s.get("idx")
        if idx is None:
            continue
        r = s.get("range", "0-0")
        try:
            a, b = (int(x) for x in r.split("-"))
            _range_map[idx] = (a, b)
        except (ValueError, AttributeError):
            continue

    # 四大类 decisions 文件
    _files = {
        "inter": _det / "judge_decisions_inter.json",
        "intra": _det / "judge_decisions_intra.json",
        "fragment": _det / "judge_decisions_fragment.json",
        "partial": _det / "judge_decisions_partial.json",
    }

    for _cat, _fpath in _files.items():
        if not _fpath.exists():
            continue
        _actual_path = _fpath

        try:
            _data = json.loads(_actual_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            errors.append(f"{_cat}: JSON 解析失败 ({_actual_path.name}): {e}")
            continue
        if not isinstance(_data, list):
            errors.append(f"{_cat}: 期望 JSON 数组，实际为 {type(_data).__name__}")
            continue

        for i, d in enumerate(_data):
            _prefix = f"{_cat}[{i}]"

            # inter: {decision: {delete_sentences: [...]}} 或 {delete_sentences: [...]}
            if _cat == "inter":
                _decision = d.get("decision", d)
                _del_sids = _decision.get("delete_sentences", [])
                if not isinstance(_del_sids, list):
                    errors.append(f"{_prefix}: delete_sentences 应为数组")
                    continue
                for sid in _del_sids:
                    if sid not in _range_map:
                        errors.append(f"{_prefix}: 引用的句子 {sid} 不在当前句子列表中")

            # intra: {decision: {delete_ranges: [[a,b],...]}} 或 {delete_ranges: [...]}
            elif _cat == "intra":
                _decision = d.get("decision", d)
                _del_ranges = _decision.get("delete_ranges", [])
                if not isinstance(_del_ranges, list):
                    errors.append(f"{_prefix}: delete_ranges 应为数组")
                    continue
                _sid = _decision.get("sentence") or d.get("sentence")
                if _sid is not None and _sid not in _range_map:
                    errors.append(f"{_prefix}: 引用的句子 {_sid} 不在当前句子列表中")
                    continue
                if _sid is not None and _sid in _range_map:
                    _sa, _sb = _range_map[_sid]
                    for _r in _del_ranges:
                        if isinstance(_r, list) and len(_r) >= 2:
                            if int(_r[0]) < _sa or int(_r[1]) > _sb:
                                errors.append(
                                    f"{_prefix}: delete_range {_r} 超出句子 {_sid} 范围 [{_sa},{_sb}]"
                                )

            # fragment: {decision: {sentence, mode, keep_head?}} 或 {sentence, mode}
            elif _cat == "fragment":
                _decision = d.get("decision", d)
                _sid = _decision.get("sentence")
                _mode = _decision.get("mode", "")
                if _sid is None:
                    errors.append(f"{_prefix}: 缺少 sentence 字段")
                    continue
                if _sid not in _range_map:
                    errors.append(f"{_prefix}: 引用的句子 {_sid} 不在当前句子列表中")
                    continue
                if _mode == "keep_head":
                    _kh = _decision.get("keep_head")
                    if _kh is None:
                        errors.append(f"{_prefix}: mode=keep_head 但缺少 keep_head 字段")
                    else:
                        _sa, _sb = _range_map[_sid]
                        _boundary = _kh[1] if len(_kh) >= 2 else (_kh[0][1] if isinstance(_kh[0], list) and len(_kh[0]) >= 2 else None)
                        if _boundary is not None:
                            # 可能是相对或绝对索引
                            if _boundary < _sa:
                                _boundary = _sa + _boundary
                            if _boundary > _sb:
                                errors.append(
                                    f"{_prefix}: keep_head boundary {_kh[-1]} 超出句子 {_sid} 范围 [{_sa},{_sb}]"
                                )

            # partial: 与 fragment 同构（sentence, mode, keep_head?），partial 仅产出 keep_head / full
            elif _cat == "partial":
                _decision = d.get("decision", d)
                _sid = _decision.get("sentence")
                _mode = _decision.get("mode", "")
                if _sid is None:
                    errors.append(f"{_prefix}: 缺少 sentence 字段")
                    continue
                if _sid not in _range_map:
                    errors.append(f"{_prefix}: 引用的句子 {_sid} 不在当前句子列表中")
                    continue
                if _mode == "keep_head":
                    _kh = _decision.get("keep_head")
                    if _kh is None:
                        errors.append(f"{_prefix}: mode=keep_head 但缺少 keep_head 字段")
                    else:
                        _sa, _sb = _range_map[_sid]
                        _boundary = _kh[1] if len(_kh) >= 2 else (_kh[0][1] if isinstance(_kh[0], list) and len(_kh[0]) >= 2 else None)
                        if _boundary is not None:
                            if _boundary < _sa:
                                _boundary = _sa + _boundary
                            if _boundary > _sb:
                                errors.append(
                                    f"{_prefix}: keep_head boundary {_kh[-1]} 超出句子 {_sid} 范围 [{_sa},{_sb}]"
                                )

    return errors


# ============================================================
#  检测主循环
# ============================================================
# MAX_DET_ROUNDS 取自 config（与 run_pipeline 默认对齐）

def run_detect_judge_loop(
    *,
    sentences: list[dict],
    analysis_dir: Path,
    words_data: list,
    sentences_path: Path,
    model: str,
    enable_deepseek_thinking: bool,
    skip_judge: bool,
    original_script: str,
    max_det_rounds: int = MAX_DET_ROUNDS,
) -> tuple[list[dict], int, int]:
    """步骤1-4: 机械检测 + LLM 研判 + 应用，多轮循环直到收敛。

    每轮：并行机械检测 → LLM 研判 → 把决策投影(apply)进 sentences.txt。
    若某轮研判后没有产生任何「新」word 删除（即无新 issue），说明已收敛，提前结束。

    Returns:
        (机械检测清洗后的句子, 累计候选异常数, 实际运行轮数)
    """
    cur_sentences = sentences
    accumulated_delete: set[int] = set()
    total_candidates = 0
    det_round = 0

    _detect_fns = {
        "inter": run_detect_inter,
        "intra": run_detect_intra,
        "fragment": run_detect_fragment,
        "partial": run_detect_partial,
    }
    # 跨轮累积的研判决策（供报告展示全部 applied 项，避免被后续轮覆盖丢失）
    _all_judge_decisions: dict[str, list] = {"inter": [], "intra": [], "fragment": [], "partial": []}
    # 跨轮去重状态：已研判/跳过的问题身份集合 + 级联抑制集合 + 每轮检测历史
    decided_keys: set = set()
    suspect_idxs: set = set()
    detect_history: list = []

    if skip_judge:
        # 跳过 LLM 研判：加载 detect 目录下已有的 judge_decisions_*.json，验证后投影一次
        print("-" * 40)
        print("[步骤1-4] 跳过 LLM 研判，加载 detect/ 目录已有 judge_decisions_*.json")
        print("-" * 40)
        det_round = 1

        # 验证 judge_decisions_*.json 是否存在且格式正确
        _validation_errors = _validate_judge_decisions(
            sentences=cur_sentences, analysis_dir=analysis_dir,
        )
        if _validation_errors:
            print(f"  \033[33m[警告] judge_decisions 存在 {len(_validation_errors)} 处问题：\033[0m")
            for _ve in _validation_errors:
                print(f"    - {_ve}")
        else:
            print(f"  [验证通过] judge_decisions_*.json 格式正确，无错误")

        _round_delete = _build_delete_indices_from_judge(
            sentences=cur_sentences, analysis_dir=analysis_dir,
        )
        accumulated_delete |= _round_delete
        cur_sentences = apply_judge_decisions(
            sentences=cur_sentences, words=words_data,
            analysis_dir=analysis_dir, delete_indices=_round_delete,
        )
        print(f"  应用完成：删除 {len(_round_delete)} 个 word 索引")
        return cur_sentences, total_candidates, det_round
        
    while det_round < max_det_rounds:
        det_round += 1
        print("-" * 40)
        print(f"[步骤1-4] 机械检测 + 研判 + 应用 (第 {det_round}/{max_det_rounds} 轮)")
        print("-" * 40)

        # 保存当前轮的 sentences（方便查看每轮检测时的输入句子状态）
        write_sentences(
            detect_repeat_dir(analysis_dir) / f"sentences_round_{det_round}.txt",
            cur_sentences,
        )

        # 1) 并行机械检测（基于当前已 applied 的 sentences.txt）
        # 忽略 text 为空的句子（已被前轮整句删除），避免对已删句子重复检测
        _detect_sents: list[dict[Any, Any]] = [s for s in cur_sentences if s.get("text", "").strip()]
        _findings: dict[str, list] = {}
        with ThreadPoolExecutor(max_workers=3) as _ex:
            _futs = {}
            for _name, _fn in _detect_fns.items():
                _futs[_ex.submit(_fn, _detect_sents, analysis_dir, words_data, original_script)] = _name
            for _fut in _futs:
                _findings[_futs[_fut]] = _fut.result()
        _fi = _findings["inter"]
        _fint = _findings["intra"]
        _fra = _findings["fragment"]
        _fpar = _findings["partial"]
        _round_candidates = len(_fi) + len(_fint) + len(_fra) + len(_fpar)

        # 2) 跨轮去重 + 级联抑制 + 统一挂载原稿对照片段 + 原稿窗口内重复排除
        #    （全部直接在原 findings 上打 skip 标记，stage 区分 dedupe_suppress /
        #    filter_repeat_in_org_window，不新建 skipped 对象）
        decided_keys, suspect_idxs, _filtered, _skipped = _prepare_findings_for_judge(
            _findings, decided_keys, suspect_idxs,
            original_script=original_script, cur_sentences=cur_sentences,
        )
        _judged = sum(len(v) for v in _filtered.values())
        _skipped_n = sum(len(v) for v in _skipped.values())
        total_candidates += _round_candidates
        print(f"\n  本轮候选异常: {_round_candidates} "
                f"(句间{len(_fi)} + 句内{len(_fint)} + 残句{len(_fra)} + 部分删{len(_fpar)})"
                f" | 送研判 {_judged} | 跳过(去重/忠诚) {_skipped_n}")

        # 3) LLM 研判（仅送 _filtered；跳过项不送，避免重复检测 + 重复花销）
        judge_results = run_all_judges(
            analysis_dir=detect_repeat_dir(analysis_dir),
            sentences=cur_sentences,
            detect_data=_filtered,
            model=model,
            enable_thinking=enable_deepseek_thinking,
            round_idx=det_round,
        )
        print(f"  研判完成，共 {len(judge_results)} 种检测器")

        # 累积本轮回决策（供报告展示全部 applied 项）
        for _cat in ("inter", "intra", "fragment", "partial"):
            _items = judge_results.get(_cat, (None, []))[1]
            if _items:
                _all_judge_decisions[_cat].extend(_items)

        # 4) applied：把本轮回决策投影为 word 级删除并应用进 sentences
        _round_delete = _build_delete_indices_from_judge(
            sentences=cur_sentences, analysis_dir=analysis_dir,
        )
        _new_delete = _round_delete - accumulated_delete
        accumulated_delete |= _round_delete
        # 更新级联抑制集合：本轮回被整句删除的句子的相邻句
        _update_suspect_from_deletions(cur_sentences, _round_delete, suspect_idxs)
        # ⚠️ 必须用累积删除集 accumulated_delete，而非单轮 _round_delete。
        # 原因：apply_judge_decisions 是用「句子原始 range + words_data」重新拼 text、
        # 跳过 delete_indices 中的词；若某轮 _round_delete 因 judge 文件被后续轮次覆盖
        # 而漏掉早期轮次已确认的删除（如 fragment 31/32），apply 会把那些词重新拼回
        # text（表现为「删除 -N 字」的回潮）。累积集始终保留全部已确认删除，可杜绝回潮。
        _sil_orig = sum(len(s.get("text", "")) for s in cur_sentences)
        cleaned = apply_judge_decisions(
            sentences=cur_sentences, words=words_data,
            analysis_dir=analysis_dir, delete_indices=accumulated_delete,
        )
        _sil_after = sum(len(s.get("text", "")) for s in cleaned)
        write_sentences(sentences_path, cleaned)
        print(f"  第 {det_round} 轮 applied: 删除 {_sil_orig - _sil_after} 字"
                f"（新增 {len(_new_delete)} 个 word 索引）")
        cur_sentences = cleaned

        # 记录本轮检测（供检查「每轮检测了什么」、以及去重/忠诚核对）
        # 注意：candidates 即原始检测结果，已被 _prepare_findings_for_judge /
        # (dedupe_suppress / filter_repeat_in_org_window) 原地打上 skip 标记（skip.stage 区分来源），
        # 不再单独维护一份 skipped 对象。judged 为实际送 LLM 的子集。
        # ⚠️ 落盘前：candidates 中 stage=dedupe_suppress（跨轮精确去重，前轮已研判过）的项
        # 不写入 history / roundN.json —— 它们本就是前轮残留，落盘既冗余又易与「本轮新发现」混淆。
        # （stage=filter_repeat_in_org_window 忠于原稿/排比的原稿窗口重复排除仍保留，便于核对。）
        _candidates_non_dedupe = {
            c: [f for f in _findings[c]
                if f.get("skip", {}).get("stage") != "dedupe_suppress"]
            for c in ("inter", "intra", "fragment", "partial")
        }
        # judged = candidates 中未标 skip 的部分，可由「candidates - 带 skip 的项」推导，
        # 不再单独存盘（见 assemble.py 用 ncand - nskipped 计算送研判数）。
        detect_history.append({
            "round": det_round,
            "candidates": _candidates_non_dedupe,
            "delete_indices_count": len(_round_delete),
            "new_delete_count": len(_new_delete),
        })
        (detect_repeat_dir(analysis_dir) / f"detect_round_{det_round}.json").write_text(
            json.dumps(detect_history[-1], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 收敛判定：本轮没有任何删除（研判全拒绝/无 issue）→ 下一轮在完全相同
        # 文本上检测结果必然一致，无新 issue，直接结束，避免无谓的 LLM 调用。
        if not _round_delete:
            print(f"  ✓ 第 {det_round} 轮无删除，检测已收敛，提前结束循环")
            break
        # 已有删除但无「新增」删除（仅重复确认已删项）→ 同样收敛
        if det_round > 1 and not _new_delete:
            print(f"  ✓ 第 {det_round} 轮无新增删除，检测已收敛，提前结束循环")
            break

    # 循环结束后，cleaned 即机械检测阶段累计清洗结果（= cur_sentences）
    cleaned = cur_sentences
   
    # 写出跨轮检测历史（每轮检测了什么、哪些被去重/级联跳过），便于核对不重复检测
    (detect_repeat_dir(analysis_dir) / "detect_history.json").write_text(
        json.dumps(detect_history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 把跨轮累积的研判决策写回磁盘（覆盖最后一轮，确保报告统计的是全部 applied 项）
    # 注意：即使某类本轮为空也要写入空数组，覆盖上一次运行遗留的旧文件，
    # 否则检测器改进后被过滤掉的误报仍会残留在旧 JSON 里（如 intra 并列枚举）。
    for _cat in ("inter", "intra", "fragment", "partial"):
        (detect_repeat_dir(analysis_dir) / f"judge_decisions_{_cat}.json").write_text(
            json.dumps(_all_judge_decisions[_cat], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    decisions_path = analysis_dir / "decisions.json"
    decisions_path.write_text(
        json.dumps(sorted(accumulated_delete), ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"   机械检测累计待删 word 数: {len(accumulated_delete)} → 已写 {decisions_path.name}")

    print(f"   已更新: {sentences_path.name} (机械检测清洗后, 共 {det_round} 轮)")

    return cleaned, total_candidates, det_round
