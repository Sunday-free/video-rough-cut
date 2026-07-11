"""review_loop_specialized.py — 按错误类型拆分的专职 Agent 循环审查系统（v2 模式）。

架构（对照 agent_review_loop.py 的 v1「单检测 + 单验证」）：
  - Detect 层：3 个专职 detect agent，并行各扫一类错误
      DetectInter / DetectIntra / DetectFragment
  - 确定性 merge + resolve（代码，不调 LLM）：去重、按优先级消解跨类冲突
      （截断续接场景 intra_repeat 优先于 fragment，物理落实「铁律」）
  - Verify 层：3 个专职 verify agent，按类型各验一类
      VerifyInter / VerifyIntra / VerifyFragment
  - Apply + 落盘：与 v1 完全一致的 _merge_decisions / _apply_and_log / _save_and_report

入口 run_agent_review_loop() 与 v1 签名一致，可直接替换；
在 agent_review_loop.py 中通过 run_review_loop(mode="v2") 调度。

为复用 v1 已验证的步骤逻辑（重试/解析/决策对象构建/落盘），本模块在调用时
惰性 import agent_review_loop 的步骤函数（避免模块加载期循环依赖）。
"""

import time
from pathlib import Path

from .agent_prompts_specialized import (
    DETECT_INTER_PROMPT,
    DETECT_INTRA_PROMPT,
    DETECT_FRAGMENT_PROMPT,
    VERIFY_INTER_PROMPT,
    VERIFY_INTRA_PROMPT,
    VERIFY_FRAGMENT_PROMPT,
)
from .agent_apply import (
    apply_deletions_to_sentences,
    build_processed_summary,
)
from ..detect.rule_corroborate import rule_corroborate

MAX_ROUNDS = 10                 # 最大循环轮次
CONSECUTIVE_EMPTY_TO_EXIT = 2   # 连续 N 轮无新问题才退出

# 三类错误 + 各自的专职 prompt
_TYPES = ["inter_repeat", "intra_repeat", "fragment"]
_DETECT_PROMPTS = {
    "inter_repeat": DETECT_INTER_PROMPT,
    "intra_repeat": DETECT_INTRA_PROMPT,
    "fragment": DETECT_FRAGMENT_PROMPT,
}
_VERIFY_PROMPTS = {
    "inter_repeat": VERIFY_INTER_PROMPT,
    "intra_repeat": VERIFY_INTRA_PROMPT,
    "fragment": VERIFY_FRAGMENT_PROMPT,
}


# ============================================================
#  主循环入口（与 v1 run_agent_review_loop 签名一致）
# ============================================================

def run_agent_review_loop(
    analysis_dir: Path,
    loop_dir: Path,
    sentences: list[dict],
    words: list[dict],
    model: str = "deepseek-v4-pro",
    max_rounds: int = MAX_ROUNDS,
    consecutive_empty_to_exit: int = CONSECUTIVE_EMPTY_TO_EXIT,
    use_original_script: bool = False,
    original_script: str = "",
    enable_thinking: bool = False,
    enable_rule_filter: bool = True,
) -> tuple[Path, list[dict], list[dict]]:
    """运行「按类型拆分的专职 Agent」循环审查系统（v2 模式）。

    参数与返回值与 agent_review_loop.run_agent_review_loop 完全一致。
    """
    # 惰性 import：避免与 agent_review_loop 的模块加载期循环依赖
    from .agent_review_loop import (
        _run_detect,
        _run_verify,
        _merge_decisions,
        _apply_and_log,
        _save_and_report,
    )

    output_path = analysis_dir / "review_loop_decisions.json"

    # 仅在启用时保留原稿，否则强制置空
    _effective_script = original_script if use_original_script else ""
    if use_original_script and original_script and original_script.strip():
        print(f"   📜 已启用原稿对照（原稿 {len(original_script.strip())} 字，将注入 detect/verify）")
    elif use_original_script:
        print(f"   ⚠️ 已启用原稿对照，但未提供 original_script，将按无原稿模式运行")

    current_sentences = sentences
    all_decisions: list[dict] = []
    consecutive_empty_rounds = 0

    print("\n" + "=" * 50)
    print("  🔬 专职 Agent 循环审查系统（v2：inter/intra/fragment 三分）")
    print(f"  终止条件: 连续 {consecutive_empty_to_exit} 轮无新问题 / 达到 {max_rounds} 轮上限")
    print("=" * 50 + "\n")

    for round_num in range(1, max_rounds + 1):
        print(f"--- Round {round_num}/{max_rounds} ---\n")

        # === Step 1: 三个专职 Detect 并行扫描 ===
        print(f"[Detect] 三类专职 agent 并行扫描...")
        collected: dict[str, list[dict]] = {}
        detect_failed = False
        for tk in _TYPES:
            issues = _run_detect(
                loop_dir=loop_dir,
                round_num=round_num,
                current_sentences=current_sentences,
                model=model,
                effective_script=_effective_script,
                all_decisions=all_decisions,
                enable_thinking=enable_thinking,
                system_prompt=_DETECT_PROMPTS[tk],
                type_filter=tk,
                stage="detect",
                log_prompt=(tk == _TYPES[0]),
            )
            if issues is None:
                detect_failed = True
                break
            collected[tk] = issues

        if detect_failed:
            break  # 任一 detect LLM 调用失败

        # === Step 1.5: 确定性 merge + resolve（不调 LLM）===
        new_issues = _merge_and_resolve(collected)

        if not new_issues:
            consecutive_empty_rounds += 1
            print(f"   ✅ 无新错误 ({consecutive_empty_rounds}/{consecutive_empty_to_exit} 连续空轮)")
            if consecutive_empty_rounds >= consecutive_empty_to_exit:
                print(f"\n   🏁 连续 {consecutive_empty_to_exit} 轮无新问题，循环结束\n")
                break
            continue

        # 有新问题 → 重置连续空轮计数
        consecutive_empty_rounds = 0

        for i, iss in enumerate(new_issues):
            dim = iss.get("dimension", "?")
            sev = iss.get("severity", "?")
            sid = iss.get("sentence_idx", "?")
            err = (iss.get("error_text") or iss.get("delete_text", ""))[:40]
            print(f"      [{sev}] #{i} {dim}: 句{sid}「{err}」")

        # === Step 1.6: 规则兜底过滤（重复类 issue）===
        if enable_rule_filter and new_issues:
            new_issues, rule_rejected = rule_corroborate(new_issues, current_sentences)
            for iss, reason in rule_rejected:
                v = {"confirmed": False, "reason": f"[规则兜底] {reason}"}
                d = _make_decision_obj_lazy(iss, v, round_num, current_sentences)
                all_decisions.append(d)
                sid = iss.get("sentence_idx")
                dim = iss.get("dimension")
                print(f"      🛡️ 规则过滤 句{sid} {dim}: {reason[:60]}")
            if rule_rejected:
                print(f"   🛡️ 规则兜底过滤 {len(rule_rejected)} 个疑似编造的重复（不进 verify）")
            if not new_issues:
                print(f"   🛡️ 本轮检测到的重复项全部被规则兜底过滤，继续下一轮\n")
                continue

        # === Step 2: 三个专职 Verify 按类型验证 ===
        print(f"\n[Verify] 三类专职 agent 按类型验证 {len(new_issues)} 个候选...")
        verified: list[dict] = []
        for tk in _TYPES:
            grp = [iss for iss in new_issues if iss.get("dimension") == tk]
            if not grp:
                continue
            vgrp = _run_verify(
                loop_dir=loop_dir,
                round_num=round_num,
                new_issues=grp,
                current_sentences=current_sentences,
                model=model,
                effective_script=_effective_script,
                enable_thinking=enable_thinking,
                system_prompt=_VERIFY_PROMPTS[tk],
                stage=f"verify_{tk}",
            )
            verified.extend(vgrp)

        # === Step 3: 合并决策 & 过滤 ===
        round_confirmed = _merge_decisions(
            new_issues=new_issues,
            verified=verified,
            round_num=round_num,
            current_sentences=current_sentences,
            all_decisions=all_decisions,
        )

        # === Step 4: 应用确认的删除 ===
        if not round_confirmed:
            print(f"\n   本轮全部被驳回，继续一下轮\n")
            continue

        current_sentences = _apply_and_log(
            loop_dir=loop_dir,
            round_num=round_num,
            current_sentences=current_sentences,
            words=words,
            round_confirmed=round_confirmed,
        )

    # === 保存最终结果 ===
    _save_and_report(output_path, all_decisions)

    return output_path, all_decisions, current_sentences


# ============================================================
#  确定性 merge + resolve（代码，不调 LLM）
# ============================================================

def _merge_and_resolve(collected: dict[str, list[dict]]) -> list[dict]:
    """把三类 detect 结果合并、去重、按优先级消解跨类冲突。

    - 去重：相同 (dimension, sentence_idx, delete_sentence_idx, delete_text, char_offset) 只保留一份
    - 优先级：截断续接场景 intra_repeat 优先于 fragment——
      若某句同时被 intra_repeat(裁前句尾) 与 fragment(整删前句) 命中，丢弃 fragment，
      物理落实「截断续接铁律：intra 裁尾、绝不 fragment 整删」。
    """
    flat: list[dict] = []
    for tk in _TYPES:
        flat.extend(collected.get(tk, []))

    # 去重
    seen: set[tuple] = set()
    dedup: list[dict] = []
    for iss in flat:
        key = (
            iss.get("dimension"),
            iss.get("sentence_idx"),
            iss.get("delete_sentence_idx"),
            iss.get("delete_text"),
            iss.get("char_offset"),
        )
        if key in seen:
            continue
        seen.add(key)
        dedup.append(iss)

    # 优先级消解：intra_repeat 命中的目标句，其 fragment 整删候选一律丢弃
    intra_targets = {
        iss.get("sentence_idx")
        for iss in dedup
        if iss.get("dimension") == "intra_repeat"
    }
    resolved = [
        iss for iss in dedup
        if not (iss.get("dimension") == "fragment"
                and iss.get("delete_sentence_idx") in intra_targets)
    ]

    dropped = len(dedup) - len(resolved)
    if dropped:
        print(f"   🔧 resolve: 丢弃 {dropped} 个与 intra_repeat 冲突的 fragment 整删候选"
              f"（截断续接铁律：intra 裁尾优先）")

    return resolved


def _make_decision_obj_lazy(iss, v, round_num, current_sentences):
    """惰性调用 v1 的 _make_decision_obj（避免在模块顶层 import agent_review_loop）。"""
    from .agent_review_loop import _make_decision_obj
    return _make_decision_obj(iss, v, round_num, current_sentences)
