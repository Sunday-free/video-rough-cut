"""
agent_review_loop.py — Agent 循环审查系统

架构设计:
  输入: Layer 2 研判后的 sentences.txt（已应用 L1+L2 删除）
  循环: detect_agent(识别错误) → verify_agent(验证) → apply(应用) → 下一轮
  终止: 某轮 detect 返回空（无新错误）
  输出: review_loop_decisions.json（所有轮次的 识别+决策 合并为同一对象数组）

每个对象格式:
{
    "round": <轮次>,
    "detect": { dimension, severity, sentence_idx, delete_sentence_idx,
                error_text, delete_text, char_offset, description },
    "decision": { confirmed: bool, reason: str },
    "status": "applied" | "rejected"
}

run_agent_review_loop() 仅做编排，四个步骤分别委托给:
  _run_detect() / _run_verify() / _merge_decisions()(+_make_decision_obj) / _apply_and_log()
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from .deepseek_client import deepseek_chat
from .llm_parse import parse_json_object, parse_json_array

from .agent_prompts import (
    DETECT_SYSTEM_PROMPT,
    VERIFY_SYSTEM_PROMPT,
    build_detect_prompt,
    build_verify_prompt,
    is_modal_only_delete,
)
from .agent_apply import (
    apply_deletions_to_sentences,
    build_processed_summary,
)
from ..detect.rule_corroborate import rule_corroborate

MAX_ROUNDS = 10          # 最大循环轮次，防止无限循环
CONSECUTIVE_EMPTY_TO_EXIT = 2  # 连续N轮无新问题才退出
DETECT_MAX_RETRY = 3     # 单轮 Detect LLM 硬失败（抛异常）时的重试次数
DETECT_RETRY_BACKOFF = 2.0  # 重试退避基数（秒），第 n 次重试等待 n * backoff


# ============================================================
#  辅助：prompt / 句子 日志落盘
# ============================================================

def _log_prompt(loop_dir: Path, round_num: int, stage: str, prompt_text: str) -> None:
    """记录本轮 detect/verify prompt 到 loop_round{N}_{stage}.txt。"""
    path = loop_dir / f"loop_round{round_num}_{stage}.txt"
    with open(path, "w", encoding="utf-8") as pf:
        pf.write(f"===== Round {round_num} | {stage.upper()} PROMPT =====\n\n")
        pf.write(prompt_text)
        pf.write("\n\n")


# ============================================================
#  辅助：单字重复判定
# ============================================================

def _is_single_char_repeat(iss: dict) -> bool:
    """单字重复判定：重复类 issue 且其删除文本仅 1 个字符（如「是」单字赘余）。

    按系统规则「单字错误不处理」，这类不进 verify，避免浪费 LLM 调用并污染决策日志。
    """
    dim = iss.get("dimension", "")
    if dim not in ("intra_repeat", "inter_repeat"):
        return False
    dt = (iss.get("delete_text") or "").strip()
    return len(dt) == 1


def _is_self_retract(iss: dict) -> bool:
    """自检撤回判定：detect 输出的 description 为「不报」等自我否定短语时，
    说明模型自己认为该候选不该报（典型场景：round≥2 复报已被驳回的旧 issue，
    或 detect 认得系统规则却仍把候选 emit 出来、在 description 里自我圆场写「不报」）。

    这类候选与「单字重复」一样，进 verify 也只是被驳回，纯属浪费 LLM 调用并污染
    决策日志，因此在 detect 出口直接丢弃，不进 verify、不进 decisions 日志。

    判定：description 去除首尾空白与末尾中文/英文句号后，命中自我否定短语集合。
    """
    desc = (iss.get("description") or "").strip().rstrip("。！？.!?；;")
    return desc in (
        "不报", "不报告", "无需报", "无需报告",
        "不应报", "不应报告", "不处理", "不建议报",
    )


# ============================================================
#  Step 1: Detect Agent
# ============================================================

def _run_detect(
    loop_dir: Path,
    round_num: int,
    current_sentences: list[dict],
    model: str,
    effective_script: str,
    all_decisions: list[dict],
    enable_thinking: bool = False,
    system_prompt: str = DETECT_SYSTEM_PROMPT,
    type_filter: str | None = None,
    stage: str = "detect",
    log_prompt: bool = True,
) -> list[dict] | None:
    """执行一轮 Detect：构建 prompt、记录日志、调用 LLM、解析、过滤纯语气词删除项。

    Args:
        system_prompt: 注入的 system prompt；专职模式（v2）传入按类型缩窄的 prompt，
            默认沿用通用 DETECT_SYSTEM_PROMPT（v1 行为不变）。
        type_filter: 若非 None，仅保留 dimension == type_filter 的候选，丢弃其他跨类候选。
        stage: 日志文件名片段。v2 三类专职 detect 共用同一份 user prompt（与类型无关），
            因此统一记为 detect，每轮只落盘一个 loop_round{N}_detect.txt（由 log_prompt 控制）。
        log_prompt: 是否落盘 detect prompt。v2 三个专职 detect 的 user prompt 完全相同，
            只需落盘一次（首个类型），其余调用置 False 以避免重复写同一文件。

    Returns:
        - 成功: 过滤后的 new_issues（list[dict]）
        - LLM 调用失败: None（调用方据此 break 退出循环）
    """
    processed_summary = build_processed_summary(all_decisions)
    detect_prompt = build_detect_prompt(
        sentences=current_sentences,
        original_script=effective_script,
        processed_summary=processed_summary,
    )
    if log_prompt:
        _log_prompt(loop_dir, round_num, stage, detect_prompt)

    t0 = time.time()
    detect_response = None
    last_err: Exception | None = None
    for attempt in range(1, DETECT_MAX_RETRY + 1):
        try:
            detect_response = deepseek_chat(
                system=system_prompt,
                user=detect_prompt,
                model=model,
                temperature=0.0,
                enable_thinking=enable_thinking,
            )
            break
        except Exception as e:
            last_err = e
            if attempt < DETECT_MAX_RETRY:
                wait = attempt * DETECT_RETRY_BACKOFF
                print(f"   ⚠️ Detect LLM 调用失败（第 {attempt}/{DETECT_MAX_RETRY} 次）: {e} → {wait:.0f}s 后重试")
                time.sleep(wait)
            else:
                print(f"   ⚠️ Detect LLM 连续 {DETECT_MAX_RETRY} 次调用失败: {e}")
    if detect_response is None:
        return None

    detect_elapsed = time.time() - t0
    detect_result = parse_json_object(
        detect_response, default={"issues": [], "_parse_error": True}
    )
    new_issues = detect_result.get("issues", [])

    # 专职模式（v2）：仅保留本类型候选，丢弃 detect 越界报出的其他类
    if type_filter is not None:
        before = len(new_issues)
        new_issues = [iss for iss in new_issues if iss.get("dimension") == type_filter]
        dropped = before - len(new_issues)
        if dropped:
            print(f"   ⏩ 过滤掉 {dropped} 个非 {type_filter} 的跨类型候选（专职 agent 越界）")

    # 过滤：只删除语气词（纯模态字）的检测项不处理，直接丢弃
    # （如「呃」重复出现，属自然语流，不在本系统处理范围内）
    before = len(new_issues)
    new_issues = [
        iss for iss in new_issues
        if not is_modal_only_delete(iss.get("delete_text", ""))
    ]
    if len(new_issues) != before:
        print(f"   ⏩ 过滤掉 {before - len(new_issues)} 个纯语气词删除项（不处理）")

    # 过滤：单字重复（delete_text 仅 1 个字符）按「单字错误不处理」策略直接丢弃，
    # 不进 verify，避免浪费 LLM 调用并污染决策日志（如「持仓的朋友既开心是又忐忑」
    # 里的赘余「是」）。这类在之前会每轮被重复 detect + verify 但仍一致驳回，纯属浪费。
    before = len(new_issues)
    new_issues = [
        iss for iss in new_issues
        if not _is_single_char_repeat(iss)
    ]
    if len(new_issues) != before:
        print(f"   ⏩ 过滤掉 {before - len(new_issues)} 个单字重复项（delete_text 仅 1 字，不处理）")

    # 过滤：自检撤回（detect 自己标「不报」的候选）直接丢弃，不进 verify。
    # 典型场景：round≥2 复报已被驳回的旧 issue，或 detect 认得规则（如单字不处理）
    # 却仍把候选 emit 出来、在 description 里写「不报」自我圆场。进 verify 也只是被
    # 驳回，纯属浪费 LLM 调用并污染决策日志（对应 review_loop_decisions.json 里
    # decision_hint/description 同写为「不报」的噪声项）。
    before = len(new_issues)
    new_issues = [
        iss for iss in new_issues
        if not _is_self_retract(iss)
    ]
    if len(new_issues) != before:
        print(f"   ⏩ 过滤掉 {before - len(new_issues)} 个自检撤回项（detect 自标「不报」，不进 verify）")

    print(f"   Detect 完成 ({detect_elapsed:.1f}s): 发现 {len(new_issues)} 个候选问题")
    return new_issues


# ============================================================
#  Step 2: Verify Agent
# ============================================================

def _run_verify(
    loop_dir: Path,
    round_num: int,
    new_issues: list[dict],
    current_sentences: list[dict],
    model: str,
    effective_script: str,
    enable_thinking: bool = False,
    system_prompt: str = VERIFY_SYSTEM_PROMPT,
    stage: str = "verify",
) -> list[dict]:
    """执行一轮 Verify：构建 prompt、记录日志、调用 LLM、解析为决策数组。

    Args:
        system_prompt: 注入的 system prompt；专职模式（v2）传入按类型缩窄的 prompt，
            默认沿用通用 VERIFY_SYSTEM_PROMPT（v1 行为不变）。
        stage: 日志文件名片段（v2 按类型记为 verify_{type}）。

    LLM 调用失败时返回全部"不确认"占位，保持与输入一一对应。
    """
    verify_prompt = build_verify_prompt(
        issues=new_issues,
        sentences=current_sentences,
        original_script=effective_script,
    )
    _log_prompt(loop_dir, round_num, stage, verify_prompt)

    t0 = time.time()
    try:
        verify_response = deepseek_chat(
            system=system_prompt,
            user=verify_prompt,
            model=model,
            temperature=0.0,
            enable_thinking=enable_thinking,
        )
    except Exception as e:
        print(f"   ⚠️ Verify LLM 调用失败: {e}")
        # 全部默认不确认
        return [
            {"index": i, "confirmed": False, "reason": f"LLM 调用失败: {e}"}
            for i in range(len(new_issues))
        ]

    verify_elapsed = time.time() - t0
    verified = parse_json_array(verify_response, expected=len(new_issues))
    print(f"   Verify 完成 ({verify_elapsed:.1f}s)")
    return verified


# ============================================================
#  Step 3: 合并决策 & 过滤
# ============================================================

_SUBTYPE_MAP = {
    "inter_repeat": "句间重复",
    "intra_repeat": "句内重复",
    "fragment": "残句",
}


def _make_decision_obj(
    iss: dict,
    v: dict,
    round_num: int,
    current_sentences: list[dict],
) -> dict:
    """根据单个 detect issue 与其 verify 结果，构建嵌套格式 decision_obj。"""
    dim = iss.get("dimension", "")
    sid = iss.get("sentence_idx")
    delete_sid = iss.get("delete_sentence_idx", sid)

    # 确定目标句子（要操作的那个）
    target_sid = delete_sid if dim == "inter_repeat" else sid

    # 获取目标句子的 range 和 text
    target_range = "?"
    target_text = ""
    for s in current_sentences:
        if s.get("idx") == target_sid:
            target_range = s.get("range", "?")
            target_text = s.get("text", "")
            break

    # 整段删除（inter_repeat/fragment）确认后 mode="full"；片段裁剪(intra_repeat)无 mode；驳回无 mode
    mode = "full" if (v.get("confirmed") and dim in ("inter_repeat", "fragment")) else None

    return {
        "round": round_num,
        "detect": {
            "type": dim,
            "subtype": _SUBTYPE_MAP.get(dim, ""),
            "sent_idx": target_sid,
            "range": target_range,
            "text": target_text[:80] if len(target_text) > 80 else target_text,
            "decision_hint": iss.get("description", ""),
            # 兼容 assemble 的原始字段
            "dimension": dim,
            "severity": iss.get("severity", ""),
            "sentence_idx": iss.get("sentence_idx"),
            "delete_sentence_idx": iss.get("delete_sentence_idx"),
            "error_text": iss.get("error_text", ""),
            "delete_text": iss.get("delete_text", ""),
            "char_offset": iss.get("char_offset"),
            "description": iss.get("description", ""),
        },
        "decision": {
            "sentence": target_sid,
            "mode": mode,
            "keep_head": None,
            "llm_reason": v.get("reason", ""),
            "confirmed": v.get("confirmed"),
            "reason": v.get("reason", ""),
        },
    }


def _merge_decisions(
    new_issues: list[dict],
    verified: list[dict],
    round_num: int,
    current_sentences: list[dict],
    all_decisions: list[dict],
) -> list[dict]:
    """将本轮 detect+verify 合并写入 all_decisions，返回本轮确认的 issue 列表。

    每个 issue 打印确认/驳回小结。
    """
    round_confirmed: list[dict] = []
    for i, iss in enumerate(new_issues):
        v = verified[i] if i < len(verified) else {
            "index": i, "confirmed": False, "reason": "无验证结果",
        }
        decision_obj = _make_decision_obj(iss, v, round_num, current_sentences)
        all_decisions.append(decision_obj)

        if v.get("confirmed"):
            round_confirmed.append(iss)
            print(f"      ✅ #{i} 确认: {v.get('reason', '')[:60]}")
        else:
            print(f"      ❌ #{i} 驳回: {v.get('reason', '')[:60]}")
    return round_confirmed


# ============================================================
#  Step 4: 应用确认删除 + 落盘本轮句子
# ============================================================

def _apply_and_log(
    loop_dir: Path,
    round_num: int,
    current_sentences: list[dict],
    words: list[dict],
    round_confirmed: list[dict],
) -> list[dict]:
    """应用本轮确认删除、打印统计、记录 apply 后句子到 loop_round{N}_sentences.txt。

    Returns:
        应用删除后的 current_sentences（idx/range 保持不变）
    """
    print(f"\n[Apply] 应用 {len(round_confirmed)} 个确认删除...")
    current_sentences = apply_deletions_to_sentences(
        sentences=current_sentences,
        words=words,
        confirmed_issues=round_confirmed,
    )

    total_chars_after = sum(len(s.get("text", "")) for s in current_sentences)
    print(f"   应用完成，当前文本总长: {total_chars_after} 字\n")

    _sent_log_path = loop_dir / f"loop_round{round_num}_sentences.txt"
    with open(_sent_log_path, "w", encoding="utf-8") as _sf:
        _sf.write(f"===== Round {round_num} | SENTENCES AFTER APPLY =====\n")
        _sf.write(f"总句数: {len(current_sentences)}  总字数: {total_chars_after}\n\n")
        for s in current_sentences:
            _sf.write(f"[句{s.get('idx')}] {s.get('text', '')}\n")

    return current_sentences


# ============================================================
#  收尾：保存结果 + 汇总打印
# ============================================================

def _save_and_report(output_path: Path, all_decisions: list[dict]) -> None:
    """保存 review_loop_decisions.json 并打印汇总。"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_decisions, f, ensure_ascii=False, indent=2)

    applied = sum(1 for d in all_decisions if d.get("decision", {}).get("confirmed"))
    rejected = sum(1 for d in all_decisions if not d.get("decision", {}).get("confirmed"))

    print("=" * 50)
    last_round = max((d.get("round", 0) for d in all_decisions), default=0)
    print(f"  循环结束！共 {last_round} 轮")
    print(f"  总决策: {len(all_decisions)} (✅ 确认 {applied}, ❌ 驳回 {rejected})")
    print(f"  结果文件: {output_path.name}")
    print("=" * 50 + "\n")


# ============================================================
#  主循环入口
# ============================================================

def run_agent_review_loop_v1(
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
    """
    运行 Agent 循环审查系统（v1：单检测 + 单验证）。

    Args:
        analysis_dir:     分析目录
        words:            subtitles_words.json 的字级数据
        model:            LLM 模型名
        max_rounds:       最大循环轮次
        use_original_script: 是否启用原稿对照（把作者原稿注入 detect/verify prompt）
        original_script:  作者原稿文本（当 use_original_script=True 时传入才生效）
        sentences:       初始句子列表（已应用 judge 删除后的版本），直接使用；
                          返回时回传循环结束后的最终句子列表，
                          供下游 run_assemble 复用同一引用。

    Returns:
        (output_path, all_decisions, current_sentences)
        output_path → review_loop_decisions.json
        all_decisions → 所有轮次的 识别+决策 合并数组
        current_sentences → 循环结束后的最终句子列表（idx/range 保持不变）
    """
    output_path = analysis_dir / "review_loop_decisions.json"

    # 仅在启用时保留原稿，否则强制置空（避免两处 prompt 无意使用到）
    _effective_script = original_script if use_original_script else ""
    if use_original_script and original_script and original_script.strip():
        print(f"   📜 已启用原稿对照（原稿 {len(original_script.strip())} 字，将注入 detect/verify）")
    elif use_original_script:
        print(f"   ⚠️ 已启用原稿对照，但未提供 original_script，将按无原稿模式运行")

    current_sentences = sentences
    all_decisions: list[dict] = []
    consecutive_empty_rounds = 0   # 连续空轮次计数

    print("\n" + "=" * 50)
    print("  🔄 Agent 循环审查系统")
    print(f"  终止条件: 连续 {consecutive_empty_to_exit} 轮无新问题 / 达到 {max_rounds} 轮上限")
    print("=" * 50 + "\n")

    for round_num in range(1, max_rounds + 1):
        print(f"--- Round {round_num}/{max_rounds} ---\n")

        # === Step 1: Detect Agent ===
        print(f"[Detect] 正在扫描当前文本...")
        new_issues = _run_detect(
            loop_dir=loop_dir,
            round_num=round_num,
            current_sentences=current_sentences,
            model=model,
            effective_script=_effective_script,
            all_decisions=all_decisions,
        )
        if new_issues is None:
            break  # Detect LLM 调用失败

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

        # === Step 1.5: 规则兜底过滤（重复类 issue）===
        # 复用 detect_intra / detect_inter 的机械结果，核对 LLM 的重复类 claim 是否
        # 真的存在机械重复；找不到对应重复的（疑似编造）→ 直接驳回，不进 verify，
        # 避免 LLM/verify 双重失误导致误删。语义正确性仍由 verify 负责。
        if enable_rule_filter and new_issues:
            new_issues, rule_rejected = rule_corroborate(new_issues, current_sentences)
            for iss, reason in rule_rejected:
                v = {"confirmed": False, "reason": f"[规则兜底] {reason}"}
                d = _make_decision_obj(iss, v, round_num, current_sentences)
                all_decisions.append(d)
                sid = iss.get("sentence_idx")
                dim = iss.get("dimension")
                print(f"      🛡️ 规则过滤 句{sid} {dim}: {reason[:60]}")
            if rule_rejected:
                print(f"   🛡️ 规则兜底过滤 {len(rule_rejected)} 个疑似编造的重复（不进 verify）")
            if not new_issues:
                # 检测到的重复全部被规则兜底过滤 → 已处理完毕，不计入空轮、不重复上报
                print(f"   🛡️ 本轮检测到的重复项全部被规则兜底过滤，继续下一轮\n")
                continue

        # === Step 2: Verify Agent ===
        print(f"\n[Verify] 正在验证 {len(new_issues)} 个候选...")
        verified = _run_verify(
            loop_dir=loop_dir,
            round_num=round_num,
            new_issues=new_issues,
            current_sentences=current_sentences,
            model=model,
            effective_script=_effective_script,
            enable_thinking=enable_thinking
        )

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

    # === 保存最终结果（detect + decision 嵌套数组）===
    _save_and_report(output_path, all_decisions)

    return output_path, all_decisions, current_sentences


# ============================================================
#  模式调度 & 向后兼容
# ============================================================

# 向后兼容：pipeline.py / test_detect.py 仍 import run_agent_review_loop，
# 默认指向 v1 行为，避免破坏既有调用。
run_agent_review_loop = run_agent_review_loop_v1

# v2 模式（按类型拆 6 个专职 agent）实现在独立文件，仅在此引用。
from .review_loop_specialized import run_agent_review_loop as run_agent_review_loop_v2


def run_review_loop(
    mode: str = "v1",
    analysis_dir: Path | None = None,
    loop_dir: Path | None = None,
    sentences: list[dict] | None = None,
    words: list[dict] | None = None,
    model: str = "deepseek-v4-pro",
    max_rounds: int = MAX_ROUNDS,
    consecutive_empty_to_exit: int = CONSECUTIVE_EMPTY_TO_EXIT,
    use_original_script: bool = False,
    original_script: str = "",
    enable_thinking: bool = False,
    enable_rule_filter: bool = True,
) -> tuple[Path, list[dict], list[dict]]:
    """调度器：按 mode 选择审查模式。

    Args:
        mode: "v1" = 单检测+单验证（默认，向后兼容）；
              "v2" / "specialized" = 按类型拆 6 个专职 agent（inter/intra/fragment 三分）。
        其余参数与 run_agent_review_loop 完全一致，原样透传。

    Returns:
        同 run_agent_review_loop：(output_path, all_decisions, current_sentences)
    """
    common = dict(
        analysis_dir=analysis_dir,
        loop_dir=loop_dir,
        sentences=sentences,
        words=words,
        model=model,
        max_rounds=max_rounds,
        consecutive_empty_to_exit=consecutive_empty_to_exit,
        use_original_script=use_original_script,
        original_script=original_script,
        enable_thinking=enable_thinking,
        enable_rule_filter=enable_rule_filter,
    )
    if mode in ("v2", "specialized"):
        print(f"   🔀 调度模式: v2（专职 Agent 三分）")
        return run_agent_review_loop_v2(**common)
    print(f"   🔀 调度模式: v1（单检测 + 单验证）")
    return run_agent_review_loop_v1(**common)
