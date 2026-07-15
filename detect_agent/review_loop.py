"""detect_agent/review_loop.py — V3 读稿错误检测循环（2 Agent：检测 + 确认）。

位于 speech_error_detector/detect_agent/，是 pipeline 唯一使用的循环审查模式。

由原先两个文件合并而来：
  - agent_review_loop.py（v1/v2 通用步骤：detect/verify/apply/落盘）
  - review_loop.py（V3 编排：检测→确认→应用）
v1/v2 已迁移到其他工程，此处只保留 V3 需要的逻辑，去掉 v1/v2 专用分支
（type_filter、批量 verify、机械 seed 补召回、inter/intra/fragment 子类型）。

V3 检测/确认专门对照原稿找"说错"（off_topic / resay，不含 misread_content）：
  - off_topic  增读/跑题：原稿无对应、非过渡的废话/脱稿/长串口头禅 → 整段删
  - resay      残句·重说：语义同但措辞异的重说（删前保后）→ 整句删
  - misread 维度：off_topic / resay 均整句删（mode=full，action="delete"）
  - 所有决策统一落 review_loop_decisions.json（含 confirmed + action）

循环：detect_agent(识别错误) → confirm_agent(逐条确认) → apply(应用) → 下一轮
终止：连续 N 轮 detect 返回空（无新错误）
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from speech_error_detector.ai.deepseek_client import deepseek_chat
from speech_error_detector.ai.llm_parse import parse_json_object

# 通用工具（V3 的 detect/apply 依赖），来自同包 agent_apply：
from speech_error_detector.detect_agent.agent_apply import (
    apply_deletions_to_sentences,
    build_processed_summary,
    is_modal_only_delete,
)

from speech_error_detector.detect_agent.prompts import (
    DETECT_V3_SYSTEM_PROMPT,
    CONFIRM_V3_SYSTEM_PROMPT,
    build_detect_v3_prompt,
    build_confirm_v3_prompt,
)

MAX_ROUNDS = 5          # 最大循环轮次，防止无限循环
CONSECUTIVE_EMPTY_TO_EXIT = 2  # 连续N轮无新问题才退出
DETECT_MAX_RETRY = 3     # 单轮 Detect LLM 硬失败（抛异常）时的重试次数
DETECT_RETRY_BACKOFF = 2.0  # 重试退避基数（秒），第 n 次重试等待 n * backoff


# ============================================================
#  辅助：prompt / 句子 日志落盘
# ============================================================

def _log_prompt(loop_dir: Path, round_num: int, stage: str, prompt_text: str) -> None:
    """记录本轮 detect/confirm prompt 到 loop_round{N}_{stage}.txt。"""
    path = loop_dir / f"loop_round{round_num}_{stage}.txt"
    with open(path, "w", encoding="utf-8") as pf:
        pf.write(f"===== Round {round_num} | {stage.upper()} PROMPT =====\n\n")
        pf.write(prompt_text)
        pf.write("\n\n")


# ============================================================
#  辅助：纯语气词 / 单字重复 / 自检撤回 过滤
# ============================================================

def _is_single_char_repeat(iss: dict) -> bool:
    """单字重复判定：重复类 issue 且其删除文本仅 1 个字符（如「是」单字赘余）。

    V3 维度为 misread（off_topic/resay），本判定对其恒为 False；保留以兼容 detect 残留。
    """
    dim = iss.get("dimension", "")
    if dim not in ("intra_repeat", "inter_repeat"):
        return False
    dt = (iss.get("delete_text") or "").strip()
    return len(dt) == 1


def _is_self_retract(iss: dict) -> bool:
    """自检撤回判定：detect 输出的 description 为「不报」等自我否定短语时，
    说明模型自己认为该候选不该报。进 verify 也只是被驳回，纯属浪费 LLM 调用并污染
    决策日志，因此在 detect 出口直接丢弃，不进 verify、不进 decisions 日志。
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
    system_prompt: str = DETECT_V3_SYSTEM_PROMPT,
    build_prompt_fn=build_detect_v3_prompt,
    stage: str = "detect",
    log_prompt: bool = True,
) -> list[dict] | None:
    """执行一轮 Detect：构建 prompt、记录日志、调用 LLM、解析、过滤纯语气词删除项。

    Args:
        build_prompt_fn: 注入的 prompt 构造器（V3 传 build_detect_v3_prompt）。
        stage: 日志文件名片段。
        log_prompt: 是否落盘 detect prompt。

    Returns:
        - 成功: 过滤后的 new_issues（list[dict]）
        - LLM 调用失败: None（调用方据此 break 退出循环）
    """
    processed_summary = build_processed_summary(all_decisions)
    detect_prompt = build_prompt_fn(
        sentences=current_sentences,
        original_script=effective_script,
        processed_summary=processed_summary,
    )
    if log_prompt:
        _log_prompt(loop_dir, round_num, stage, detect_prompt)

    t0 = time.time()
    detect_response = None
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
    # 不进 verify，避免浪费 LLM 调用并污染决策日志。
    before = len(new_issues)
    new_issues = [
        iss for iss in new_issues
        if not _is_single_char_repeat(iss)
    ]
    if len(new_issues) != before:
        print(f"   ⏩ 过滤掉 {before - len(new_issues)} 个单字重复项（delete_text 仅 1 字，不处理）")

    # 过滤：自检撤回（detect 自己标「不报」的候选）直接丢弃，不进 verify。
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
#  Step 2: Confirm Agent（逐候选独立调用）
# ============================================================

def _run_verify(
    loop_dir: Path,
    round_num: int,
    new_issues: list[dict],
    current_sentences: list[dict],
    model: str,
    effective_script: str,
    enable_thinking: bool = False,
    system_prompt: str = CONFIRM_V3_SYSTEM_PROMPT,
    stage: str = "confirm",
    build_verify_prompt_fn=build_confirm_v3_prompt,
    single: bool = True,
) -> list[dict]:
    """执行一轮 Confirm：逐候选独立调用 LLM，物理消除批量一致性偏置。

    与 llm_judge 的逐条研判一致——避免"多数在删"氛围把并列候选顺手归删。
    build_verify_prompt_fn 必须是单候选构造器 (iss, sentences, original_script)，
    如 build_confirm_v3_prompt。

    LLM 调用失败时返回全部"不确认"占位，保持与输入一一对应。
    """
    # V3 仅支持逐条模式（v1/v2 批量模式已移走）
    verified: list[dict] = []
    prompt_parts: list[str] = []
    t0 = time.time()
    for iss in new_issues:
        prompt = build_verify_prompt_fn(iss, current_sentences, effective_script)
        prompt_parts.append(prompt)
        try:
            resp = deepseek_chat(
                system=system_prompt,
                user=prompt,
                model=model,
                temperature=0.0,
                enable_thinking=enable_thinking,
            )
            dec = parse_json_object(resp)
            if not isinstance(dec, dict):
                dec = {}
        except Exception as e:
            print(f"   ⚠️ Confirm LLM 调用失败: {e}")
            dec = {"confirmed": False, "reason": f"LLM 调用失败: {e}"}
        if "confirmed" not in dec:
            dec["confirmed"] = False
        if "action" not in dec:
            dec["action"] = "delete"
        verified.append(dec)
    # 合并落盘所有单条 prompt（保持可整体核查，对齐 judge 的合并写法）
    merged = "\n\n".join(
        f"{'=' * 50}  候选 #{i}  {'=' * 50}\n\n{p}"
        for i, p in enumerate(prompt_parts)
    )
    try:
        (loop_dir / f"loop_round{round_num}_{stage}.txt").write_text(
            merged, encoding="utf-8"
        )
    except Exception as e:
        print(f"   ⚠️ confirm prompt 合并写入失败: {e}")
    print(f"   Confirm 完成 ({time.time() - t0:.1f}s), 逐条 {len(verified)} 项")
    return verified


# ============================================================
#  Step 3: 构建决策对象（misread 维度：off_topic / resay）
# ============================================================

def _make_decision_obj(
    iss: dict,
    v: dict,
    round_num: int,
    current_sentences: list[dict],
) -> dict:
    """根据单个 detect issue 与其 confirm 结果，构建嵌套格式 decision_obj。

    V3 仅处理 misread 维度：off_topic / resay 均整句删（mode=full，action="delete"）。
    目标句取 delete_sentence_idx（off_topic/resay 通常与 sentence_idx 相同）。
    """
    dim = iss.get("dimension", "")
    sub = iss.get("subtype", "")
    delete_sid = iss.get("delete_sentence_idx", iss.get("sentence_idx"))
    target_sid = delete_sid if delete_sid is not None else iss.get("sentence_idx")

    # 获取目标句子的 range 和 text
    target_range = "?"
    target_text = ""
    for s in current_sentences:
        if s.get("idx") == target_sid:
            target_range = s.get("range", "?")
            target_text = s.get("text", "")
            break

    confirmed = v.get("confirmed", False)
    action = v.get("action", "delete")
    # 整段删除（off_topic/resay 确认且 action==delete）→ mode="full"；其余无 mode
    mode = "full" if (confirmed and action == "delete") else None

    return {
        "round": round_num,
        "detect": {
            "type": dim,
            "subtype": sub,
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
            "action": action,
            "llm_reason": v.get("reason", ""),
            "confirmed": confirmed,
            "reason": v.get("reason", ""),
        },
        "action": action,
    }


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
#  主循环入口：V3 读稿错误检测（检测 + 确认）
# ============================================================

def run_agent_review_loop_v3(
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
) -> tuple[Path, list[dict], list[dict]]:
    """运行 V3 读稿错误检测循环（检测→确认→应用）。

    Returns: (output_path, all_decisions, current_sentences)
    """
    output_path = loop_dir / "review_loop_decisions.json"

    _effective_script = original_script if use_original_script else ""
    if use_original_script and original_script and original_script.strip():
        print(f"   📜 V3 已启用原稿对照（原稿 {len(original_script.strip())} 字）")

    current_sentences = sentences
    all_decisions: list[dict] = []
    consecutive_empty_rounds = 0

    print("\n" + "=" * 50)
    print("  🔄 V3 读稿错误检测循环（检测 + 确认）")
    print("=" * 50 + "\n")

    for round_num in range(1, max_rounds + 1):
        print(f"--- V3 Round {round_num}/{max_rounds} ---\n")

        # === Step 1: Detect（用 V3 prompt 构造器 + V3 system prompt）===
        print(f"[Detect V3] 正在对照原稿扫描...")
        new_issues = _run_detect(
            loop_dir=loop_dir,
            round_num=round_num,
            current_sentences=current_sentences,
            model=model,
            effective_script=_effective_script,
            all_decisions=all_decisions,
            enable_thinking=enable_thinking,
            system_prompt=DETECT_V3_SYSTEM_PROMPT,
            stage="detect_v3",
            build_prompt_fn=build_detect_v3_prompt,
        )
        if new_issues is None:
            break
        # 安全过滤：只保留 misread 维度（检测越界报的其它类不要）
        new_issues = [iss for iss in new_issues if iss.get("dimension") == "misread"]
        if not new_issues:
            consecutive_empty_rounds += 1
            print(f"   ✅ 无新错误 ({consecutive_empty_rounds}/{consecutive_empty_to_exit} 连续空轮)")
            if consecutive_empty_rounds >= consecutive_empty_to_exit:
                print(f"\n   🏁 连续 {consecutive_empty_to_exit} 轮无新问题，循环结束\n")
                break
            continue
        consecutive_empty_rounds = 0

        for i, iss in enumerate(new_issues):
            sev = iss.get("severity", "?")
            sub = iss.get("subtype", iss.get("dimension"))
            sid = iss.get("sentence_idx", "?")
            print(f"      [{sev}] #{i} {sub}: 句{sid}「{(iss.get('error_text') or '')[:40]}」")

        # === Step 2: Confirm（逐候选独立调用，与 judge 一致，消除批量一致性偏置）===
        print(f"\n[Confirm V3] 正在逐条确认 {len(new_issues)} 个候选...")
        verified = _run_verify(
            loop_dir=loop_dir,
            round_num=round_num,
            new_issues=new_issues,
            current_sentences=current_sentences,
            model=model,
            effective_script=_effective_script,
            enable_thinking=enable_thinking,
            system_prompt=CONFIRM_V3_SYSTEM_PROMPT,
            stage="confirm_v3",
            build_verify_prompt_fn=build_confirm_v3_prompt,
            single=True,
        )

        # === Step 3: 合并 + 分流（删 / 仅标注）===
        round_confirmed: list[dict] = []
        for i, iss in enumerate(new_issues):
            v = verified[i] if i < len(verified) else {
                "index": i, "confirmed": False, "action": "delete", "reason": "无验证结果",
            }
            obj = _make_decision_obj(iss, v, round_num, current_sentences)
            all_decisions.append(obj)
            if v.get("confirmed"):
                action = v.get("action", "delete")
                if action == "delete":
                    round_confirmed.append(iss)
                    print(f"      ✅ #{i} 删除: {v.get('reason', '')[:60]}")
                else:  # report：仅标注，不删（仍随 all_decisions 落单文件）
                    print(f"      📌 #{i} 仅标注(report): {v.get('reason', '')[:60]}")
            else:
                print(f"      ❌ #{i} 驳回: {v.get('reason', '')[:60]}")

        if not round_confirmed:
            print(f"\n   本轮无确认删除，继续下一轮\n")
            continue

        current_sentences = _apply_and_log(
            loop_dir=loop_dir,
            round_num=round_num,
            current_sentences=current_sentences,
            words=words,
            round_confirmed=round_confirmed,
        )

    # === 保存决策（删除 / 仅标注 / 驳回 全部合并为单文件，对齐 福总 格式）===
    _save_and_report(output_path, all_decisions)
    report_n = sum(
        1 for d in all_decisions if d.get("decision", {}).get("action") == "report"
    )
    if report_n:
        print(f"   📌 仅标注项（已并入 {output_path.name}）: {report_n} 项")

    return output_path, all_decisions, current_sentences
