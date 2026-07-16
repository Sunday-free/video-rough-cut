"""detect_agent/review_loop.py — V3 读稿错误检测循环（2 Agent：检测 + 确认）。

位于 speech_error_detector/detect_agent/，是 pipeline 唯一使用的循环审查模式。

由原先两个文件合并而来：
  - agent_review_loop.py（v1/v2 通用步骤：detect/verify/apply/落盘）
  - review_loop.py（V3 编排：检测→确认→应用）
v1/v2 已迁移到其他工程，此处只保留 V3 需要的逻辑，去掉 v1/v2 专用分支
（type_filter、批量 verify、机械 seed 补召回、inter/intra/fragment 子类型）。

V3 检测/确认专门对照原稿找"说错"（仅 resay：
  - resay      残句·重说：语义同但措辞异的重说（删前保后）→ 整句删
  - misread 维度：resay 整句删（mode=full，action="delete"）
  - 所有决策统一落 review_loop_decisions.json（含 confirmed + action）

循环：detect_agent(识别错误) → confirm_agent(逐条确认) → apply(应用) → 下一轮
终止：连续 N 轮 detect 返回空（无新错误）
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from speech_error_detector.ai.deepseek_client import deepseek_chat
from speech_error_detector.ai.llm_parse import parse_json_object

# 通用工具（V3 的 detect/apply 依赖），来自同包 agent_apply：
from speech_error_detector.detect_agent.agent_apply import (
    apply_deletions_to_sentences,
    build_rejected_summary,
    is_modal_only_delete,
)
from speech_error_detector.utils.sentence_io import write_sentences

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
#  辅助：纯语气词 过滤
# ============================================================


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
    stage: str = "detect",
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
    rejected_summary = build_rejected_summary(all_decisions)
    detect_prompt = build_detect_v3_prompt(
        sentences=current_sentences,
        original_script=effective_script,
        rejected_summary=rejected_summary,
    )
    _log_prompt(loop_dir, round_num, stage, detect_prompt)

    t0 = time.time()
    detect_response = None
    for attempt in range(1, DETECT_MAX_RETRY + 1):
        try:
            detect_response = deepseek_chat(
                system=DETECT_V3_SYSTEM_PROMPT,
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
    stage: str = "confirm",
) -> list[dict]:
    """执行一轮 Confirm：逐候选独立调用 LLM，物理消除批量一致性偏置。

    与 llm_judge 的逐条研判一致——避免"多数在删"氛围把并列候选顺手归删。
    build_verify_prompt_fn 必须是单候选构造器 (iss, sentences, original_script)，
    如 build_confirm_v3_prompt。

    并发 4 线程执行，LLM 调用失败时返回「不确认」占位，保持与输入一一对应。
    """
    n = len(new_issues)
    sent_map = {s["idx"]: s.get("text", "") for s in current_sentences}

    # 预构建所有 prompt（主线程，安全）；idx 无效/已删的候选直接驳回占位，不送 LLM
    prompt_parts: list[str] = []
    valid_orig_idx: list[int] = []
    prefilled: dict[int, dict] = {}
    for i, iss in enumerate(new_issues):
        sid = iss.get("sentence_idx")
        if sid is None or sid not in sent_map:
            prefilled[i] = {
                "action": "keep",
                "delete_text": "",
                "reason": f"候选句 句{sid} 不在当前文本中（可能已被前轮删除或 idx 无效），跳过确认",
            }
            continue
        prompt_parts.append(
            build_confirm_v3_prompt(iss, current_sentences, effective_script, context_radius=3)
        )
        valid_orig_idx.append(i)
    results: dict[int, dict] = {}
    t0 = time.time()

    def _call_one(orig_i: int, prompt: str) -> tuple[int, dict]:
        try:
            resp = deepseek_chat(
                system=CONFIRM_V3_SYSTEM_PROMPT,
                user=prompt,
                model=model,
                temperature=0.0,
                enable_thinking=enable_thinking,
            )
            dec = parse_json_object(resp)
            if not isinstance(dec, dict):
                dec = {}
        except Exception as e:
            print(f"   ⚠️ Confirm LLM 调用失败 (候选#{orig_i}): {e}")
            dec = {"confirmed": False, "reason": f"LLM 调用失败: {e}"}
        # 规范化为「action + delete_text」模型
        if "action" not in dec:
            # 兼容旧格式：confirmed=false → keep；否则 delete
            dec["action"] = "delete" if dec.get("confirmed") else "keep"
        if "delete_text" not in dec:
            dec["delete_text"] = ""
        if "reason" not in dec:
            dec["reason"] = ""
        return orig_i, dec

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_call_one, orig_i, p): orig_i
            for orig_i, p in zip(valid_orig_idx, prompt_parts)
        }
        for future in as_completed(futures):
            orig_i, dec = future.result()
            results[orig_i] = dec

    # 按原始顺序还原（无效候选用 prefilled 占位）
    verified: list[dict] = []
    for i in range(n):
        if i in prefilled:
            verified.append(prefilled[i])
        else:
            verified.append(results[i])

    # 合并落盘所有单条 prompt（保持可整体核查，对齐 judge 的合并写法）
    merged = "\n\n".join(
        f"{'=' * 50}  候选 #{orig_i}  {'=' * 50}\n\n{p}"
        for orig_i, p in zip(valid_orig_idx, prompt_parts)
    )
    try:
        (loop_dir / f"loop_round{round_num}_{stage}.txt").write_text(
            merged, encoding="utf-8"
        )
    except Exception as e:
        print(f"   ⚠️ confirm prompt 合并写入失败: {e}")
    elapsed = time.time() - t0
    print(f"   Confirm 完成 ({elapsed:.1f}s), 并发4, 逐条 {n} 项")
    return verified


# ============================================================
#  Step 3: 构建决策对象（misread 维度：仅 resay）
# ============================================================

def _make_decision_obj(
    iss: dict,
    v: dict,
    round_num: int,
    current_sentences: list[dict],
) -> dict:
    """根据单个 detect issue 与其 confirm 结果，构建嵌套格式 decision_obj。

    V3 仅处理 misread 维度：resay 整句删（mode=full，action="delete"）。
    目标句即该句 sentence_idx。
    """
    dim = iss.get("dimension", "")
    sub = iss.get("subtype", "")
    delete_sid = iss.get("sentence_idx")
    target_sid = delete_sid if delete_sid is not None else iss.get("sentence_idx")

    # 获取目标句子的 range 和 text
    target_range = "?"
    target_text = ""
    for s in current_sentences:
        if s.get("idx") == target_sid:
            target_range = s.get("range", "?")
            target_text = s.get("text", "")
            break

    action = v.get("action", "delete")
    delete_text = (v.get("delete_text") or "").strip()
    # 由 delete_text 驱动：给了精确待删文字即视为确认删除
    confirmed = (action == "delete" and bool(delete_text))

    return {
        "round": round_num,
        "detect": {
            "type": dim,
            "subtype": sub,
            "sent_idx": target_sid,
            "range": target_range,
            "text": target_text[:80] if len(target_text) > 80 else target_text,
            "decision_hint": iss.get("error_text", ""),
            # 兼容 assemble 的原始字段
            "dimension": dim,
            "severity": iss.get("severity", ""),
            "sentence_idx": iss.get("sentence_idx"),
            "error_text": iss.get("error_text", ""),
            "delete_text": delete_text,
        },
        "decision": {
            "sentence": target_sid,
            "action": action,
            "delete_text": delete_text,
            "confirmed": confirmed,
            "reason": v.get("reason", ""),
            "llm_reason": v.get("reason", ""),
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
) -> tuple[list[dict], list[dict]]:
    """应用本轮确认删除、打印统计、记录 apply 后句子到 loop_round{N}_sentences.txt。

    Returns:
        (应用删除后的 current_sentences, applied_records)
        - current_sentences: idx/range 保持不变
        - applied_records: 与 round_confirmed 平行，含每条实际删除的文字/词索引/模式
    """
    print(f"\n[Apply] 应用 {len(round_confirmed)} 个确认删除...")
    current_sentences, applied_records = apply_deletions_to_sentences(
        sentences=current_sentences,
        words=words,
        confirmed_issues=round_confirmed,
    )

    # 逐条精确打印实际删除内容
    for iss, rec in zip(round_confirmed, applied_records):
        sid = rec.get("sid")
        mode = rec.get("mode")
        dt = rec.get("deleted_text", "")
        if mode == "full":
            tag = "整句"
        elif mode == "partial":
            tag = "片段"
        else:
            tag = "未命中"
        print(f"      🗑 句{sid} [{tag}删除] 实际删除: 「{dt}」")

    total_chars_after = sum(len(s.get("text", "")) for s in current_sentences)
    print(f"   应用完成，当前文本总长: {total_chars_after} 字\n")

    write_sentences(
        loop_dir / f"loop_round{round_num}_sentences.txt",
        current_sentences,
    )

    return current_sentences, applied_records


# ============================================================
#  收尾：保存结果 + 汇总打印
# ============================================================

def _save_decisions_file(output_path: Path, all_decisions: list[dict]) -> None:
    """落盘 review_loop_decisions.json（每轮循环都调用，便于观察中间过程）。"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_decisions, f, ensure_ascii=False, indent=2)


def _save_and_report(output_path: Path, all_decisions: list[dict]) -> None:
    """保存 review_loop_decisions.json 并打印汇总。"""
    _save_decisions_file(output_path, all_decisions)

    applied = sum(1 for d in all_decisions if d.get("decision", {}).get("confirmed"))
    rejected = sum(1 for d in all_decisions if not d.get("decision", {}).get("confirmed"))

    print("=" * 50)
    last_round = max((d.get("round", 0) for d in all_decisions), default=0)
    print(f"  循环结束！共 {last_round} 轮")
    print(f"  总决策: {len(all_decisions)} (✅ 确认 {applied}, ❌ 驳回 {rejected})")
    print(f"  结果文件: {output_path.name}")
    print("=" * 50 + "\n")


def _dedup_issues(issues: list[dict]) -> list[dict]:
    """同一 sentence_idx 重复检出 → 保留第一个。"""
    seen_sids: dict[int, dict] = {}
    deduped: list[dict] = []
    for iss in issues:
        sid = iss.get("sentence_idx")
        if sid is None:
            deduped.append(iss)
            continue
        if sid not in seen_sids:
            seen_sids[sid] = iss
            deduped.append(iss)
    return deduped


def _normalize_for_dedup(text: str) -> str:
    """去标点、去空格、小写 → 用于物理去重的文本归一化。"""
    import re
    t = re.sub(r"[\s，。！？；、：""'…—\-\(\)\[\]【】《》〈〉]", "", text)
    return t.lower()


def _dedup_issues_physical(issues: list[dict], sentences: list[dict]) -> list[dict]:
    """物理去重：不同 sentence_idx 但指向相同归一化句文本 → 合并，保留第一个。"""
    sent_text_map = {s["idx"]: s.get("text", "") for s in sentences}
    seen_texts: dict[str, dict] = {}
    deduped: list[dict] = []
    for iss in issues:
        sid = iss.get("sentence_idx")
        if sid is None:
            deduped.append(iss)
            continue
        raw = sent_text_map.get(sid, "")
        norm = _normalize_for_dedup(raw)
        if not norm:
            deduped.append(iss)
            continue
        if norm not in seen_texts:
            seen_texts[norm] = iss
            deduped.append(iss)
    return deduped


def _dedup_issues_cross_round(
    new_issues: list[dict],
    all_decisions: list[dict],
) -> list[dict]:
    """跨轮去重：过滤 all_decisions 中已决策过的 sentence_idx（不论 confirmed 还是 rejected）。"""
    decided_sids: set[int] = set()
    for d in all_decisions:
        iss = d.get("issue", {})
        sid = iss.get("sentence_idx")
        if sid is not None:
            decided_sids.add(int(sid))

    deduped: list[dict] = []
    for iss in new_issues:
        sid = iss.get("sentence_idx")
        if sid is not None and int(sid) in decided_sids:
            print(f"      ⚠️ 跨轮去重: 句{sid} 已在先前轮次决策过，跳过")
            continue
        deduped.append(iss)

    return deduped


# ============================================================
#  主循环入口：V3 读稿错误检测（检测 + 确认）
# ============================================================

def run_agent_review_loop_v3(
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
            stage="detect_v3",
        )
        if new_issues is None:
            break
        # 安全过滤：只保留 misread 维度（检测越界报的其它类不要）
        new_issues = [iss for iss in new_issues if iss.get("dimension") == "misread"]
        new_issues = _dedup_issues(new_issues)
        new_issues = _dedup_issues_physical(new_issues, current_sentences)
        new_issues = _dedup_issues_cross_round(new_issues, all_decisions)
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
            stage="confirm_v3",
        )

        # === Step 3: 合并 + 分流（删 / 保留）===
        # 新模型：confirm 只输出 delete_text；action=delete 且 delete_text 非空即确认删除，
        # 整句删还是部分删由 apply 根据 delete_text 与整句关系确定性判定（去掉 mode/keep_head）。
        round_confirmed: list[dict] = []
        round_confirmed_objs: list[dict] = []  # 与 round_confirmed 平行，指向 all_decisions 中对应决策对象
        for i, iss in enumerate(new_issues):
            v = verified[i] if i < len(verified) else {
                "index": i, "action": "keep", "delete_text": "", "reason": "无验证结果",
            }
            obj = _make_decision_obj(iss, v, round_num, current_sentences)
            all_decisions.append(obj)
            action = v.get("action", "delete")
            delete_text = (v.get("delete_text") or "").strip()
            if action == "delete" and delete_text:
                aug = dict(iss)
                aug["delete_text"] = delete_text  # 以 confirm 给出的精确待删文字为准
                round_confirmed.append(aug)
                round_confirmed_objs.append(obj)
                print(f"      ✅ #{i} 删除: {v.get('reason', '')[:60]} ｜待删='{delete_text}'")
            else:
                print(f"      ❌ #{i} 保留: {v.get('reason', '')[:60]}")

        # 每轮合并完决策后立即落盘，便于观察中间过程
        _save_decisions_file(output_path, all_decisions)

        if not round_confirmed:
            print(f"\n   本轮无确认删除，继续下一轮\n")
            continue

        current_sentences, applied_records = _apply_and_log(
            loop_dir=loop_dir,
            round_num=round_num,
            current_sentences=current_sentences,
            words=words,
            round_confirmed=round_confirmed,
        )

        # 把实际删除文字写回决策对象（与 round_confirmed / round_confirmed_objs 平行对齐）
        for obj, rec in zip(round_confirmed_objs, applied_records):
            obj.setdefault("decision", {})["applied_text"] = rec.get("deleted_text", "")
        _save_decisions_file(output_path, all_decisions)

    # === 保存决策（删除 / 仅标注 / 驳回 全部合并为单文件，对齐 福总 格式）===
    _save_and_report(output_path, all_decisions)
    report_n = sum(
        1 for d in all_decisions if d.get("decision", {}).get("action") == "report"
    )
    if report_n:
        print(f"   📌 仅标注项（已并入 {output_path.name}）: {report_n} 项")

    return output_path, all_decisions, current_sentences
