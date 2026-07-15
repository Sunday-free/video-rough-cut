"""
llm_judge.py — 步骤4: LLM / Agent 研判层

读取 detect_*.json + 原稿 + 句子列表 → 结构化判断
输出格式（与 assemble.py 兼容）:
  - decisions_inter.json:   [{"finding":"...", "delete_sentences":[19], "llm_reason":"..."}]
  - decisions_intra.json:   [{"sentence":16, "delete_ranges":[], "llm_reason":"..."}]
  - decisions_fragment.json:[{"sentence":11, "mode":"keep_head", "keep_head":[472,492], "llm_reason":"..."}]
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from speech_error_detector.ai.deepseek_client import deepseek_chat
from speech_error_detector.ai.llm_parse import parse_json_object
from speech_error_detector.detect_repeat.judge_prompts import (
    PROMPTS,
)



def _extract_target_indices(detect_results: list[dict], detector_name: str) -> set[int]:
    """从检测结果中提取所有涉及的句子 idx（用于圈定上下文范围）"""
    indices: set[int] = set()
    for item in detect_results:
        for key in (
            "idx_a", "idx_b", "sentence_idx", "sentence",
            "sent_a_idx", "sent_b_idx", "sent_idx", "idx",
        ):
            if key in item and isinstance(item[key], int):
                indices.add(item[key])
    return indices


def build_judge_prompt(
    detect_results: list[dict],
    sentences: list[dict],
    detector_name: str,
    context_radius: int = 2,
) -> str:
    """构建 LLM 研判 Prompt（包含目标句 + 上下文；策略 6 的 finding 自带原文稿窗口）"""
    
    # 建立 idx → sentence 索引
    idx_map = {s["idx"]: s for s in sentences}
    
    # 提取所有待研判句子的 idx
    target_indices = _extract_target_indices(detect_results, detector_name)
    
    # 圈定上下文范围：目标句 ±context_radius
    context_indices: set[int] = set()
    for ti in target_indices:
        for offset in range(-context_radius, context_radius + 1):
            neighbor = ti + offset
            if neighbor in idx_map:
                context_indices.add(neighbor)
    
    # 合并目标句和上下文，按 idx 排序
    all_indices = target_indices | context_indices
    sorted_indices = sorted(all_indices)
    
    # 构建上下文句子列表，用省略号标记跳过的区间
    sent_parts = []
    prev_idx = None
    for idx in sorted_indices:
        s = idx_map[idx]
        # 标记目标句
        marker = " ← 待研判" if idx in target_indices else ""
        sent_parts.append(f"{s['idx']}|{s['text']}{marker}")
        if prev_idx is not None and idx - prev_idx > 1:
            sent_parts.insert(-1, f"...({idx - prev_idx - 1} 句省略)...")
        prev_idx = idx
    
    sent_section = "\n".join(sent_parts)
    
    # === 待研判的检测结果 ===
    detect_block = ""
    for item in detect_results:
        # 逐条独立调用下每条必为单条，序号无意义；统一不加 "### 检测 #i" 头
        for k, v in item.items():
            if k == "org_script_window":
                # 原文稿对照片段
                detect_block += f"- **原文稿对照**: {v}\n"
            elif k in ('text_a', 'text_b', 'text', 'text_c'):
                text_show = v[:80] + ('...' if len(v) > 80 else '')
                detect_block += f"- **{k}**: {text_show}\n"
            else:
                detect_block += f"- **{k}**: {v}\n"
    
    prompt = f"""## 【口播句子列表】（仅展示待研判句及其上下文，共 {len(sorted_indices)} 句，全文共 {len(sentences)} 句）
{sent_section}

---

## 【待研判的 {detector_name} 检测】

{detect_block}

---

请判定上述检测。严格按上方格式输出一个 JSON 对象。"""
    
    return prompt


def run_llm_judge(
    detect_results: list[dict],
    sentences: list[dict],
    output_dir: Path,
    detector_name: str,
    model: str = "deepseek-v4-pro",
    enable_thinking: bool | None = None,
    round_idx: int | None = None,
) -> tuple[Path, list]:
    """
    运行 LLM 研判并保存为合并格式（detect + decision 嵌套，对齐 review_loop_decisions.json）。
    
    输出: judge_decisions_{detector_name}.json
      [{round:0, detect:{...}, decision:{...}, status:"applied"|"rejected"}]
    """
    output_path = output_dir / f"judge_decisions_{detector_name}.json"
    
    if not detect_results:
        print(f"   [llm_judge] {detector_name}: 无检测项，跳过")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f)
        return output_path, []
    
    print(f"   [llm_judge] {detector_name}: 逐条研判 {len(detect_results)} 条检测（独立调用，避免批量一致性偏置）...")

    system_prompt = PROMPTS.get(detector_name, '')
    _round_tag = f"_round{round_idx}" if round_idx is not None else ""

    # ⚠️ 关键修复：逐条独立调用 LLM，而非所有 findings 拼进一个 prompt 批量判定。
    #    证据：同一 finding 单条输入判对(skip)、6 条批量判错(keep_head)——批量"多数在删"的氛围
    #    + decision_hint 权威指令导致一致性偏置，把并列误报条也顺手归删。逐条调用从物理上消除该污染。
    # ⚠️ 并发：线程池 max_workers=4，逐条独立调用 LLM（消除批量一致性偏置），
    #    按 idx 归位保证 all_decisions 顺序与输入 detect_results 一致。
    # 先按序构建全部 prompt（用于合并落盘，且供并发调用复用，避免重复构建）
    round_prompts: list[str] = [
        build_judge_prompt([det], sentences, detector_name) for det in detect_results
    ]
    all_decisions: list[dict] = [{} for _ in range(len(detect_results))]

    def _judge_one(idx: int) -> tuple[int, dict]:
        try:
            response_text = deepseek_chat(
                system=system_prompt,
                user=round_prompts[idx],
                model=model,
                temperature=0.1,
                enable_thinking=enable_thinking,
            )
            dec = parse_json_object(response_text)
            if not isinstance(dec, dict):
                dec = {}
        except Exception as e:
            print(f"      ⚠️ 第 {idx} 条 LLM 调用失败: {e}, 该条记为空判定")
            dec = {}
        return idx, dec

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_judge_one, i) for i in range(len(detect_results))]
        for fut in futures:
            idx, dec = fut.result()
            all_decisions[idx] = dec

    # 落盘：本轮所有逐条 prompt 合并写入一个文件（区分 round），便于整体核查
    prompt_path = output_dir / f"judge_prompt_{detector_name}{_round_tag}.txt"
    try:
        with open(prompt_path, "w", encoding="utf-8") as f:
            for i, p in enumerate(round_prompts):
                if i > 0:
                    f.write("\n\n" + "=" * 50 + f"  检测 #{i}  " + "=" * 50 + "\n\n")
                f.write(p)
        print(f"      📝 本轮 {len(round_prompts)} 条 user prompt 已合并写入: {prompt_path.name}")
    except Exception as e:
        print(f"      ⚠️ user prompt 合并写入失败: {e}")

    # === 合并 detect + decision 为统一格式（对齐 review_loop_decisions.json）===
    merged = _merge_detect_decision(detect_results, all_decisions, detector_name)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    applied_n = sum(1 for m in merged if m.get("status") == "applied")
    rejected_n = len(merged) - applied_n
    print(f"      ✅ 完成: 确认 {applied_n} | 排除 {rejected_n} 误报")
    
    return output_path, merged


def _merge_detect_decision(
    detect_results: list[dict],
    decisions: list[dict],
    detector_name: str,
) -> list[dict]:
    """将 detect 结果和 LLM decision 合并为统一格式"""
    merged = []
    
    for i, det in enumerate(detect_results):
        dec = decisions[i] if i < len(decisions) else {}
        
        # 判断是否确认（根据维度类型不同）
        confirmed = _is_confirmed(dec, detector_name)
        status = "applied" if confirmed else "rejected"
        
        merged.append({
            "round": 0,
            "detect": det,
            "decision": dec,
            "status": status,
        })
    
    return merged


def _is_confirmed(decision: dict, detector_name: str) -> bool:
    """判断 LLM 决策是否为确认删除"""
    if not decision:
        return False
    
    if detector_name == "inter":
        return bool(decision.get("delete_sentences"))
    elif detector_name == "intra":
        return bool(decision.get("delete_ranges"))
    elif detector_name == "fragment":
        mode = decision.get("mode", "")
        return mode not in ("skip", None, "")
    elif detector_name == "partial":
        mode = decision.get("mode", "")
        # partial 仅产出 keep_head / full（部分或整句删除）；skip 视为误报拒绝
        return mode in ("keep_head", "full")
    else:
        return True


# ============================================================
#  批量运行所有检测器的研判
# ============================================================

def run_all_judges(
    analysis_dir: Path,
    sentences: list[dict],
    detect_data: dict[str, list[dict]],
    model: str = "deepseek-v4-pro",
    enable_thinking: bool | None = None,
    round_idx: int | None = None,
) -> dict[str, tuple[Path, list]]:
    """
    对所有三个检测器运行 LLM 研判。
    
    Args:
        analysis_dir:   输出目录
        sentences:      句子列表
        detect_data:    {detector_name: [findings]} 内存数据（优先使用）；
                        为 None 时 fallback 读磁盘 detect_*.json
                      （策略 6 的 finding 已内嵌 org_script_window，无需外部传稿）
    """
    all_results = {}
    
    for det_name in ["inter", "intra", "fragment", "partial"]:
        # 优先使用内存传入的检测数据
        det_results = detect_data[det_name]
        
        if not det_results:
            print(f"   [llm_judge] {det_name}: 无检测项，跳过")
            continue
        
        result = run_llm_judge(
            detect_results=det_results,
            sentences=sentences,
            output_dir=analysis_dir,
            detector_name=det_name,
            model=model,
            enable_thinking=enable_thinking,
            round_idx=round_idx,
        )
        all_results[det_name] = result
    
    return all_results
