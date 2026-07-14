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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ..loop.deepseek_client import deepseek_chat
from ..loop.llm_parse import parse_json_array
from ..base.fillers import MODAL_LIST_ZH, MODAL_LIST_ZH_REPEAT

# ============================================================
#  各检测器的专用 Prompt
# ============================================================

INTER_JUDGE_PROMPT = """你是口播视频质检专家，对「句间重复」检测结果做最终判定。

## 判定标准
- 相邻/隔句开头相同且意思完全一致（后句覆盖/完善前句）→ 确认为口误，删前保后
- 前句的内容被后句的非头部区域大量包含（后句重说了前句的内容，但开头不同）→ 确认为口误，删前保后
- 只是巧合开头相似但内容走向不同 → 误报，不处理

## 输出格式（严格 JSON 数组）
[
  {
    "finding": "<一句话描述该发现>",
    "delete_sentences": [<要删除的句子idx列表，如 [19]>],
    "llm_reason": "<判定理由>"
  }
]

如果某条检测是误报（不应删除），则 delete_sentences 为空数组 []。"""

INTRA_JUDGE_PROMPT = """你是口播视频质检专家，对「句内重复」检测结果做最终判定。

## 判定标准
- 明显的口吃/重说（如"上行上行"、"彻底打开上...彻底打开上"）→ 确认是口误
- 正常的并列结构（"又快又好"、"情绪面/市场情绪"）→ 误报不处理
- ⚠️ 语气词/叹词（""" + MODAL_LIST_ZH + """等）的重复（如""" + MODAL_LIST_ZH_REPEAT + """）→ 属于自然语气流露，不算口误，一律判为误报（delete_ranges 为 []），不要删除

## 输出格式（严格 JSON 数组）
[
  {
    "sentence": <句子idx>,
    "delete_ranges": [[start_word_idx, end_word_idx], ...],  // 要删除的字索引区间；误报则为 []
    "llm_reason": "<判定理由>"
  }
]

每条输入检测对应输出一项。如果确认是口误，delete_ranges 填具体区间；
如果是误报（正常表达），delete_ranges 为 []。"""

FRAGMENT_JUDGE_PROMPT = """你是口播视频质检专家，对「残句」检测结果做最终判定。

## 判定标准
- **孤立编号**（单独一个数字如"三"/"二"）：
  - ⚠️ 先检查上下文：如果全文有其他以中文数字开头的句子（如"一xxx"、"二xxx"、"第三个是..."），且该数字是编号体系的正常序号 → **误报，不删除**
  - 只有确认是真正孤立的、无意义的重复编号才删除 (mode:"full")
  
- 前句卡断被后句接续重说（subtype="残句(被后句接续重说)"）→ 默认保头删尾：
  - 检测结果已给出 head_text（前句独有头部）与 next_sent_text（后句全文）。
  - 只要 head_text **不在** next_sent_text 中（头部未被后句覆盖）→ **必须 mode=keep_head**，
    keep_head 填绝对 word_idx（保留到 head_text 末尾那个词，如 [479, 493]）。
  - 仅当 head_text **完整包含于** next_sent_text（前句信息后句都有）→ 才允许 mode=full 整句删。
  - ⚠️ 头部独有论点（如"沪指成功突破4100点强压力位"）一旦整句删会丢失信息，严禁误判 full。

- 极短但语义完整 → 误报

## 输出格式（严格 JSON 数组）
[
  {
    "sentence": <句子idx>,
    "mode": "<full | keep_head | skip>",   // skip=误报不处理
    "keep_head": [<保留的头部的起止word_idx>, ...],  // 仅 mode=keep_head 时需要
    "llm_reason": "<判定理由，说明为什么这样处理>"
  }
]

每条输入检测对应输出一项。
- mode="skip"：误报（如合法序号），不删除
- mode="full"：整句删除（真正的孤立编号、无独有内容的残句）
- mode="keep_head"：保头删尾（头部有独有论点，只删尾部重复部分）
  - ⚠️ keep_head 必须是 **绝对 word_idx**（即 sentences 中该句子范围的起止索引，如 [472, 491]），
    绝对不能是相对偏移（如 [0, 6] 是错的！）
  - 必须根据句子范围确定正确的绝对索引值"""


PROMPTS = {
    "inter": INTER_JUDGE_PROMPT,
    "intra": INTRA_JUDGE_PROMPT,
    "fragment": FRAGMENT_JUDGE_PROMPT,
}


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
    for i, item in enumerate(detect_results):
        detect_block += f"\n### 检测 #{i}\n"
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

## 【待研判的 {detector_name} 检测】共 {len(detect_results)} 条：

{detect_block}

---

请逐条判定上述每条检测。严格按上方格式输出 JSON 数组。"""
    
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
    
    print(f"   [llm_judge] {detector_name}: 研判 {len(detect_results)} 条检测...")
    
    # 构建 Prompt（目标句+上下文）
    user_prompt = build_judge_prompt(
        detect_results, sentences, detector_name
    )
    system_prompt = PROMPTS.get(detector_name, INTER_JUDGE_PROMPT)

    # 落盘：每次研判的 user prompt 写到 detect 文件夹，便于核查（区分 round）
    _round_tag = f"_round{round_idx}" if round_idx is not None else ""
    prompt_path = output_dir / f"judge_prompt_{detector_name}{_round_tag}.txt"
    try:
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(user_prompt)
        print(f"      📝 user prompt 已写入: {prompt_path.name}")
    except Exception as e:
        print(f"      ⚠️ user prompt 写入失败: {e}")

    # 调用 LLM
    try:
        response_text = deepseek_chat(
            system=system_prompt,
            user=user_prompt,
            model=model,
            temperature=0.1,
            enable_thinking=enable_thinking,
        )
    except Exception as e:
        print(f"      ⚠️ LLM 调用失败: {e}, 写入空结果")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f)
        return output_path, []
    
    # 解析 LLM 判断结果
    parsed = parse_json_array(response_text)
    
    # === 合并 detect + decision 为统一格式（对齐 review_loop_decisions.json）===
    merged = _merge_detect_decision(detect_results, parsed, detector_name)
    
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
    else:
        return True


# ============================================================
#  批量运行所有检测器的研判
# ============================================================

def run_all_judges(
    analysis_dir: Path,
    sentences: list[dict],
    detect_data: dict[str, list[dict]] | None = None,
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
    
    for det_name in ["inter", "intra", "fragment"]:
        # 优先使用内存传入的检测数据
        if detect_data and det_name in detect_data:
            det_results = detect_data[det_name]
        else:
            # Fallback：从磁盘读取已有文件
            det_path = analysis_dir / f"detect_{det_name}.json"
            if not det_path.exists():
                print(f"   [llm_judge] {det_name}: 检测数据不存在，跳过")
                continue
            with open(det_path, "r", encoding="utf-8") as f:
                det_results = json.load(f)
        
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
