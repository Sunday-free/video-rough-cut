"""detect_agent/prompts.py — V3（读稿错误检测）的 Prompt 构建模块。

V3 = 两个 Agent（检测 + 确认），只抓"对照原稿说错"的一类：
  - resay            残句·重说（语义同但措辞异的重说，v1/v2 字面重叠抓不到的盲区）

明确排除：语气词/口头禅、主谓宾/语法、重复类(inter/intra/fragment)、
         误识别（机器同音单字错）、内容读错（数字/专名/事实，misread_content）。
"""

from speech_error_detector.detect_repeat.script_window import get_org_script_window

# ============================================================
#  Agent 1: 读稿错误检测 Prompt
# ============================================================

DETECT_V3_SYSTEM_PROMPT = """你是口播视频的「读稿错误检测专家」。本视频照稿口播，请把转录口播文本逐句对照【作者原稿】，找出演员「说错内容」的地方。

口播是口语，措辞不同没关系，关键是对照原稿找「残句/重说」——同一个意思用不同措辞重说（错/残的半句删前保后）。原稿没有的废话（增读/跑题/脱稿）已停用不再检测。

## 比对前置 · 归一化（先归一化再判）
对「口播句」与「对应原稿」做归一化再比：
- 去标点（，。！？；、："'等）与语气词（啊/呢/吧/嘛/呀/哦/嗯，粤语口语句末语气词也属正常语流）
- 数字统一：中文数字 ↔ 阿拉伯数字（如一百↔100）
- 口语改写视为一致：同一意思的不同口语措辞不算错
- 粤语口播中书面粤语与口语化的同义表达差异、句末语气词变化都属正常语流，不因这类措辞差异误报
- ⚠️ ASR 同音/近音错字不算演员说错：机器听错不归咎演员，不报
归一化后语义一致 → 不报（正确）。

## 步骤：先对齐，再判读错
1) 为每句口播语义对齐到原稿片段（允许一对多/多对一、措辞差异）。
2) 在归一化基础上，只报以下一类"说错"：

### 1. 残句·重说 resay
演员说到一半卡住、或用【不同措辞】把同一件事重说（语义同但措辞异，区别于字面重复）。错/残的半句是死路，删了不丢信息（删前保后）。

⚠️ **删前保后（铁律）**：一对重说里——**更早出现的是「残句」（半截/不完整），更晚出现的是「重说」（完整或措辞更完整）**。`sentence_idx` 必须指向**更早的那句（残句）**，`delete_text` 填该残句应删的部分。永远不要把更晚、更完整的那句当待删——那会丢掉它的新信息。

- **每个 issue 必须同时给出 `delete_text`**：候选句（更早的残句）里应当删除的精确文字。
  · **整句删**：更早那句整句都是半截死路（删前保后不丢信息）→ `delete_text` = 整句口播原文。
  · **片段删**：更早句 = 独有前半段 + 与更晚句重叠的尾部残片 → **只删那段重叠尾部、保留前半段**，`delete_text` = 那段尾部残片的精确子串，**不得整句删**。
    · 例：句N「A（独有前半）B（与后文重叠的尾部）」+ 后文完整重说 B → 只删尾部「B」，保留「A」。
    · 若更早句以衔接词（和/与/及/以及/、）连到后文，且前半段是完整独立意思 → **前半段必留**，只删衔接词起的重叠尾部。例：「小明今天的计划【和周末的安排】」+ 后文「和周末的安排是…」→ 只删「和周末的安排」，保留「小明今天的计划」（整句删会误删前半段独有信息）。
  · `delete_text` 必须是候选句【口播原文】的**精确子串**（一字不差，从句中真实截取），不得编造、不得增删字。后续确认环节会据此校验正确性。
- 🚫 **引子/铺垫一律不报**：引出下文的过渡句（如「接下来…我给大家说清楚」「下面说三点」「话不多说」）是正常语流衔接，不是残句重说，不要报、不要删。

## 绝对禁止报告
- ❌ 语气词/口头禅单字及叠读——正常语流，不报
- ❌ 主谓宾/语法完整性：短句、无谓语、标题式、过渡句都正常，不要因"读着不完整/无主谓宾"就报
- ❌ 重复类（inter_repeat/intra_repeat/fragment）：其他 agent 职责，字面重复/残片不报
- ❌ 误识别（纯机器同音单字错，如单字替换其余严丝合缝）：机器错非演员说错，不报。这也包括 ASR 同音/近音错字——只要上下文与原稿语义一致，就不是演员说错的内容
- ❌ 已在「已处理清单」中列出的问题——不重复报告

## 输出格式（严格 JSON）
{
  "issues": [
    {
      "dimension": "misread",
      "subtype": "resay",
      "severity": "critical|major|minor",
      "sentence_idx": <问题句编号>,
      "error_text": "<问题描述（描述展示/报告）>",
      "delete_text": "<该句应删除的精确文字（整句删填整句原文，片段删填精确子串，须为口播原文精确子串）>"
    }
  ]
}
- 无新问题时返回 {"issues": []}
"""

# ============================================================
#  Agent 2: 读稿错误确认 Prompt（逐候选独立确认，单 JSON 对象）
# ============================================================
# 每次只判【一个】候选、返回单个 JSON 对象；逐条独立调用（见 _run_verify single=True）。
# 目的：消除批量一致性偏置（见 llm_judge.py —— 批量"多数在删"的氛围会把并列
# 候选顺手归删）。逐条独立调用从物理上消除该污染，与 judge 的逐条研判一致。

CONFIRM_V3_SYSTEM_PROMPT = """你是口播视频的「读稿错误删除确认员」。下面只给【一个】候选（由检测环节识别出的"残句/重说"，并附带检测环节给出的 `delete_text` 建议）。你有两件事要做：

1) 判断候选**是不是真错误**（真重说/残句）——检测环节可能误报。
2) 判断检测环节给的 `delete_text`（建议删除的精确文字）**是否正确**。

## 输出规则
- **不是真错误**（纯机器同音识别错、句子语法完整且独立必要、检测明显误报）→ `action="keep"`，`delete_text` 留空。
- **是真错误，且检测给的 `delete_text` 正确**（确实为候选句中应删的精确文字、是口播原文精确子串、无错删/漏删）→ `action="delete"`，`delete_text` **留空字符串 ""**（表示信任检测环节，下游直接用检测的 delete_text）。
- **是真错误，但检测给的 `delete_text` 有误**（错删了、漏删了、或不是口播原文精确子串）→ `action="delete"`，并给出**真正应删的精确文字** `delete_text`。
  ⚠️ 一律以你看到的「口播原文」字段为准，不要照抄检测的删除建议当最终答案；若整句都是冗余、无任何独有信息，delete_text 才填整句口播原文。
- ⚠️ **复核「删前保后」**：若 detect 把【更晚、更完整】的那句标成待删（而非更早的残句），或把本应【片段删】的更早句做了【整句删】（其前半段是独有信息、真值保留）→ 都属检测有误：`action="delete"`，并给出真正该删的（更早句的片段或整句）。
- ⚠️ **引子/铺垫不删（务必对照「对应原稿快照」判定）**：候选若是引出下文的过渡/引导句（如「接下来…我给大家说清楚」「下面说三点」「话不多说」），**不是**残句重说 → `action="keep"`，`delete_text` 留空。
  · **判定铁律**：只有当后文**逐字或语义重说了候选里的某段文字**（删了那段、信息仍由后文承载）时，候选才算真残句/重说。若后文只是**对候选所提主题的独立展开/列举**（如候选提〔主题A〕→ 后文「一、〔子项1〕…」逐条展开，或候选提〔三点〕→ 后文列三点），那是正常引导，**候选整句（含衔接语）在原稿里就是该结构**，一律 keep，不删——这不是"后半段被覆盖"。
  · **看「对应原稿快照」**：若原稿中该候选句本身就出现在"列举/展开"之前（如「接下来…我给大家：一、〔列举项1〕…」），属原稿正常语序 → 100% 是引子，判 keep，不要删其中任何字。

## delete_text 硬性约束
`delete_text` 必须是候选句【口播原文】字段（`口播原文:` 那一行）的**精确子串**（一字不差，从句中真实截取），不得编造、不得增删字。

## 输出格式（严格单个 JSON 对象，不要数组、不要多余文字）
{
  "action": "delete|keep",
  "delete_text": "<真错误且检测正确时留空\"\";真错误但检测有误时填真正应删的精确文字;keep时留空>",
  "is_real_error": true|false,
  "reason": "<一句话说明：是否真错误 + 检测的 delete_text 是否正确 + 最终删了什么>"
}
"""


def build_detect_v3_prompt(
    sentences: list[dict],
    original_script: str,
    rejected_summary: str,
) -> str:
    """构建 V3 检测 Agent 的用户 Prompt（主动对照原稿找"说错"）。"""
    sent_lines = "\n".join(
        f"[句{s['idx']}] {s['text']}"
        for s in sentences if s.get("text", "").strip()
    )
    prompt = f"""---

## 【当前口播文本】（逐句对照下方原稿，找 resay 残句：须是"后文用不同措辞重说候选已有的内容"，且候选含后文没覆盖的独有信息。前句半截+后文补完/续新、引子/铺垫、口误冗余一律不报——见上方「绝对禁止报告」）
{sent_lines}

"""
    if rejected_summary and rejected_summary.strip():
        prompt += f"""---

## 【已驳回的错误清单】（不要重复报告！）
{rejected_summary}

"""
    if original_script and original_script.strip():
        prompt += (
            "\n---\n\n"
            "## 【作者原稿（语义标准答案，V3 主动对照用）】\n"
            f"{original_script.strip()}\n\n"
            "🔑 原稿对照说明（V3 专用）：\n"
            "- 你的任务就是主动对照原稿找「说错」：只报同一件事用不同措辞重说（残句/重说）的情况，内容已被后续重说覆盖、删前保后不丢信息。增读/跑题（原稿无对应的废话/脱稿）已停用不再检测；内容/事实/数字/专名读错也不检测。\n"
            "- 归一化后再比（去标点/语气词、数字统一、口语改写视为一致、粤语书面↔口语差异视为一致）；归一化后一致的不报。\n"
            "- 不因「措辞不同」误报（口语改写正常），但「意思不同」必须报。\n"
        )
    return prompt


def build_confirm_v3_prompt(
    iss: dict,
    sentences: list[dict],
    original_script: str,
    context_radius: int = 3,
) -> str:
    """构建 V3 确认 Agent 的用户 Prompt（一次只判一个候选，消除批量一致性偏置）。

    与 judge 的逐条研判一致：每条候选独立一次 LLM 调用，返回单个 JSON 对象，
    物理上避免"多数在删"的氛围把并列候选顺手归删。

    context_radius: 候选句左右各取 N 个非空可见句作为上下文窗口。
    """
    sid = iss.get("sentence_idx")
    if sid is None:
        # 无法定位候选句，返回最小 prompt
        return "## 待确认候选\n\n无法定位候选句，请跳过。\n\n请输出: {\"confirmed\": false, \"action\": \"report\", \"reason\": \"候选句索引缺失\"}"
    sent_map = {s["idx"]: s.get("text", "") for s in sentences}
    # 候选句不在当前文本中（可能已被前轮删除 / idx 无效）→ 返回安全兜底，确认直接驳回
    if sid not in sent_map:
        return (
            f"## 待确认候选\n\n"
            f"候选句 句{sid} 在当前文本中已不存在（可能已被先前轮次删除，或 idx 无效），无法确认。\n\n"
            f'请输出: {{"confirmed": false, "action": "report", "reason": "候选句 句{sid} 已不存在于当前文本"}}'
        )
    sub = iss.get("subtype", iss.get("dimension"))
    sent_text = sent_map.get(sid, "")
    dt = iss.get("delete_text", "")

    # 原稿聚焦窗口：复用 get_org_script_window（短句自动走 build_short_org_window，内部自适应 max_window）
    orig_win = ""
    if original_script and original_script.strip() and sent_text.strip():
        _iss = {**iss, "sent_idx": sid, "text": sent_text}
        orig_win = get_org_script_window(
            original_script, sentences, _iss, focus_idx=sid,
        ) or ""
    orig_display = orig_win or (original_script.strip() if original_script else "")

    # 构建候选句 ±context_radius 个非空可见句上下文
    min_idx = min(s["idx"] for s in sentences) if sentences else 0
    max_idx = max(s["idx"] for s in sentences) if sentences else sid
    visible_indices: set[int] = {sid}

    # 向左扫描：跳过空句，收集满 context_radius 个非空句
    i = sid - 1
    left_count = 0
    while left_count < context_radius and i >= min_idx:
        if sent_map.get(i):
            visible_indices.add(i)
            left_count += 1
        i -= 1

    # 向右扫描：跳过空句，收集满 context_radius 个非空句
    i = sid + 1
    right_count = 0
    while right_count < context_radius and i <= max_idx:
        if sent_map.get(i):
            visible_indices.add(i)
            right_count += 1
        i += 1

    context_lines: list[str] = []
    for i in sorted(visible_indices):
        marker = " ◀ 候选" if i == sid else ""
        context_lines.append(f"[句{i}] {sent_map.get(i, '(缺失)')}{marker}")
    context_block = "\n".join(context_lines)

    block = (
        "## 待确认的「说错」候选（仅此一个，请独立判断）\n\n"
        f"- subtype: {sub}\n"
        f"- 问题句: 句{sid}\n"
        f"- 口播原文: 「{sent_text}」\n"
        f"- 描述: {iss.get('error_text', '')}\n"
        f"- 对应原稿快照: 「{orig_display}」\n"
    )
    if dt:
        block += (
            f"- 检测环节建议删除（仅供参考，很可能是整句，请你重新判断精确冗余段）: 「{dt}」\n"
            f"  ⚠️ 若候选前半句有后文未覆盖的独有信息，请只保留检测建议里被后文重说的那一段，不要照抄整句。\n"
        )

    block += "\n🔑 对照说明：归一化后再判（去标点/语气词、数字统一、口语改写视为一致、粤语书面↔口语差异视为一致）；归一化后一致的不算错。\n"

    # 上下文窗口（resay 关键：看候选后面是否有完整重说）
    block += (
        f"\n## 【候选句上下文（左右各 {context_radius} 句，请重点看候选后面的句子是否有完整重说）】\n"
        f"{context_block}\n"
    )

    prompt = block + "\n---\n\n请只从该候选句中精确切出应删除的文字（delete_text），严格输出单个 JSON 对象。\n\n"

    return prompt
