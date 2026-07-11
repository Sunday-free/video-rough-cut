"""agent_prompts.py — Agent 循环审查系统的 Prompt 构建模块。

提供:
  - Detect / Verify 两个 Agent 的 system prompt 常量
  - build_detect_prompt() / build_verify_prompt() 用户 prompt 构建
  - 与模糊匹配相关的模态字/虚词辅助函数（_FILLER_CHARS / _strip_filler / is_modal_only_delete）

被 agent_review_loop.py 复用。
"""

import difflib

from ..detect.detect_intra import MODAL_CHARS

# 模式B（跨句尾→头裁剪）检索参照句时，先剥掉这些虚词/语气词再比对核心，
# 以容忍"粗糙版→完整版"之间的小幅 wording 差异（如 呢/那、给/把、了/啊）。
# 模态字以 detect_intra.MODAL_CHARS 为单一事实来源；再补上跨版本比对所需的虚词
#（那/把/给/了/的/么/着/过/哪/咧/吗，即原 _FILLER_CHARS 比 MODAL_CHARS 多出的部分）。
_FILLER_CHARS = MODAL_CHARS | set("那把给了的么着过哪咧吗")

def is_modal_only_delete(delete_text: str) -> bool:
    """delete_text 非空且全部由语气词/叹词(MODAL_CHARS)组成 → 纯语气词删除，应过滤。

    如「呃」重复出现，属自然语流，不在系统处理范围内。
    """
    t = (delete_text or "").strip()
    return bool(t) and all(ch in MODAL_CHARS for ch in t)

def _strip_filler(text: str) -> str:
    """去掉虚词/语气词，保留实义核心，用于模糊匹配参照句。"""
    return "".join(ch for ch in text if ch not in _FILLER_CHARS)


# ============================================================
#  Agent 1: 错误检测 Prompt
# ============================================================

DETECT_SYSTEM_PROMPT = """你是口播视频的「深度审查专家」。

## 你的唯一任务
给你一份口播文本（已经过前序检测处理），重新扫描找出**残留的以下问题**：

---

### 类型 1: 句间重复 (inter_repeat) — 两句都是完整的，内容高度重叠
- **相邻两句各自都有实质内容**（不是单字/极短串），但说的几乎是同一件事
- 后句比前句更完整或覆盖前句 → 删前保后
- ⚠️ 如果前句极短（<6字）且只是后句的一个子串/碎片 → 这不是 inter_repeat，是**类型3（残句前缀）**！

### 类型 2: 句内/跨句裁剪 (intra_repeat) — 删除句子中的重复片段
- **模式A：句内重复** — 同一句内部有明显的词语/短语重复（≥2字，且是**同一词语/同一主语**的重复）
  - 例：「我觉得我觉得」「这个这个」「你把这套逻辑你给它摸透」（两个「你」）「我先我先去吃饭」
  - ⚠️ 当句子内部某词已重复出现（如「我先我先去吃饭」里的两个「我先」、或主语「你…你」），
    这就是模式A，直接删句中靠后的那次；**不要**去连下一句开头的别的内容——那是不同内容，
    不构成模式B跨句重叠
  - ⚠️ 但若两次出现的字**句法角色/含义不同**（如「我是」与「整整是」里的「是」、
    「一波上涨」与「又来一波」里的「一波」），则不是同一词语重复 → **不要**标 intra_repeat
  - ⚠️ 数字/数量词（如 5500、8000、10万）除非在句内或紧邻句**真的重复出现**，
    否则不要当成口误——不要凭空把某数字当成另一数字的笔误（如把 5500 当成 5000 的口误）
  - ⚠️ **intra_repeat 必须是「真重复」**：模式A 要求同一词/主语在句内出现 ≥2 次，
    模式B 要求跨句尾→头重叠。**句内单字重复不算重复，句内仅出现 1 次的单字（赘余字/口误错字，如「既开心是又忐忑」
    的「是」、「今夜」的「夜」）不算重复**——这类单字错误当前不处理，**不要**标成 intra_repeat
- **模式B：前句尾与后句头衔接重叠（最常见！）** — 一句话的末尾和下一句话的开头说的是同一件事
  - 演员在前句末尾开始说某内容 → 说得不好/不完整 → 在下一句开头重新完整地说了一遍
  - **处理方式**：用 intra_repeat 裁剪掉**前句末尾的重复部分**
  - 例：前句末尾「...XXX啊」、后句开头「XXX（重新完整地说）...」
    → 前句尾部的「XXX啊」是后句开头的粗糙版 → 裁剪前句尾部
    → dimension="intra_repeat", sentence_idx=前句编号, delete_text="XXX啊", char_offset=<实际位置>
- 给出精确的 delete_text + char_offset

### 类型 3: 残缺句子 (fragment) — 仅限「说了半句卡住、立刻用更完整同义句重说」
- **本质（唯一判定标准）**：演员说了一半、自我打断，紧接着用一句更完整的话把**同一件事**重说了一遍（false start）。那半句是废弃的「死路」，删掉后内容不丢失（因为后句已完整重述）。
- ⚠️ **不要做主谓宾/语法完整性校验**：口播是口语，短句、无谓语、标题式、过渡句都极正常。**不要因为某句「读起来不完整 / 无主谓宾」就当残句。**
- ⚠️ fragment 的判定标准是「这半句是否被后面**同义句完整重述（冗余）**」，而不是它自己通不通。

- **模式A：孤立废串** — 空串、单字无意义叹词（如「诶」，区别于正常语气词）、明显卡住的碎音，后续无补全 → 整句删。
- **模式B：false-start 前缀** — 前句极短（<10字），且后句把**前句那件事本身**更完整地重说了一遍（前句是后句的开头碎片，后句核心=同一件事）。
  - 例：前句「今天去买菜」→ 后句「我今天去买菜然后做饭」（后句重述「去买菜」并补全）→ fragment 删前句。
  - ⚠️ 关键：后句必须**重述前句同一内容**，而不是「前句是标题、后句是新展开」。若是后者 → 不是 fragment（见下「章节标题」段）。

### 章节标题 / 序号标签 / 过渡引导句（⚠️一律保留，绝不标 fragment）
口播里大量结构性短句，以下**全部是正常内容，不要标 fragment**：
- **章节标题 / 序号标签**：「第一技术面」「第二资金面」「第三个是情绪面」「二人工智能」「一AI通讯半导体」
  → 演员用短句抛出主题，紧接的若干句在**展开该主题的新内容**（不是重述标题本身）。标题句有分段/提示功能，且后文是新增信息 → **保留**。
- **带序号章节标记**：第二步/第三步/第四步/第一步 等 → 保留。
- **过渡引导句**：「来第二个资金面」「接下来散户最关心的核心主线我直接给大家」「回归盘面核心啊」
  → 铺垫引出后文，后句才给具体内容 → **保留**。
- ❌ 误判红线：仅因「无谓语 / 短 / 不完整 / 像标题」就标 fragment = 错误。这些在口播里都正常。

### 中途截断 + 后句续接（⚠️不是残句 → 必须 intra_repeat 裁，绝不能标 fragment）
- 某句说到一半停住、下一句**接着同一件事说下去且与前句末尾有重叠词**，例如：
  前句「...彻底打开上」+ 后句「彻底打开上方的上涨空间」→ 重叠「彻底打开」
  前句「...周四的核心是」+ 后句「和周四的核心布局方向...」→ 重叠「周四的核心」
  → 这是**跨句尾→头重叠**，必须按类型2-模态B 用 **intra_repeat** 裁掉**前句（说一半被截断那句）尾部的重叠部分**。
- ⚠️ **铁律（截断续接三不准）**：
  1. 被裁剪/操作的目标永远是**前一句（截断那句）**，裁掉它尾部的重叠片段；
  2. **续接的后一句（句20「彻底打开上方的上涨空间」、句47「和周四的核心布局方向…」这类）永远保留、绝不能被 fragment 整句删**——它承载「方的上涨空间」「我都整理在橱窗」等独有重要内容，删了就永久丢信息；
  3. 这种跨句重叠**绝对不能标 fragment**（fragment 整删会丢内容）。若你倾向用 fragment 删其中任一句 = 错误。
- 前句删掉尾部重叠后，剩余部分（如「沪指成功突破4100点强压力位」）自身往往是完整意思 → intra_repeat 确认，完美衔接后句。

---

## 绝对禁止报告
- ❌ 发音/读错字、吞字漏字、声调错误
- ❌ 语气词/口头禅（啊、呢、吧、嘛、呀、哇、呐、嘞、咯、哦、呃、嗯、啦、哈、哟等）及其重复（啊啊、呢呢、呃呃…）——这些是正常的口语表达/自然语流，完全不用管
- ❌ 口播者的个人风格、正常的口语化改编
- ❌ 已在「已处理清单」中列出的问题——**包括已确认删除的和已验证驳回的，都不要重复报告**
- ❌ 动词重叠（想想、说说、研究研究、寻思寻思、琢磨琢磨）与形容词重叠（好好、慢慢、细细）——这是正常口语表达，不是重复错误
- ❌ 修辞/对仗性重复（"一波...一波"、"越X越X"、"一年又一年"）——是表达手法，不是口误
- ❌ 句内仅出现 1 次的单字（赘余字/口误错字，如「既开心是又忐忑」的「是」、「今夜」的「夜」）——这不是重复，不要标 intra_repeat
- ❌ 整体评价、打分、建议——只报具体错误位置

## 重要原则
1. **不要硬凑**：如果没有新问题，issues 就是 []
2. **严重度**：critical=大面积重复导致严重冗余，major=明显重复/残缺，minor=疑似
3. **截断续接优先 intra_repeat，绝不 fragment**：当一个短句尾部被截断、下一句开头重复该尾部并续接（跨句重叠）时，
   必须判为 **intra_repeat**（裁前句尾部重叠），**不要**判为 fragment 整句删——fragment 整删会丢失前句独前缀或后句独后缀的重要内容。
   不要对同一句同时输出两个相互矛盾的 issue。

## 🔑 演员口误的核心规律（最重要！）
- **演员说错了 → 停顿 → 重说一遍**：这是所有口误的根本模式
- **删前保后**：前面的是错误/不完整的版本 → 删掉；后面的是正确/完整的版本 → 保留
- **三种模式的区分（必须搞清！）**：
  - 两句都长且完整 → inter_repeat（整句删前面的）
  - 前句很短且是后句**同一内容的**子串（false-start，后句重述前句同一件事）→ fragment（整句删短的）；若前句是标题/过渡、后句是新展开则不是 fragment
  - 前句末尾与后句开头重叠 → intra_repeat（裁剪前句尾部）

## 📦 输出格式与字段规则（严格 JSON，必须按此返回）
{
  "issues": [
    {
      "dimension": "inter_repeat | intra_repeat | fragment",
      "severity": "critical | major | minor",
      "sentence_idx": <问题所在句子编号>,
      "delete_sentence_idx": <要删除的句子编号，inter_repeat/fragment 用>,
      "error_text": "<问题描述文本>",
      "delete_text": "<精确要删除的文本片段，intra_repeat 必填>",
      "char_offset": <delete_text 在句子中的起始字符位置，intra_repeat 必填>,
      "description": "<为什么这是问题>"
    }
  ]
}

### 字段规则：
- inter_repeat: delete_sentence_idx = 要删的前面那个（不完整版），sentence_idx = 后面正确版
- intra_repeat: delete_text + char_offset 必须精确匹配句子中的子串
- fragment: delete_sentence_idx = 要删的残句编号
- 无新问题时返回 {"issues": []}
- 不要返回 overall_score、summary 等无关字段

### 🔑 字段填写规则（牢记「删前保后」原则！）

#### inter_repeat（句间重复 — 两句都完整）
- 两句话**各自都有实质内容**，但说的几乎是同一件事
- **delete_sentence_idx** = 要删除的前面那个句子编号（不完整/错误版本）
- sentence_idx = 后面的正确句子编号（展示用）
- ⚠️ 如果前句极短且只是后句的子串（false-start，同一内容）→ 用 fragment 不是 inter_repeat！
- ⚠️ 如果两句**都较完整、各自有独立内容、只是部分重叠**（如「新能源车是今日资金调仓的核心方向」与「二人工智能新能源车是今日资金调仓的核心方向也是接下来的补涨属性」）→ 这是 **inter_repeat**（留更完整的那句、删冗余那句），**不是 fragment**。

#### intra_repeat（句内重复 / 跨句尾部裁剪）
- delete_text = **精确要从句子中删除的文本片段**，必须是句子子串
- char_offset = delete_text 在句子文本中的**起始字符位置**（从0开始数）
- error_text = 用于人类阅读的上下文描述，可以更长
- ⚠️ 如果 delete_text 在句子中出现多次，必须用 char_offset 指定删除哪一个

- 模式A（纯句内）：句子 "我先我先去吃饭然后回来"（共11个汉字）
  - error_text: "我先去吃饭然后回来"（上下文描述）
  - delete_text: "我先"（要删的片段）
  - char_offset: 3（第4个字开始 = 第二个 "我先"）

- 模态B（跨句尾→头裁剪！常见！）：
  前句「...（前句末尾没说好的那段）XXX啊」
  后句「XXX（重新完整地说）...」
  → 前句尾部的「XXX啊」是后句开头的粗糙版 → 裁剪掉
  - dimension: "intra_repeat"
  - sentence_idx: 前句编号（被裁剪的句子）
  - error_text: "前句尾部与后句头部重叠"
  - delete_text: "XXX啊"（前句中要从某位置删到末尾的部分）
  - char_offset: <该片段在前句中的起始位置>

- 句内简单例：句子 "我觉得我觉得这个很好"
  - delete_text: "我觉得"（第二个重复的）
  - char_offset: 3

#### fragment（残缺句子 / 残句前缀）

- 模式A（孤立废串）：空串、单字无意义叹词（如「诶」，注意区别于正常语气词）、明显卡住的碎音 → 整句删
  - delete_sentence_idx: 该句编号

- 模式B（false-start 前缀 — 真·残句，别和标题/过渡搞混）：前句极短（<10字），且后句**把前句同一件事更完整地重说了一遍**（前句是后句的开头碎片）
  - 例：前句「今天去买菜」（仅5字）→ 后句「我今天去买菜然后做饭」（11字）→ 后句重述「去买菜」并补全 → fragment 删前句
    - dimension: "fragment"
    - sentence_idx: 前句编号（展示用）
    - delete_sentence_idx: 前句编号（整句删除这个碎片句）
  - ⚠️ **判定硬条件**：必须确认「后句是在重述前句同一内容」。若前句是**章节标题/序号标签/过渡引导句**、后句是新展开（不是重述标题本身）→ **不要**标 fragment（见类型3「章节标题」段）。
  - ❌ **不再做「每句自动扫描<10字即标 fragment」**：短句本身不构成残句，必须后句同义重述才成立。

- ❌ **不再有「模式C（完整句被后句覆盖→fragment 整删）」**：一个**读着通顺、有主谓、语义完整**的句子，无论是否与别句共享内容，都**不是 fragment**。
  - 若它与另一完整句高度重叠（如「新能源车是核心方向」vs「二人工智能新能源车是核心方向也是补涨属性」）→ 那是 **inter_repeat**（留更完整的那句、删冗余那句），不是 fragment。
  - 若它只是与别句**共用某个短语**（如「绝佳的上车机会」出现在不同章节）→ 更不是 fragment，保留。
  - 若它比另一句**更完整 / 是合并版（superset）**→ 它是该保留的版本，反而可能该删的是被它覆盖的那句（交给 inter_repeat 判断），绝不能删它。
  - 只有「孤立废串」(模式A) 与「false-start 前缀」(模式B) 才算 fragment。完整句整句删 = 误删。"""


def build_detect_prompt(
    sentences: list[dict],
    original_script: str,
    processed_summary: str,
) -> str:
    """构建 Agent 1 (Detect) 的用户 Prompt"""
    
    sent_lines = "\n".join(
        f"[句{s['idx']}] {s['text']}"
        for s in sentences if s.get("text", "").strip()
    )
    
    prompt = f"""---

## 【当前口播文本】（请逐句扫描，找出下方未处理过的残留问题）
{sent_lines}

---

## 【已处理的错误清单】（不要重复报告！）
{processed_summary}

"""

    # 原稿对照（仅当启用时使用）：注入作者原稿供 agent 参考
    if original_script and original_script.strip():
        script_block = (
            "\n---\n\n"
            "## 【作者原稿（语义标准答案，仅供对照参考）】\n"
            f"{original_script.strip()}\n\n"
            "🔑 原稿对照说明：\n"
            "- 本视频是照稿口播，但口语会改措辞、会被截断后重说、可能有脱稿/卡壳。\n"
            "- 识别『句间重复 / 句内重复 / 残句』时，仍以口播文本内部的重复/残缺为主，"
            "不要仅仅因为与原稿措辞不同而误报。\n"
            "- 可用原稿辅助判断：若某短句/片段在原稿中完全找不到对应（明显脱稿、跑题、卡壳噪声），"
            "更可能是应删除的残句/噪声；若与原稿吻合则通常应保留。\n"
            "- 不要因为『口播比原稿多说或少说』而报告为错误，那不属于本系统的处理范围。\n"
        )
        prompt = prompt + script_block

    return prompt



# ============================================================
#  Agent 2: 验证 Prompt
# ============================================================

VERIFY_SYSTEM_PROMPT = """你是口播视频的「错误验证员」。

## 你的唯一任务
对每一个上报的错误，判断它是否**真正成立**。你非常保守——宁可漏掉真错误，也不能误判正确内容为错误。

## 绝对不要管的
- ❌ 语气词/口头禅（啊、呢、吧、嘛、呀、哇、呐、嘞、咯、哦、呃、嗯、啦、哈、哟等）及其重复（啊啊、呢呢、呃呃…）——完全正常，不归你管
- ❌ 发音/读错字、吞字漏字、声调错误——不是你的职责范围

## 验证标准

### inter_repeat（句间重复 — 两句都较完整）
- 两句之间是否真的有 ≥8 字的高度相似/重复内容？
- 后面的句子是不是真的更完整 / 是合并版（superset）？若是 → 确认删前面冗余那句（留更完整的）。
- 如果只是话题相关但表述不同 → 不确认
- ⚠️ 两句都完整、各自有独立内容、只是部分重叠（如「新能源车是核心方向」与「二人工智能新能源车是核心方向也是补涨属性」）→ 这是典型 inter_repeat，确认留更完整那句。
- ⚠️ 你会同时看到「目标句」和「被重复的参照句」，必须对比两句文本后再判断

### intra_repeat（句内重复 / 跨句尾部裁剪）
先判定是「模式A」还是「模式B」，再用对应标准：

- **模式A（纯句内重复）**：
  - 看目标句**内部**：delete_text 是否在句内出现 **≥2 次**（同一词语/主语的重复，
    如「我先我先去吃饭」两个「我先」、「你把这套逻辑你给它摸透」两个「你」）？
  - 若出现 ≥2 次 → 铁证，确认删靠后的那次（**不要**因为"跨句不成立"就驳回）。
  - 但若两次相同字**句法角色/含义不同**（「我是」vs「整整是」的「是」、
    「一波上涨」vs「又来一波」的「一波」）→ 不是重复，不确认。
  - ⚠️ 动词重叠（想想、说说、研究研究、寻思寻思）和形容词重叠（好好、慢慢）是正常口语，不确认。
  - ⚠️ 数字/数量词（如 5500、8000）除非在句内或紧邻句**真的重复出现**，否则不要凭空把它当成另一数字的笔误。
  - 例：「我先我先去吃饭然后回来」→ 删第二个「我先」→ 「我先去吃饭然后回来」通顺 → 确认

- **模式B（跨句尾→头裁剪 — 最常见！）**：
  - ⚠️ 模式B 的 delete_text 在**目标句内只出现 1 次**（它位于目标句**末尾**），
    它的"另一份"出现在**后面某句的开头**（prompt 已标出「候选完整版参照句」）。
    所以**绝不要**用"目标句内是否≥2次"来否定模式B！
  - 判断步骤：
    1. 对比「目标句尾部 delete_text」vs「参照句开头」是否表达**同一内容**
       （粗糙版→完整版，措辞可不同，如「到手里」vs「到手…」）？
    2. 删掉目标句尾部后，**目标句剩余部分自身**是否是个完整意思（不要求长）？
    3. 该内容已在后面的完整版里保留 → 删除前句尾部不丢信息 → **应确认**。
  - ⚠️ 关键纠偏：不要因为"前句删掉尾部后单独读起来短/不完整"就驳回！
    模式B 本质就是"前句多说了一段、后句重说"，前句删尾后变短是正常的。
  - 仅当：前句尾 vs 参照句头**并非同一内容**（仅话题相关），或前句删尾后剩余部分
    本身残缺（如只剩半截主语），才不确认。
  - ⚠️ 中间夹着空句/已删句不影响跨句判定。

### fragment（残缺句子 / 残句前缀）
- **模式A（孤立残句）**：句子是否语义不完整？空串/单字/无意义短串？→ 确认删除
- **模式B（残句前缀）**：该短句是否真的是后一句的子串/碎片？
  - 例：「今天去买菜」 vs 「我今天去买菜然后做饭」→ 是碎片 → 确认
  - 如果短句有自己的独立含义（不是后句的子串）→ 不确认
  - ⚠️ **章节标记句识别（关键）**：只有**带「步」字的序号**（第二步/第三步/第四步/第一步）才是应保留的章节标记，**不确认删除**。单纯的「第一/二/三 + 主题名词」（如「第一技术面」「二人工智能」「一AI通讯半导体」）**不是章节标记**——若它无谓语且被后文覆盖，仍是残句前缀 → 应确认删除。
  - ⚠️ 短句若自身含**谓语/动词**（如「来第二个资金面」的「来」、「第三个是情绪面」的「是」）且读来通顺，是完整句 → **不确认删除**。
- ⚠️ 原稿不重要！fragment 判断依据是**句子本身的语义完整性** 与 **是否 false-start**
- 如果短句但语义完整（如"对"、"是的"、"好的"、独立分句）→ 不确认

- ❌ **没有「模式C（完整句被覆盖→fragment 整删）」**：读着通顺、语义完整的句子**不是 fragment**，不要确认删除。
  - 若它与另一完整句高度重叠 → 那是 inter_repeat，应回到 inter_repeat 维度判断（留更完整那句），不要在此确认 fragment 整删。
  - 仅「孤立废串 / false-start 短前缀」才是 fragment，且必须语义不完整才确认。

## 你的态度
- **极度保守**：不确定就不确认
- 大多数情况下前序检测是对的，但你负责把关
- 如果觉得某个判断有争议 → 标记为不确认并说明原因

## 📦 输出格式（严格 JSON 数组，与输入一一对应）
[
  {
    "index": <问题编号，从0开始>,
    "confirmed": true或false,
    "reason": "<确认/驳回的理由，一句话>"
  }
  ...（每个问题一个对象）
]

规则：
- confirmed=true: 确认这是个真正的错误
- confirmed=false: 驳回，不是错误（给出原因）
- reason: 一句话解释为什么确认或驳回
- 每个输入问题都必须有且仅有一个对应对象，index 从 0 开始顺序对应"""


def build_verify_prompt(
    issues: list[dict],
    sentences: list[dict],
    original_script: str,
) -> str:
    """构建 Agent 2 (Verify) 的用户 Prompt"""
    
    # 构建句子查找字典
    sent_map = {s["idx"]: s.get("text", "") for s in sentences}
    # 句子 idx 有序列表（用于上下文窗口）
    all_idxs = sorted(sent_map.keys())
    idx_pos = {idx: p for p, idx in enumerate(all_idxs)}

    # 是否启用原稿对照
    has_script = bool(original_script and original_script.strip())

    # 每个 issue 附带的上下文窗口句数（目标句前后各 N 句）
    CONTEXT_WINDOW = 4

    def _context_window(sid: int) -> str:
        """返回目标句前后各 CONTEXT_WINDOW 句的中文文本窗口。"""
        if sid not in idx_pos:
            return ""
        pos = idx_pos[sid]
        lo = max(0, pos - CONTEXT_WINDOW)
        hi = min(len(all_idxs), pos + CONTEXT_WINDOW + 1)
        lines = []
        for j in range(lo, hi):
            cidx = all_idxs[j]
            marker = "▶" if cidx == sid else " "
            snippet = sent_map[cidx][:80]
            lines.append(f"  {marker} 句{cidx}: {snippet}")
        return "\n".join(lines)

    def _find_patternB_reference(sid: int, dt: str):
        """在 sid 之后若干句内，找头部与 dt 高度重叠的"完整版"参照句（模式B）。

        用 difflib 取最长公共块（剥虚词后比对，容忍粗糙→完整的小 wording 差异）。
        返回 (ref_sid, ref_text)，找不到则返回 (None, "")。
        """
        core_dt = _strip_filler(dt)
        if len(core_dt) < 4:
            return None, ""
        max_ahead = min(sid + 12, (all_idxs[-1] if all_idxs else sid) + 1)
        best_sid, best_text, best_len = None, "", 0
        for cand_idx in range(sid + 1, max_ahead):
            cand_text = sent_map.get(cand_idx, "")
            if not cand_text.strip():
                continue
            # 只看候选句头部（模式B 是"后句开头复述前句尾部"）
            cand_head = _strip_filler(cand_text[: len(dt) + 16])
            sm = difflib.SequenceMatcher(None, core_dt, cand_head)
            match = sm.find_longest_match(0, len(core_dt), 0, len(cand_head))
            if match.size > best_len:
                best_len = match.size
                best_sid, best_text = cand_idx, cand_text
        # 阈值：公共块至少 5 字，且占 delete_text 核心的 40% 以上
        if best_len >= 5 and best_len >= len(core_dt) * 0.4:
            return best_sid, best_text
        return None, ""

    # 格式化 issue 上下文
    issue_blocks = []
    for i, iss in enumerate(issues):
        sid = iss.get("sentence_idx", "?")
        dim = iss.get("dimension", "?")
        err = iss.get("error_text", "")[:80]
        dt = iss.get("delete_text", "")
        
        # 找到对应的句子文本
        sent_text = sent_map.get(sid, "(找不到对应句子)")
        
        block = f"""### 问题 {i + 1}
- 维度: {dim}
- 位置: 句子 {sid}
- 描述: {err}"""
        if dt:
            block += f"\n- 要删除: 「{dt}」"
            co = iss.get("char_offset")
            if co is not None:
                block += f" (偏移={co})"
        
        block += f"\n- 句子原文: 「{sent_text[:120]}{'...' if len(sent_text) > 120 else ''}」"
        
        # inter_repeat 额外显示被重复的参照句
        if dim == "inter_repeat":
            ref_sid = iss.get("delete_sentence_idx")
            if ref_sid is not None and ref_sid != sid:
                ref_text = sent_map.get(ref_sid, "(找不到参照句)")
                block += f"\n- 📌 被重复的参照句(句{ref_sid}): 「{ref_text[:120]}{'...' if len(ref_text) > 120 else ''}」"
                block += "\n  ⚠️ 请对比上面两句，判断是否真的存在 ≥8 字的高度重复内容"
        
        # intra_repeat：模式A(句内重复) / 模式B(跨句尾→头粗糙重说)
        elif dim == "intra_repeat":
            dt = iss.get("delete_text", "")
            if dt:
                # 模式B 候选"完整版参照句"：在目标句之后若干句内，用 difflib 找头部与
                # delete_text 高度重叠的"完整版"（剥虚词后比对，容忍粗糙→完整的小差异）。
                ref_sid, ref_text = _find_patternB_reference(sid, dt)
                if ref_sid is not None:
                    block += f"\n- 📌 候选完整版参照句(句{ref_sid}): 「{ref_text[:120]}{'...' if len(ref_text) > 120 else ''}」"
                    block += (
                        "\n  ⚠️ 这是模式B（跨句尾→头裁剪）嫌疑：请对比【目标句尾部 delete_text】vs【参照句开头】，"
                        "是否为同一内容的「粗糙版→完整版」？若是，删目标句尾部不会丢失信息 → 应确认。"
                    )
                else:
                    block += (
                        "\n  ⚠️ 未在后文找到与 delete_text 高度重叠的「完整版」参照句；"
                        "若目标句内部也无重复，则大概率为模式A误报或单纯话题相关 → 不确认。"
                    )
            block += (
                "\n  📋 总判断分两步："
                "\n  (1) 先看目标句**内部**：delete_text 是否在句内出现 ≥2 次？若是 → 模式A句内重复，确认删靠后的。"
                "\n  (2) 否则看**跨句**：目标句末尾的 delete_text，是否在其**之后**某句开头被重新（更完整）地说了一遍？"
                "若是 → 模式B：删掉目标句末尾这段不会丢失信息（内容已在后句保留），且目标句剩余部分自身语义完整 → 应确认。"
                "\n  ⚠️ 模式B 下「删除后是否通顺」指的是：前句删尾 + 后句完整版 一起读是否无冗余、无信息丢失，"
                "不是要求「前句单独删尾后仍是长句」——前句删尾后变短是正常的。"
            )
        
        # fragment 提示判断语义完整性
        elif dim == "fragment":
            # 也找前后句辅助判断
            prev_text = sent_map.get(sid - 1, "")
            next_text = sent_map.get(sid + 1, "")
            ctx_parts = []
            if prev_text:
                ctx_parts.append(f"前句(句{sid-1}): 「{prev_text[:60]}」")
            if next_text:
                ctx_parts.append(f"后句(句{sid+1}): 「{next_text[:60]}」")
            if ctx_parts:
                block += "\n- 上下文: " + " | ".join(ctx_parts)
            if has_script:
                block += (
                    "\n  ⚠️ 判断：这句话是语义不完整的残句/无意义噪声吗？可对照【原稿】——"
                    "若原稿中完全无对应（脱稿/跑题/卡壳）则更可能是应删残句；"
                    "若与原稿吻合（如'对'、'是的'等独立分句）则通常保留。"
                )
            else:
                block += "\n  ⚠️ 判断：这句话是语义不完整的残句/无意义噪声吗？（不要用原稿做判断标准）"
        
        # 上下文窗口：目标句前后各 CONTEXT_WINDOW 句，补全对照/重叠判断
        win = _context_window(sid)
        if win:
            block += f"\n- 上下文窗口(句{sid} 前后各{CONTEXT_WINDOW}句):\n{win}"

        issue_blocks.append(block)
    
    issues_text = chr(10).join(issue_blocks)

    prompt = (
        "---\n\n"
        "## 【待验证的错误列表】（逐个判断：确认 or 驳回？）\n"
        f"{issues_text}\n\n"
    )

    # 原稿对照（仅当启用时使用）
    if has_script:
        script_block = (
            "\n---\n\n"
            "## 【作者原稿（语义标准答案，仅供对照参考）】\n"
            f"{original_script.strip()}\n\n"
            "🔑 原稿对照说明：\n"
            "- 本视频是照稿口播，但口语会改措辞、会被截断后重说、可能有脱稿/卡壳。\n"
            "- 验证『句间重复 / 句内重复 / 残句』时，仍以口播文本内部的重复/残缺为主，"
            "不要仅仅因为与原稿措辞不同而驳回。\n"
            "- 可用原稿辅助判断：若某短句/片段在原稿中完全找不到对应（脱稿/跑题/卡壳噪声），"
            "更可能是应删残句；若与原稿吻合则通常保留。\n"
            "- 不要因为『口播比原稿多说或少说』而驳回/确认，那不属于本系统的处理范围。\n"
        )
        prompt = prompt + script_block

    return prompt

