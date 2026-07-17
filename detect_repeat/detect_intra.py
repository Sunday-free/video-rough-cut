"""
detect_intra.py — 句内重复机械检测

检测策略: N-gram 子串在短距离（<=3字符）内重复出现

输出: detect_intra.json
"""

from pathlib import Path

from speech_error_detector.utils.fillers import MODAL_CHARS
from speech_error_detector.config import INTRA_MAX_GAP_DIGIT, INTRA_MAX_GAP_OTHER

# 自然并列重复的连接性虚词：两个重复单元之间仅隔这些字（1~3 个）时，
# 视为汉语正常的并列强调（如"一波还有一波""一个又一个""一遍又一遍"），不算口误。
# 注意：只收真正"并列/连接"义虚词（还/又/再/来/也/的/就/而/则/且/跟/与/和/等），
#      刻意排除"啊呀哦嗯呐嘞咯呃哇嘢哟哈"等犹疑 filler —— 它们是口吃/停顿而非并列，
#      若纳入会把"我们啊我们"这类带 filler 的重说误判为自然并列而漏掉，且与"今天呢今天呢"
#      （"呢"不在集合而被标记）造成同构不同判的不一致。带 filler 的重复应交由上层判定。
NATURAL_CONNECTOR_CHARS = set(
    "还又再来也的就而则且跟与和等"
)

# 常见双字连接词短语：间隔为这些短语时同样视为自然并列重复。
NATURAL_CONNECTOR_PHRASES = {
    "还有", "而后", "然后", "之后", "接着", "跟着", "以及", "再来",
    "又来", "而又", "再又", "又再",
}

# ---- 紧邻重复（gap 为空）的「合法叠词」判定（仅这些才在紧邻时跳过）----
# 口播里紧邻整词双写大多是口误重说（"我说我说""我们我们""这个这个"），应保留送研判；
# 只有下列真正合法的叠词形式才视作自然叠词跳过，以提升召回、避免永久漏检。
# 1) 数字+量词 的"逐一"强调，且只用"一/两"（如"一个一个""一波一波""两个两个"）；
#    不用其他数字，避免把"3个3个"这类重说也误当强调跳过。
_REDUP_EMPHASIS_NUM = set("一两")
_REDUP_CLASSIFIERS = set(
    "个只头匹辆本支枝双对颗棵朵片位种台部根把张件间所家口层页"
    "艘架盏尾群套份杯碗盘粒杆顶扇面节首句波遍回次下阵番场名"
    "位条声步年天月日时周季分秒户门"
)
# 2) 动词/形容词 AA 式，及其 AABB 式的 AB 段（"研究研究""高高兴兴""明明白白"）
_REDUP_WORDS = {
    # 动词 AA
    "研究", "考虑", "商量", "讨论", "学习", "休息", "锻炼", "调查", "打听", "打扮",
    "收拾", "打扫", "整理", "检查", "测试", "练习", "准备", "安排", "交流", "沟通",
    "活动", "运动", "体验", "感受", "认识", "了解", "尝试", "思考", "回忆", "观察",
    "比较", "分析", "总结", "说明", "表示", "表现", "处理", "解决", "适应", "接受",
    # 动词 AA（补充：口播中高频 ABAB 重叠，如 琢磨琢磨/溜达溜达）
    "琢磨", "合计", "盘算", "掂量", "斟酌", "品尝", "打量", "端详", "溜达", "放松",
    "复习", "修改", "调整", "规划", "布置", "核对", "确认", "咨询", "请教", "揣摩",
    "回味", "回顾", "熟悉", "消化", "比划", "寻思",
    # 形容词（同时覆盖 AABB：高高兴兴/快快乐乐/明明白白/干干净净/清清楚楚…）
    "高兴", "快乐", "明白", "干净", "清楚", "安静", "热闹", "辛苦", "努力", "认真",
    "漂亮", "开心", "健康", "整齐", "悠闲", "马虎", "模糊", "简单", "复杂", "兴奋",
    # 形容词 ABAB 重叠形（如 舒服舒服/轻松轻松/暖和暖和）
    "舒服", "暖和", "凉快", "轻松", "痛快", "自在", "宽松", "利索",
}

# 阿拉伯数字 + 中文基数词（数字判定用集合）。
# 注意：十/百/千/万/廿/卅 不在此集合——它们在"数字+单位"里常作量词单位
# （5万、3千、2百）而非数位；若被当数字，会导致 _expand_full 把后续数字链也吞入，
# 使两个不同枚举项被误判为同一数字而漏跳。
_DIGIT_CHARS = set("0123456789零一二三四五六七八九两廿卅")
# 兼容别名（枚举检测用，不含十百千万）
_NUMERALS = _DIGIT_CHARS

# 量词内部连接符：并列枚举的共用前缀后常跟这些（如"百分之三"的"之"），
# 判断"前缀后是否为不同数字"时需跳过它们再取首个内容字符。
_MEASURE_CONNECTORS = set("之的个")

# 中文物量词（用于"数字+量词+名词"并列枚举，如"3个苹果4个苹果"的"苹果"）。
# 仅收录"名词性物量词"，刻意排除"块/名/倍/岁/日/月/年/号"等易歧义或已由单位
# 逻辑处理的词，避免误吞真实重复（如"8块你都8块你都"）。
_MEASURE_WORDS = set(
    "个只条头匹辆本支枝双对颗棵朵片位种台部根把张件间所家口层页"
    "艘架盏尾群套份杯碗盘粒杆顶扇面节首句"
)


def _post_content_char(txt: str, idx: int) -> str:
    """从 idx 起跳过量词内部连接符（之/的/个），返回首个内容字符（越界返回空串）。"""
    j = idx
    while j < len(txt) and txt[j] in _MEASURE_CONNECTORS:
        j += 1
    return txt[j] if j < len(txt) else ""


def _num_prefix_len(s: str) -> int:
    """返回 s 开头连续数字/数位的个数（如 "12234日" → 5）。"""
    n = 0
    while n < len(s) and s[n] in _DIGIT_CHARS:
        n += 1
    return n


# 向右扩展时作为"单位"吞入的字符判定：非数字、非标点空白的字符（日/月/岁/块/蚊/号…）
_PUNCT = set("，。、；：！？,.!?;: \n\t（）()“”\"'《》<>…—")


def _is_unit_char(c: str) -> bool:
    """单位字符判定：非数字、非标点空白（日/月/岁/块/蚊/号…）。"""
    return c not in _DIGIT_CHARS and c not in _PUNCT


def _is_digit_char(c: str) -> bool:
    """数字段字符：阿拉伯/中文数字，或小数点（允许 3.5 这类小数作为整体数字段）。"""
    return c in _DIGIT_CHARS or c == "."


def _expand_full(txt: str, start: int, end: int):
    """若 [start,end) 落在某个"数字(+单位)"片段内部，则把区间扩展到完整
    "数字+单位"，保证「任何数字重叠都完整匹配到数字」：
      - 先向左吞掉连续前置数字/小数点（如 "234日"→"12234日"、"5亿"→"3.5亿"、
        "公斤"→"0.5公斤"），把紧邻左侧的数字前缀也并入；
      - 若区间已含单位字符（如 "4岁""234日""5亿""公斤"），数字+单位已完整，不再向右扩；
      - 若区间仅含数字（纯数字片段如 "23"），向右吞同数字剩余数字/小数点，再吞一个
        单位字符（"23"→"234日"），避免单位落在更右侧而漏判。
    否则（纯中文名词如 "苹果"，紧邻左侧也非数字）原样返回，由量词分支另行处理。
    """
    s = start
    while s > 0 and _is_digit_char(txt[s - 1]):
        s -= 1
    e = end
    contains_digit = any(c in _DIGIT_CHARS for c in txt[s:e])
    contains_unit = any(_is_unit_char(c) for c in txt[s:e])
    if contains_digit and not contains_unit:
        # 纯数字片段：向右吞同数字的剩余数字/小数点，再吞一个单位字符（如 "23"→"234日"）
        while e < len(txt) and _is_digit_char(txt[e]):
            e += 1
        if e < len(txt) and _is_unit_char(txt[e]):
            e += 1
    return s, e


def _num_before_measure(txt: str, measure_pos: int) -> str:
    """measure_pos 指向量词字符；返回其紧邻左侧连续数字段（前列举的数字），无则空串。"""
    e = measure_pos
    s = e
    while s > 0 and _is_digit_char(txt[s - 1]):
        s -= 1
    return txt[s:e]


def _is_parallel_enumeration(txt: str, sub: str, pos1: int, pos2: int) -> bool:
    """判断该重复是否来自"数字+单位"并列枚举（共用词缀，非口误）。

    特征：重复子串是并列项的共用前缀或后缀，且两个出现处"另一侧"紧邻的
    内容字符都是数字/数字段（阿拉伯数字或中文数字，允许含小数点），且两者不同
    （被列举的是不同项）。

    例：
      - 5日10日20日：sub="0日" 为后缀，前文分别是 "1""2"（数字）→ 枚举
      - 百分之三百分之四：sub="百分之"/"百分" 为前缀，跳过"之"后分别是 "三""四" → 枚举
      - 七蚊九七蚊八：sub="七蚊" 为前缀，后文分别是 "九""八"（数字）→ 枚举
      - 12日13日 / 34岁35岁：sub 仅匹配到尾部片段，相邻首位数字恰相同会被旧逻辑
        误判；本函数向左补全为完整数字再比，数字不同→枚举
      - 12234日 234日：sub 只匹配到尾部 "234日"，向左补全为 "12234日"vs"234日"，
        数字不同且单位相同→枚举
      - 300万400万 / 3090万3095万：若把"万"当数字会误吞后续；本函数把"万/千/百"
        视为量词单位，sub="0万"向左补全为 "300万"vs"400万"、"3090万"vs"3095万"，
        数字不同且单位"万"相同→枚举
      - 3.5亿4.5亿 / 3.5%4.5%：sub 落在小数尾部（"5亿"/"5%"），向左跨小数点补全为
        "3.5亿"vs"4.5亿"、"3.5%"vs"4.5%"，数字不同且单位相同→枚举
      - 3个苹果4个苹果：sub="苹果" 前一字符是量词"个"→共用量词的并列列举→枚举
    反之：「百分之三百分之三」「8块你都8块你都」（后文相同）、「34岁34岁」
    （完整数字相同，属 false-start 重说）、以及「3个苹果3个苹果」（数字相同）
    不会被误判为枚举，仍按真实重复处理。
    """
    length = len(sub)
    # 前缀型：看两个出现处"之后"的内容字符（跳过量词内部连接符如"之"）
    post1 = _post_content_char(txt, pos1 + length)
    post2 = _post_content_char(txt, pos2 + length)
    if post1 and post2 and post1 in _DIGIT_CHARS and post2 in _DIGIT_CHARS and post1 != post2:
        return True
    # 后缀型：看两个出现处"之前"的字符
    pre1 = txt[pos1 - 1: pos1] if pos1 > 0 else ""
    pre2 = txt[pos2 - 1: pos2] if pos2 > 0 else ""
    if pre1 and pre2 and pre1 in _DIGIT_CHARS and pre2 in _DIGIT_CHARS and pre1 != pre2:
        return True
    # 中文量词前缀枚举：sub 紧邻前是物量词（个/只/条…），且左列数字不同 → 共用量词+
    # 名词的并列列举（如"3个苹果4个苹果"的"苹果"）。若数字相同（如"3个苹果3个苹果"）
    # 属重说/口误，不在此跳过，交由上层判定，避免漏检真实重复。
    if pre1 in _MEASURE_WORDS and pre2 in _MEASURE_WORDS:
        n1 = _num_before_measure(txt, pos1 - 1)
        n2 = _num_before_measure(txt, pos2 - 1)
        if n1 and n2 and n1 != n2:
            return True
    # 多位数字枚举：sub 常为数字片段（如 "234日""2日""4岁""234"），单位可能落在
    # sub 右侧。向左跨小数点补全数字、向右吞单位字符得到完整"数字+单位"，再比：
    # 两个完整数字不同、且单位后缀相同且非空 → 枚举；完整数字相同（如 false-start）
    # 或没有共享单位 → 仍按真实重复处理。
    s1, e1 = _expand_full(txt, pos1, pos1 + length)
    s2, e2 = _expand_full(txt, pos2, pos2 + length)
    if s1 != pos1 or s2 != pos2 or e1 != pos1 + length or e2 != pos2 + length:
        num1, num2 = txt[s1:e1], txt[s2:e2]
        if (num1 and num2 and num1 != num2
                and any(c in _DIGIT_CHARS for c in num1)
                and any(c in _DIGIT_CHARS for c in num2)):
            u1 = num1[_num_prefix_len(num1):]
            u2 = num2[_num_prefix_len(num2):]
            if u1 and u1 == u2:  # 单位后缀相同且非空 → 共用词缀的并列列举
                return True
    return False



def _is_legit_reduplication(sub: str) -> bool:
    """判断紧邻重复（gap 为空）的子串 sub 是否为汉语合法叠词，需跳过：
      - A一A 式：想一想 / 看一看 / 试一试
      - 数字(一/两)+量词 的「逐一」强调：一个一个 / 一波一波 / 两个两个
      - 动词/形容词 AA 式及其 AABB 式的 AB 段：研究研究 / 高高兴兴 / 明明白白
    其余紧邻双写（我说我说、我们我们、这个这个）视为口误重说，不跳过。
    """
    if not sub:
        return False
    if len(sub) == 3 and sub[0] == sub[2] and sub[1] == "一":
        return True
    if len(sub) == 2:
        if sub[0] in _REDUP_EMPHASIS_NUM and sub[1] in _REDUP_CLASSIFIERS:
            return True
        if sub in _REDUP_WORDS:
            return True
    return False


def _is_natural_reduplication(txt: str, sub: str, pos1: int, pos2: int) -> bool:
    """判断该重复是否为自然并列重复（不是口误，应跳过）。

    两类形态：
      1) 紧邻重复（两单元间隔 0 字）：ABAB 式。但**仅当 sub 是「合法叠词」才跳过**
         （见 _is_legit_reduplication）；其余紧邻双写（我说我说、我们我们、这个这个）
         视作口误重说，保留送研判 —— 漏检(false negative)在此会被永久跳过、不再送 LLM，
         危害大于偶尔误报（误报可由上层 LLM 研判驳回）。
      2) 短连接词隔开的并列重复：如「一波还有一波」「一个又一个」，两个重复单元之间
         仅隔 1~3 个连接性虚词或连接词短语（注意集合已不含犹疑 filler）。
    """
    length = len(sub)
    gap_start = pos1 + length
    gap_end = pos2
    gap = txt[gap_start:gap_end]
    if not gap:
        return _is_legit_reduplication(sub)
    if 1 <= len(gap) <= 3 and (
        gap in NATURAL_CONNECTOR_PHRASES
        or all(ch in NATURAL_CONNECTOR_CHARS for ch in gap)
    ):
        return True
    return False


def detect_intra(sentences: list[dict], original_script: str = "") -> list[dict]:
    """
    执行句内重复检测。
    
    策略: 对每句话，找长度 2~4 的子串，检查是否在短距离内再次出现
    
    Returns:
        findings: 检测结果列表
    """
    findings = []
    
    for sent in sentences:
        txt = sent["text"]
        hits = []
        
        # 尝试不同长度的子串 (2~4 字)
        for length in range(2, 5):
            for i in range(len(txt) - length):
                sub = txt[i:i + length]
                # 纯语气词/叹词组成的子串重复（啊啊、呢呢、呃呃…）跳过，不算口误
                if all(ch in MODAL_CHARS for ch in sub):
                    continue
                j = txt.find(sub, i + length)  # 在后面查找相同子串
                # 距离阈值：纯数字子串放宽到 8 字符（数字远距离重复几乎都是口误重说，
                # 如"8000股全部卖掉8000"），其余仍为 <= 3 字符。
                max_gap = INTRA_MAX_GAP_DIGIT if all(_is_digit_char(ch) for ch in sub) else INTRA_MAX_GAP_OTHER
                if j != -1 and (j - (i + length)) <= max_gap:
                    # 自然叠词 / 并列重复（如"一波一波""一波还有一波""寻思寻思"）
                    # 是正常口语，不是口误，跳过。
                    if _is_natural_reduplication(txt, sub, i, j):
                        continue
                    # 数字+单位并列枚举（如 5日10日20日、百分之三百分之四、
                    # 七蚊九七蚊八）是共用词缀的并列列举，不是口误，跳过。
                    if _is_parallel_enumeration(txt, sub, i, j):
                        continue
                    hits.append({
                        "phrase": sub,
                        "pos1": i,
                        "pos2": j,
                    })
        
        if hits:
            findings.append({
                "type": "intra_repeat",
                "sent_idx": sent["idx"],
                "range": sent["range"],
                "text": txt,
                "hits": hits,
                "decision_hint": (
                    "句内重复→只删前面片段(精确word_idx), 不整句删; "
                    "需LLM判断是否误报(自然并列不算)"
                ),
            })
    
    return findings


def run_detect_intra(
    sentences: list[dict],
    output_dir: Path,
    words: list[dict],
    original_script: str,
) -> list[dict]:
    """运行句内重复检测（不写文件，仅返回结果）"""
    
    findings = detect_intra(sentences, original_script)
    
    print(f"   [detect_intra] 句内重复发现: {len(findings)} 句")
    for fnd in findings:
        desc = ", ".join(f"{h['phrase']}@{h['pos1']}..{h['pos2']}" for h in fnd["hits"])
        print(f"      - 句{fnd['sent_idx']}: {desc}")
    if not findings:
        print(f"      (无句内重复)")
    
    return findings
