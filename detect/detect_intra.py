"""
detect_intra.py — 句内重复机械检测

检测策略: N-gram 子串在短距离（<=3字符）内重复出现

输出: detect_intra.json
"""

from pathlib import Path

from ..base.fillers import MODAL_CHARS

# 自然叠词 / 并列重复的连接性虚词：两个重复单元之间仅隔这些字（1~3 个）时，
# 视为汉语正常的并列强调（如"一波还有一波""一个又一个""一遍又一遍"），不算口误。
NATURAL_CONNECTOR_CHARS = set(
    "还又再来也的就而则且跟与和等啊呀哦嗯呐嘞咯呃哇嘢哟哈有"
)

# 常见双字连接词短语：间隔为这些短语时同样视为自然并列重复。
NATURAL_CONNECTOR_PHRASES = {
    "还有", "而后", "然后", "之后", "接着", "跟着", "以及", "再来",
    "又来", "而又", "再又", "又再",
}


def _is_natural_reduplication(txt: str, sub: str, pos1: int, pos2: int) -> bool:
    """
    判断该重复是否为自然叠词 / 并列重复（不是口误，应跳过）。

    自然重复的两类形态：
      1) 紧邻重复（两单元间隔 0 字）：ABAB 式叠词，如「寻思寻思」「一波一波」
         「研究研究」「想一想」。机械检测无法区分"口吃重说"与"合法叠词"，
         但紧邻整词双写在口语口播里绝大多数是正常叠词，故跳过。
      2) 短连接词隔开的并列重复：如「一波还有一波」「一个又一个」，两个重复
         单元之间仅隔 1~3 个连接性虚词或连接词短语。
    """
    length = len(sub)
    gap_start = pos1 + length
    gap_end = pos2
    gap = txt[gap_start:gap_end]
    if not gap:
        # 紧邻重复 → ABAB 叠词
        return True
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
                if j != -1 and (j - (i + length)) <= 3:  # 距离 <= 3 字符
                    # 自然叠词 / 并列重复（如"一波一波""一波还有一波""寻思寻思"）
                    # 是正常口语，不是口误，跳过。
                    if _is_natural_reduplication(txt, sub, i, j):
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
