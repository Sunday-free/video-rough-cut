"""
detect_fragment.py — 残句机械检测 + 原稿对齐提示

检测策略:
1. 孤立编号句 (如 "三" 单独成句)
2. 极短孤立句 (<=2字符)
3. 前句尾与后句头重叠 (残句被后句接续重说)

输出: detect_fragment.json
"""

from pathlib import Path

def detect_fragment(sentences: list[dict]) -> list[dict]:
    """
    执行残句检测。
    
    Returns:
        findings: 检测结果列表
    """
    findings = []
    
    # 中文数字编号集合（有序）
    numbers = {"一", "二", "三", "四", "五", "六", "七", "八", "九", "十"}
    
    def _is_ordinal_sequence(sent_idx: int) -> bool:
        """检查孤立编号是否为有序列的一部分"""
        t = sentences[sent_idx]["text"]
        
        # 查找前后是否有其他序号句（形成序列证据）
        has_ordinal_neighbor = False
        for other in sentences:
            if other["idx"] == sent_idx:
                continue
            ot = other["text"]
            # 其他句子以中文数字或"第X"开头 → 说明这是编号体系
            if ot and (ot[0] in numbers or ot.startswith("第")):
                has_ordinal_neighbor = True
                break
        
        if not has_ordinal_neighbor:
            return False
        
        # 检查后一句是否像内容延续（非同样极短）
        next_sent = None
        for s in sentences:
            if s["idx"] == sent_idx + 1:
                next_sent = s
                break
        
        if next_sent and len(next_sent["text"]) > 5:
            # 若后句以同一中文数字开头 → 说明本句是 fragment 残留，不是合法序号
            # 例：句40 "二" / 句42 "二人工智能..." → 句40 是残句，不应放过
            # if next_sent["text"][0] == t:
            #     return False
            return True
        
        return False
    
    # 1. 极短句检测 (孤立编号 / 极短孤立)
    for sent in sentences:
        t = sent["text"]
        if len(t) <= 2:
            if t in numbers:
                # 检查是否为有序列中的合法序号
                if _is_ordinal_sequence(sent["idx"]):
                    continue  # 是合法序号，跳过
                findings.append({
                    "type": "fragment",
                    "subtype": "孤立编号",
                    "sent_idx": sent["idx"],
                    "range": sent["range"],
                    "text": t,
                    "decision_hint": "孤立编号(如'三')→删整句, 内容在后续句",
                })
            else:
                findings.append({
                    "type": "fragment",
                    "subtype": "极短孤立句",
                    "sent_idx": sent["idx"],
                    "range": sent["range"],
                    "text": t,
                    "decision_hint": "疑似残句/孤立→删整句",
                })
    
    # 2. 前句尾与后句头重叠（残句被接续）
    for i in range(len(sentences) - 1):
        a, b = sentences[i]["text"], sentences[i + 1]["text"]
        overlap = 0
        # 检查 a 的尾部是否与 b 的头部重叠 (4~8 字符)
        for L in range(4, min(8, len(a), len(b)) + 1):
            if a[-L:] == b[:L]:
                overlap = L
        
        if overlap >= 4:
            a_head = a[: len(a) - overlap] if overlap > 0 else a
            # 头部为空 → 整句完整包含于后句，无独有内容，应整句删除
            if len(a_head) == 0:
                decision_hint = (
                    f"删前保后→句{sentences[i]['idx']}『{a}』完整包含于后句『{b}』头部"
                    f"(重叠{overlap}字), head_text 为空无独有内容, 必须 mode=full 整句删除"
                )
            else:
                decision_hint = (
                    f"删前保后→默认保头删尾; 句{sentences[i]['idx']}头部『{a_head}』为独有内容"
                    f"(后句『{b}』不含该内容), 必须 mode=keep_head 保留头部、只删尾部重叠的"
                    f"『{a[len(a) - overlap:]}』; 仅当头部被后句完整包含时才允许 mode=full 整句删"
                )
            findings.append({
                "type": "fragment",
                "subtype": "残句(被后句接续重说)",
                "sent_idx": sentences[i]["idx"],
                "range": sentences[i]["range"],
                "text": a,
                "next_sent_idx": sentences[i + 1]["idx"],
                "next_sent_text": b,
                "overlap_len": overlap,
                "head_text": a_head,
                "head_char_len": len(a_head),
                "decision_hint": decision_hint,
            })
    
    return findings


def run_detect_fragment(
    sentences: list[dict],
    output_dir: Path | None = None,
) -> list[dict]:
    """运行残句检测（不写文件，仅返回结果）"""
    
    findings = detect_fragment(sentences)
    
    print(f"   [detect_fragment] 残句发现: {len(findings)} 处")
    for fnd in findings:
        print(f"      - {fnd['subtype']} 句{fnd['sent_idx']}: {fnd['text'][:24]}")
    if not findings:
        print(f"      (无残句)")
    
    return findings
