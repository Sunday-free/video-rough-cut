"""
detect_fragment.py — 残句机械检测 + 原稿对齐提示（整句删除专用）

检测策略（全部产出整句删除候选，即 mode="full"）：
1. 孤立编号句 (如 "三" 单独成句)
2. 极短孤立句 (<=2字符)
3. 跨句头-头口误重说（隔1-2句，去语气词后头部 ≥5 字匹配）→ 前句为口误版本，整句删
4. 跨句尾部重叠（隔1-2句，去语气词后尾部 ≥4 字匹配）→ 前句为残次版本，整句删

注：凡「前句头部独有、尾部被后句重说（keep_head 保头删尾）」的现象已迁至
detect_partial.py（句间部分删除），本模块只做整句删除。

输出: detect_fragment.json
"""

from pathlib import Path

from speech_error_detector.detect_repeat import CN_DIGIT_MAP, normalize_numerals
from speech_error_detector.utils.fillers import MODAL_CHARS

def _norm_frag(t: str) -> str:
    """去掉语气衬字 + 统一数字后的规范化文本（用于头/尾重叠匹配）。"""
    return "".join(CN_DIGIT_MAP.get(ch, ch) for ch in t if ch not in MODAL_CHARS)

def detect_fragment(sentences: list[dict], words: list | None = None, original_script: str = "") -> list[dict]:
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
        # 纯语气词短句（如"啊""嗯""哦"）属自然语流填充，非残句，跳过
        if t and all(ch in MODAL_CHARS for ch in t):
            continue
        if len(t) <= 2:
            if t in numbers:
                # 检查是否为有序列中的合法序号
                if _is_ordinal_sequence(sent["idx"]):
                    continue  # 是合法序号，跳过
                _f = {
                    "type": "fragment",
                    "subtype": "孤立编号",
                    "sent_idx": sent["idx"],
                    "range": sent["range"],
                    "text": t,
                    "decision_hint": "孤立编号(如'三')→删整句, 内容在后续句",
                }
                findings.append(_f)
            else:
                _f = {
                    "type": "fragment",
                    "subtype": "极短孤立句",
                    "sent_idx": sent["idx"],
                    "range": sent["range"],
                    "text": t,
                    "decision_hint": "疑似残句/孤立→删整句",
                }
                findings.append(_f)
    
    # （前句尾与后句头重叠 / 头体重叠 等 keep_head 现象已迁至 detect_partial.py）

    # 2. 跨句头-头口误重说（隔1-2句，去语气词后头部 ≥5 字匹配）
    #    前句 A 头部与后句 B 头部相同 → A 为口误版本，应整句删除(mode=full)
    #    例: 34="后来呢他给我呃在复盘以前做的对照实验"
    #        36="后来呢他给我复盘以前做的对照实验我亲眼看见"
    #        → 34 头-头匹配 5 字，口误重说，整句删 34
    _HEAD_HH_MIN = 5
    _CROSS_GAP_MAX = 4
    for j in range(2, len(sentences)):
        b_text = sentences[j]["text"]
        if len(b_text) < _HEAD_HH_MIN:
            continue
        nb = _norm_frag(b_text)
        for gap in range(2, min(j, _CROSS_GAP_MAX) + 1):
            i = j - gap
            a_text = sentences[i]["text"]
            if len(a_text) < _HEAD_HH_MIN:
                continue
            na = _norm_frag(a_text)
            k = 0
            while k < min(len(na), len(nb)) and na[k] == nb[k]:
                k += 1
            if k >= _HEAD_HH_MIN:
                findings.append({
                    "type": "fragment",
                    "subtype": f"跨句头-头口误重说(跳过{gap-1}句)",
                    "sent_idx": sentences[i]["idx"],
                    "range": sentences[i]["range"],
                    "text": a_text,
                    "next_sent_idx": sentences[j]["idx"],
                    "next_sent_text": b_text,
                    "overlap_len": k,
                    "head_text": "",
                    "head_char_len": 0,
                    "decision_hint": (
                        f"删前保后(mode=full)→句{sentences[i]['idx']}头部与句{sentences[j]['idx']}"
                        f"头部{k}字相同(去语气词,跳过{gap-1}句)，前句为口误版本，"
                        f"后句已重说并补全，应整句删除句{sentences[i]['idx']}"
                    ),
                })
                break

    # 3. 跨句尾部重叠（隔1-2句，去语气词后尾部 ≥4 字匹配）
    #    前句 A 尾部与后句 B 尾部相同 → A 为残次版本，应整句删除(mode=full)
    #    例: 33="我呢一直是记到现在" 35="...把我镇住了我一直记到现在"
    #        → 33 尾部 4 字与 35 尾部相同，残次版本，整句删 33
    _TAIL_OVERLAP_MIN = 4
    for j in range(2, len(sentences)):
        b_text = sentences[j]["text"]
        nb = _norm_frag(b_text)
        if len(nb) < _TAIL_OVERLAP_MIN:
            continue
        for gap in range(2, min(j, _CROSS_GAP_MAX) + 1):
            i = j - gap
            a_text = sentences[i]["text"]
            na = _norm_frag(a_text)
            if len(na) < _TAIL_OVERLAP_MIN or len(na) >= len(nb):
                continue
            k = min(_TAIL_OVERLAP_MIN, len(na), len(nb))
            if na[-k:] == nb[-k:]:
                findings.append({
                    "type": "fragment",
                    "subtype": f"跨句尾部重叠(跳过{gap-1}句)",
                    "sent_idx": sentences[i]["idx"],
                    "range": sentences[i]["range"],
                    "text": a_text,
                    "next_sent_idx": sentences[j]["idx"],
                    "next_sent_text": b_text,
                    "overlap_len": k,
                    "head_text": "",
                    "head_char_len": 0,
                    "decision_hint": (
                        f"删前保后(mode=full)→句{sentences[i]['idx']}尾部与句{sentences[j]['idx']}"
                        f"尾部重叠{k}字(去语气词,跳过{gap-1}句)，前句为残次版本，"
                        f"应整句删除句{sentences[i]['idx']}"
                    ),
                })
                break

    return findings

def run_detect_fragment(
    sentences: list[dict],
    output_dir: Path,
    words: list[dict],
    original_script: str,
) -> list[dict]:
    """运行残句检测（不写文件，仅返回结果）"""
    
    findings = detect_fragment(sentences, words=words, original_script=original_script)
    
    print(f"   [detect_fragment] 残句发现: {len(findings)} 处")
    for fnd in findings:
        print(f"      - {fnd['subtype']} 句{fnd['sent_idx']}: {fnd['text'][:24]}")
    if not findings:
        print(f"      (无残句)")
    
    return findings
