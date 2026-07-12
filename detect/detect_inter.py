"""
detect_inter.py — 句间重复机械检测

检测策略:
1. 相邻句头前缀匹配 (head_eq >= 5字)
2. 隔句重复（中间是短残句）

输出: detect_inter.json
"""

from pathlib import Path

from . import CN_DIGIT_MAP, normalize_numerals


def head_eq(a: str, b: str, n: int = 5) -> int:
    """计算两字符串的共同前缀长度（最多 n 字符），归一化数字后比较。"""
    na, nb = normalize_numerals(a), normalize_numerals(b)
    k = 0
    while k < n and k < len(na) and k < len(nb) and na[k] == nb[k]:
        k += 1
    return k


# 常见语气/停顿衬字：做"近重复"判定时忽略，避免 呢/呃 等差导致漏检
_FILLERS = set("呢呃啊呀嘛吧嗯哦呐哈嘞哟喂啦咯")


def _norm(t: str) -> str:
    """去掉语气衬字 + 统一数字后的规范化文本（用于近重复匹配）。"""
    return "".join(CN_DIGIT_MAP.get(ch, ch) for ch in t if ch not in _FILLERS)


def detect_inter(sentences: list[dict]) -> list[dict]:
    """
    执行句间重复检测。
    
    Args:
        sentences: 句子列表（含 range 字段，如 \"522-536\"）
        words: 词级时间数据列表（含 start/end 时间戳），用于计算句间间隔
    
    Returns:
        findings: 检测结果列表
    """
    findings = []
    N = 5  # 最小前缀匹配长度
    
    # 1. 相邻句头前缀重复
    for i in range(len(sentences) - 1):
        a, b = sentences[i]["text"], sentences[i + 1]["text"]
        k = head_eq(a, b, N)
        if k >= N and len(a) > 0 and len(b) > 0:
            findings.append({
                "type": "inter_repeat",
                "subtype": "相邻句头重复",
                "sent_a_idx": sentences[i]["idx"],
                "sent_a_range": sentences[i]["range"],
                "sent_b_idx": sentences[i + 1]["idx"],
                "sent_b_range": sentences[i + 1]["range"],
                "text_a": a,
                "text_b": b,
                "common_prefix_len": k,
                "head_a": a[:8],
                "head_b": b[:8],
                "decision_hint": f"删前保后→删除句子{sentences[i]['idx']}",
            })
    
    # 2. 隔句重复（中间是短残句/孤立编号）
    for i in range(len(sentences) - 2):
        mid = sentences[i + 1]["text"]
        if len(mid) <= 5:  # 中间句很短
            a, c = sentences[i]["text"], sentences[i + 2]["text"]
            k = head_eq(a, c, N)
            if k >= N:
                findings.append({
                    "type": "inter_repeat",
                    "subtype": "隔句重复(中间短残句)",
                    "sent_a_idx": sentences[i]["idx"],
                    "sent_a_range": sentences[i]["range"],
                    "mid_sent_idx": sentences[i + 1]["idx"],
                    "mid_range": sentences[i + 1]["range"],
                    "sent_c_idx": sentences[i + 2]["idx"],
                    "sent_c_range": sentences[i + 2]["range"],
                    "text_a": a,
                    "text_c": c,
                    "common_prefix_len": k,
                    "decision_hint": f"删前保后→删除句{sentences[i]['idx']}及中间句{sentences[i+1]['idx']}",
                })
    
    # 3. 非前缀子串重叠检测（句子B包含了句子A的大部分内容，但不在开头）
    #    例: 句A="我一直记到现在..." 句B="...也把我镇住了我一直记到现在..."
    #    前缀没对上但句B包了句A的大部分 → 句A是前序不完整版本
    for i in range(len(sentences) - 1):
        a_text, b_text = sentences[i]["text"], sentences[i + 1]["text"]
        na, nb = _norm(a_text), _norm(b_text)
        if len(na) < 8 or len(nb) < 12:
            continue
        
        # 用滑动窗口检测：从 A 的多个偏移位置取子串，检查是否在 B 中
        max_overlap = 0
        for start in range(1, min(len(na) - 8, 12)):
            # 从 start 开始取最长子串在 B 中匹配
            remaining = na[start:]
            for win_len in range(min(len(remaining), 30), 7, -1):
                sub = remaining[:win_len]
                if sub in nb:
                    max_overlap = max(max_overlap, win_len)
                    break
        
        if max_overlap >= 10 and len(a_text) > 0:
            findings.append({
                "type": "inter_repeat",
                "subtype": "非前缀子串重叠",
                "sent_a_idx": sentences[i]["idx"],
                "sent_a_range": sentences[i]["range"],
                "sent_b_idx": sentences[i + 1]["idx"],
                "sent_b_range": sentences[i + 1]["range"],
                "text_a": a_text,
                "text_b": b_text,
                "common_prefix_len": max_overlap,
                "head_a": a_text[:8],
                "head_b": b_text[:8],
                "decision_hint": f"句{sentences[i+1]['idx']}的非头部包含了句{sentences[i]['idx']}的大量内容(重叠{max_overlap}字)→疑似前句为残次版本，建议删前保后",
            })

    # 4. 子串完全包含（句A 是句B 的子串 → 句A 为冗余残句）
    #    例: 句A="6万4的资金" 句B="八块成交到手6万4的资金" → A 完全包含于 B，
    #        无论前缀还是后缀重叠都应判为前句残次、删前保后。
    #    注: 策略3 要求 len(a) >= 8 会漏掉短残句，这里单独处理（len >= 4 即可）。
    for i in range(len(sentences) - 1):
        a_text = sentences[i]["text"]
        b_text = sentences[i + 1]["text"]
        na, nb = _norm(a_text), _norm(b_text)
        if (len(na) >= 5
                and len(na) < len(nb)
                and na in nb):
            findings.append({
                "type": "inter_repeat",
                "subtype": "子串完全包含(前句为残句)",
                "sent_a_idx": sentences[i]["idx"],
                "sent_a_range": sentences[i]["range"],
                "sent_b_idx": sentences[i + 1]["idx"],
                "sent_b_range": sentences[i + 1]["range"],
                "text_a": a_text,
                "text_b": b_text,
                "common_prefix_len": len(na),
                "head_a": a_text[:8],
                "head_b": b_text[:8],
                "decision_hint": f"句{sentences[i]['idx']}『{a_text}』(去语气词后)完全包含于句{sentences[i+1]['idx']}，为冗余残句，建议删前保后",
            })

    # 5. 尾部重叠（前句是后句结尾的残次近重复）
    #    例: 句A="我呢一直是记到现在" 句B="...把我镇住了我一直记到现在"
    #        A 的句尾 与 B 的句尾 高度一致（差个语气词/赘余字），A 为未完成口误，
    #        应删前保后。精确/子串匹配会被 呢/是 等差漏掉，故用尾部对齐 + 去语气词。
    TAIL_K = 4
    for i in range(len(sentences) - 1):
        a_text = _norm(sentences[i]["text"])
        if len(a_text) < 5:
            continue
        hit = False
        for j in range(i + 1, min(i + 3, len(sentences))):
            b_text = _norm(sentences[j]["text"])
            k = min(TAIL_K, len(a_text), len(b_text))
            if k < TAIL_K:
                continue
            if (len(a_text) < len(b_text)
                    and a_text[-k:] == b_text[-k:]):
                findings.append({
                    "type": "inter_repeat",
                    "subtype": "尾部重叠(前句为残次近重复)",
                    "sent_a_idx": sentences[i]["idx"],
                    "sent_a_range": sentences[i]["range"],
                    "sent_b_idx": sentences[j]["idx"],
                    "sent_b_range": sentences[j]["range"],
                    "text_a": sentences[i]["text"],
                    "text_b": sentences[j]["text"],
                    "common_prefix_len": k,
                    "head_a": sentences[i]["text"][:8],
                    "head_b": sentences[j]["text"][:8],
                    "decision_hint": f"句{sentences[i]['idx']}『{sentences[i]['text']}』句尾与句{sentences[j]['idx']}句尾重叠{k}字(去语气词)，为残次近重复，建议删前保后",
                })
                hit = True
                break
        if hit:
            continue

    # 去重：同一句子对可能被多个策略命中（如"相邻句头重复"+"子串完全包含"），
    # 保留最具体的 subtype，去除冗余。
    _PRIORITY = {
        "子串完全包含(前句为残句)": 5,
        "非前缀子串重叠": 4,
        "尾部重叠(前句为残次近重复)": 3,
        "相邻句头重复": 2,
        "隔句重复(中间短残句)": 1,
    }
    seen: dict[tuple, tuple[int, int | None]] = {}  # key -> (priority, deduped_index)
    deduped: list[dict | None] = []
    for i, f in enumerate(findings):
        pair_idx = f.get("sent_b_idx", f.get("sent_c_idx", -1))
        key = (f["sent_a_idx"], pair_idx)
        p = _PRIORITY.get(f["subtype"], 0)
        if key not in seen or p > seen[key][0]:
            if key in seen:
                deduped[seen[key][1]] = None  # 标记旧项待移除
            seen[key] = (p, len(deduped))
            deduped.append(f)
        # lower priority, skip

    findings = [f for f in deduped if f is not None]

    return findings


def run_detect_inter(
    sentences: list[dict],
    output_dir: Path,
    words: list[dict],
    original_script: str,
) -> list[dict]:
    """运行句间重复检测（不写文件，仅返回结果）"""
    
    findings = detect_inter(sentences)
    
    print(f"   [detect_inter] 句间重复发现: {len(findings)} 处")
    for fnd in findings:
        other = fnd.get("sent_b_idx") or fnd.get("sent_c_idx")
        print(f"      - {fnd['subtype']} 句{fnd['sent_a_idx']} vs 句{other} "
              f"(同前缀{fnd['common_prefix_len']}字)")
    if not findings:
        print(f"      (无句间重复)")
    
    return findings
