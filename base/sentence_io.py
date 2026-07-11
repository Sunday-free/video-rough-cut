"""统一的 sentences.txt 读写工具。

sentences.txt 每行格式:  idx|start-end|text
- idx:       句子编号 (int)
- start-end: word 索引区间 (字符串)
- text:      句子文本

load_sentences 返回统一的 [{idx, range, text}, ...]。
expand_range=True 时额外解析出 startIdx/endIdx 整数字段（assemble 等需要）。
write_sentences 把统一的 [{idx, range, text}] 写回文件；idx 断档时保留空行。
"""

from pathlib import Path


def _parse_line(line: str) -> dict | None:
    line = line.rstrip("\n")
    if not line:
        return None
    p = line.split("|")
    idx = int(p[0])
    rng = p[1] if len(p) > 1 else ""
    text = p[2] if len(p) > 2 else ""
    return {"idx": idx, "range": rng, "text": text}


def load_sentences(path: Path, *, expand_range: bool = False) -> list[dict]:
    """加载 sentences.txt → [{idx, range, text}, ...]。

    expand_range=True 时额外附带 startIdx/endIdx（从 range 解析）。
    """
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = _parse_line(line)
            if item is None:
                continue
            if expand_range and item["range"] and "-" in item["range"]:
                a, b = item["range"].split("-")
                item["startIdx"] = int(a)
                item["endIdx"] = int(b)
            out.append(item)
    return out


def _range_of(s: dict) -> str:
    """从句子 dict 取得 range 字符串（兼容 range 字段或 startIdx/endIdx）。"""
    r = s.get("range")
    if r:
        return r
    a, b = s.get("startIdx"), s.get("endIdx")
    if a is not None and b is not None and a >= 0 and b >= 0:
        return f"{a}-{b}"
    return ""


def write_sentences(
    path: Path,
    sentences: list[dict],
    *,
    original: list[dict] | None = None,
) -> None:
    """写出 sentences_*.txt（统一格式: idx|start-end|text）。

    与 load_sentences 配对。若句子列表的 idx 不连续（存在被整句删除、从而
    缺失的行），会**保留空行**以保持行号对齐：

        30|531-562|那会呢我还没有完全地琢磨明白但是呢这句话可是给我镇住了啊
        31|581-601|            ← 断档处空行（range 取自 original）
        32|581-601|
        33|581-601|
        34|581-601|后来呢他给我呃在复盘以前做的对照实验

    规则：
    - 文本被删空但仍保留的句子 → 写为 `idx|range|`（空 text，不跳过）。
    - idx 断档（整句删除，已不在 sentences 中）→ 写为空行；range 尽量从
      original 取，无 original 时 range 留空（写作 `idx||`）。

    Args:
        sentences: 当前要写出的句子列表（可能缺失被整句删除的 idx）。
        original:  可选，删除前的全量句子（含被删句的 idx/range）。仅用于为
                   断档处的空行补全 range；不传则断档行 range 留空。
    """
    sent_by_idx = {s["idx"]: s for s in sentences if "idx" in s}
    orig_by_idx = {s["idx"]: s for s in (original or []) if "idx" in s}

    idxs = set(sent_by_idx) | set(orig_by_idx)
    if not idxs:
        content = ""
    else:
        lo, hi = min(idxs), max(idxs)
        lines = []
        for i in range(lo, hi + 1):
            s = sent_by_idx.get(i)
            if s is not None:
                r = _range_of(s)
                t = s.get("text", "")
            else:
                # 断档：被整句删除，补空行；range 尽量从 original 取
                o = orig_by_idx.get(i)
                r = _range_of(o) if o is not None else ""
                t = ""
            lines.append(f"{i}|{r}|{t}")
        content = "\n".join(lines) + "\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
