"""compare_truth.py — 把「检测修正后句子」与「真值句子」做精确比对。

真值 = base/sentences.txt（原始口播，无误删/漏删）；
final = 检测修正后句子（如 run_speech_pipeline 的 2_分析/sentences.txt，
        或 test_v3 的 sentences_final.txt）。

基于「谁包含谁」精确分类，避免把【片段级漏删/误删】藏进文本差异：
  - 误删(整句): 真值有文本 / final 空
  - 漏删(整句): 真值空 / final 有文本
  - 漏删(片段): 都有文本且 真值⊂final → final 多保留了真值删掉的片段（就是漏删！）
  - 误删(片段): 都有文本且 final⊂真值 → final 把真值保留的文本删掉了一段（就是误删！）
  - 文本差异:   都有文本但互不包含（内容级替换/改写，非简单的多删少删）
  - 仅final多出: final 有、真值无的句编号
"""

import re
from pathlib import Path


def _compare_parse(path: Path) -> dict:
    """解析 `idx|range|text` 为 {idx: text.strip()}。"""
    m = {}
    if not path.exists():
        return m
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 2)
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0].strip())
        except ValueError:
            continue
        m[idx] = parts[2].strip() if len(parts) >= 3 else ""
    return m


def compare_with_truth(final_path: Path, truth_path: Path) -> dict:
    """比对 final 句子与 truth 真值句子，打印并落盘 compare_vs_truth.txt。

    final_path: 检测修正后的 sentences（如 2_分析/sentences.txt / sentences_final.txt）
    truth_path: 真值 sentences（如 base/sentences.txt）
    返回汇总 dict；真值文件不存在时打印跳过并返回空 dict。
    """
    final_path = Path(final_path)
    truth_path = Path(truth_path)
    if not truth_path.exists():
        print(f"\n⚠️ 未找到真值文件 {truth_path}，跳过比对")
        return {}
    truth = _compare_parse(truth_path)
    final = _compare_parse(final_path)
    missing, extra = [], []          # 整句 误删 / 漏删
    frag_miss, frag_false = [], []   # 片段 漏删 / 误删
    diff, only_final = [], []        # 内容级差异 / final 多出句
    ok_keep = ok_del = 0
    for idx, t in truth.items():
        f = final.get(idx, "")
        if t and not f:
            missing.append((idx, t))
        elif (not t) and f:
            extra.append((idx, f))
        elif t and f:
            if t == f:
                ok_keep += 1
            else:
                tn = re.sub(r"\s+", "", t)
                fn = re.sub(r"\s+", "", f)
                if tn and tn in fn and tn != fn:
                    # 真值是 final 的子串 → final 多保留了真值删掉的片段 = 漏删（片段级）
                    frag_miss.append((idx, t, f))
                elif fn and fn in tn and fn != tn:
                    # final 是真值的子串 → final 删掉了一段真值保留的文本 = 误删（片段级）
                    frag_false.append((idx, t, f))
                else:
                    diff.append((idx, t, f))
        else:  # 都空 → 都删，一致
            ok_del += 1
    for idx in final:
        if idx not in truth:
            only_final.append(idx)
    base = truth_path.parent.name
    lines = []
    lines.append(f"===== 比对（{base}）：真值 {truth_path.name} vs {final_path.name} =====")
    n_miss = len(missing) + len(frag_false)
    n_extra = len(extra) + len(frag_miss)
    lines.append(f"  ✅ 一致保留 {ok_keep} | ✅ 一致删除 {ok_del}")
    lines.append(
        f"  ❌ 误删 {n_miss}（整句{len(missing)}/片段{len(frag_false)}）"
        f" | ⚠️ 漏删 {n_extra}（整句{len(extra)}/片段{len(frag_miss)}）"
        f" | 🔶 文本差异 {len(diff)} | 🔸 仅final多出 {len(only_final)}"
    )
    for idx, t in missing:
        lines.append(f"    [误删·整句] 句{idx}: 真值={t!r}")
    for idx, f in extra:
        lines.append(f"    [漏删·整句] 句{idx}: final={f!r}")
    for idx, t, f in frag_miss:
        tn = re.sub(r"\s+", "", t)
        fn = re.sub(r"\s+", "", f)
        # 真值被 final 包含 → 多出来的片段即被漏删的文本（去空白后直观展示）
        extra_txt = fn.replace(tn, "", 1) if tn else fn
        lines.append(
            f"    [漏删·片段] 句{idx}: 真值删掉了「{extra_txt}」但 final 仍保留它\n"
            f"        真值={t!r}\n        final={f!r}"
        )
    for idx, t, f in frag_false:
        fn = re.sub(r"\s+", "", f)
        tn = re.sub(r"\s+", "", t)
        dropped = tn.replace(fn, "", 1) if fn else tn
        lines.append(
            f"    [误删·片段] 句{idx}: final 多删了「{dropped}」（真值保留）\n"
            f"        真值={t!r}\n        final={f!r}"
        )
    for idx, t, f in diff:
        lines.append(f"    [文本差异] 句{idx}:\n        真值={t!r}\n        final={f!r}")
    for idx in only_final:
        lines.append(f"    [仅final多出] 句{idx}: final={final[idx]!r}")
    for line in lines:
        print(line)

    # 落盘：与 final_path 同目录，文件名 compare_vs_truth.txt
    out_path = final_path.parent / "compare_vs_truth.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"📄 比对结果已保存: {out_path}\n")
    return {"missing": missing, "extra": extra, "frag_miss": frag_miss,
            "frag_false": frag_false, "diff": diff, "only_final": only_final,
            "ok_keep": ok_keep, "ok_del": ok_del, "report_path": str(out_path)}
