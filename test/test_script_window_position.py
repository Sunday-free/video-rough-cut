"""
test_script_window_position.py — 验证原文稿窗口定位的「位置消歧」

背景 / 复现
===========
红姐语料里，句31「寄到」、句32「县」是极短孤立句，build_short_org_window
借前后句(±1 / ±2) 的长句在原稿中定位，再截取相邻片段作为窗口。

旧实现（无位置先验）的问题：
  - match_script_position 对重复/相似文本只取最靠前的匹配块（difflib 的 b.a /
    find 的首个出现）。
  - 原稿是口播稿，套话/相似结构很多。相邻长句被模糊匹配到原稿**更靠前**的
    相似片段，导致窗口错位到与上下文无关的位置
    （如句32「县」的窗口误命中「跌个二十个点就到底反弹…」段）。
  - 实证：旧 judge_prompt_fragment_round1.txt 里 #1（句32）的对照片段就是错位的。

新实现（位置消歧）：
  - match_script_position 新增 expected_pos(idx/总句数 × 原稿长) 与 lower(顺序下界)。
  - 多候选时选离 expected_pos 最近的；后句匹配不得早于前句结束。
  - 旧 judge_prompt 为消歧前生成；当前代码实测句31/32 窗口已正确落在
    「那会…琢磨明白…这句话给我镇住了」故事段。

运行:
  python3 -m speech_error_detector.test.test_script_window_position
"""

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speech_error_detector.test.test_fragment_pipeline import HONGJIE_SENTENCES
from speech_error_detector.detect_repeat.script_window import (
    build_short_org_window,
    match_script_position,
)

_SCRIPT_REL = "2026-07-07_红姐/original_script.txt"
_SENT_ROUND1_REL = "2026-07-07_红姐/2_分析/detect_repeat/sentences_round_1.txt"


def _load_script() -> str:
    for base in (os.getcwd(), _ROOT):
        p = os.path.join(base, _SCRIPT_REL)
        if os.path.exists(p):
            return open(p, encoding="utf-8").read()
    raise FileNotFoundError(_SCRIPT_REL)


def _load_round1_sentences() -> list:
    """加载 round1 的真实句子列表（句31=寄到 / 句32=县 非空）。

    这是关键的回归数据源：此前用 HONGJIE_SENTENCES 常量测时句31/32 文本为空，
    极短邻居被跳过，掩盖了「真实非空极短句被当邻居误匹配并拽长窗口」的 bug。
    """
    for base in (os.getcwd(), _ROOT):
        p = os.path.join(base, _SENT_ROUND1_REL)
        if not os.path.exists(p):
            continue
        out = []
        for line in open(p, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            idx_s, rest = line.split("|", 1)
            rng, txt = rest.split("|", 1)
            out.append({"idx": int(idx_s), "range": rng, "text": txt})
        return out
    raise FileNotFoundError(_SENT_ROUND1_REL)


def _snippet(script: str, pos: int, before: int = 20, after: int = 50) -> str:
    if pos < 0:
        return "<未匹配>"
    return script[max(0, pos - before): pos + after].replace("\n", " ")


def test_short_window_lands_on_correct_story():
    """修复后：极短句的窗口应落在『那会…琢磨明白…镇住了…记到现在』故事段。"""
    script = _load_script()
    for idx in (31, 32):
        w = build_short_org_window(script, HONGJIE_SENTENCES, idx)
        print(f"[正确行为] 句{idx} WINDOW = {w!r}")
        assert w is not None, f"句{idx} 应能定位到原稿"
        assert "琢磨" in w, f"句{idx} 窗口错落在无关片段: {w!r}"
    print("  ✅ 句31/32 窗口均正确落在『那会…琢磨明白…镇住了』故事段\n")


def test_short_window_focused_not_overlong():
    """核心回归：真实 round1 句子（句31=寄到 / 句32=县 非空）的窗口必须聚焦在
    『记到现在』附近，且不得延伸到更后的『复盘…赚七成收益』段。

    此前 bug：build_short_org_window 只取单侧邻居 + 右边界用 pos+len(长句) 越界，
    导致窗口一路滑到『赚了七成收益。』。
    """
    script = _load_script()
    sentences = _load_round1_sentences()
    for idx in (31, 32):
        w = build_short_org_window(script, sentences, idx)
        print(f"[聚焦回归] 句{idx} WINDOW = {w!r}")
        assert w is not None, f"句{idx} 应能定位到原稿"
        # 必须包含『记到现在』——极短句寄到/县 应夹在『震住了』与『记到现在』之间
        assert "记到现在" in w, f"句{idx} 窗口未覆盖『记到现在』: {w!r}"
        # 允许包含紧邻的后续从句（如『后来他给我复盘…』）作为尾随上下文；
        # 但不得一路涨到更后的段落（『赚了七成收益』）。
        assert "赚了七成收益" not in w, f"句{idx} 窗口过长、越界到『赚七成收益』: {w!r}"
        # 长度应紧凑（聚焦窗口远小于整段，且不盲目膨胀）
        assert len(w) < 100, f"句{idx} 窗口过长({len(w)}字): {w!r}"
    print("  ✅ 句31/32 窗口聚焦在『记到现在』附近，未越界到后续段落\n")


def test_position_disambiguation_picks_later_occurrence():
    """合成复现：文本重复出现时，expected_pos 应消歧到靠后一处。"""
    script = "开场白。今天天气真好。我们开会讨论方案。中间过渡。今天天气真好。会议结束了。"
    text = "今天天气真好"
    p0 = match_script_position(script, text)                     # 旧行为：首个出现
    p1 = match_script_position(script, text, expected_pos=30)    # 新行为：就近消歧
    print(f"[合成复现] 无expected_pos={p0}  有expected_pos=30→{p1}")
    assert p0 == 4, p0
    assert p1 == 25, p1
    print("  ✅ 多候选时按 expected_pos 就近消歧\n")


def diagnose_legacy_mislocation():
    """诊断：旧逻辑（无位置先验）下，句32 的邻居长句会匹配到哪。

    直观展示「为什么旧 prompt 里句32 的对照片段会错位」。
    """
    script = _load_script()
    n = len(HONGJIE_SENTENCES)
    idx = 32
    print(f"[诊断] 句{idx}「县」的邻居定位对比（旧=无expected_pos / 新=带expected_pos）:")
    for j in (idx - 2, idx - 1, idx + 1, idx + 2):
        s = next((x for x in HONGJIE_SENTENCES if x["idx"] == j), None)
        if not s or not s["text"].strip():
            continue
        expected = j / n * len(script)
        p_old = match_script_position(script, s["text"])  # 旧：首个/最前匹配块
        p_new = match_script_position(script, s["text"], expected_pos=expected)  # 新
        mark = "⚠ 错位" if _snippet(script, p_old) not in _snippet(script, p_new) and p_old != p_new else "ok"
        print(f"  邻居句{j} {s['text']!r}")
        print(f"    旧 pos={p_old:>4} 片段={_snippet(script, p_old)!r}  {mark}")
        print(f"    新 pos={p_new:>4} 片段={_snippet(script, p_new)!r}")
    print()


if __name__ == "__main__":
    test_short_window_lands_on_correct_story()
    test_short_window_focused_not_overlong()
    test_position_disambiguation_picks_later_occurrence()
    diagnose_legacy_mislocation()
    print("全部通过 ✅")
