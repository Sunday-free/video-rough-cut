"""speech_error_detector.detect — 子模块包"""

# 中文数字 → 阿拉伯数字字符级映射（用于归一化比较）
CN_DIGIT_MAP = {
    '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
    '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
}


def normalize_numerals(t: str) -> str:
    """将中文数字字符统一转为阿拉伯数字，使 八==8 在比较时视为相同。"""
    return "".join(CN_DIGIT_MAP.get(ch, ch) for ch in t)


# ============================================================
#  增强版：中文数词（多位数）解析器
#  仅用于「原稿窗口内重复排除（忠于原稿/排比）」（sentence↔原稿窗口 比对），不影响任何
#  detector / LLM prompt。旧 normalize_numerals（单字版）完全不动。
#
#  覆盖：零~九、两/俩、十/百/千/万/亿，以及 幾/几（十幾→10几）。
#  防误伤：scale 字（十/百/千/万/亿）仅在「紧贴数词表达式」时才转换，
#          否则保留原字（如「十分」「十月」里的「十」不转，避免与「10分」假匹配）。
# ============================================================

_CN_NUM_FULL = {
    '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    '两': 2, '俩': 2,
}
_CN_SCALE_FULL = {'十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}


def _is_cn_num_start(t: str, i: int) -> bool:
    """位置 i 是否处于一个中文数词表达式的开头（决定 scale 字是否转换）。"""
    n = len(t)
    ch = t[i]
    if ch in _CN_NUM_FULL:
        return True
    if ch in _CN_SCALE_FULL:
        # scale 开头：需右邻数字/scale/幾几 才构成数词（如「十三」「十万」「十幾」）
        j = i + 1
        if j < n and t[j] in ('幾', '几'):
            return True
        if j < n and (t[j] in _CN_NUM_FULL or t[j] in _CN_SCALE_FULL):
            return True
        # 或左邻数字/scale（如「二十」「三百」）
        if i > 0 and (t[i - 1] in _CN_NUM_FULL or t[i - 1] in _CN_SCALE_FULL):
            return True
        return False
    return False


def _parse_cn_number_segment(t: str, i: int) -> tuple[str | None, int]:
    """从 i 开始解析一段连续中文数词，返回 (数字字符串或 None, 新下标)。

    数字字符串可能含『几』占位（来自 幾/几，如 十幾→"10几"）。
    若 i 处不是数词开头，返回 (None, i)。

    算法：数字累积进 current（c = c*10 + d），scale 将 current 乘 scale 累加到
          total 并重置 current。这样 十八→10+8=18、二百三十四→200+30+4=234。
    """
    n = len(t)
    total = 0
    c = 0
    has = False
    j = i
    while j < n:
        ch = t[j]
        if ch in _CN_NUM_FULL:
            c = c * 10 + _CN_NUM_FULL[ch]
            has = True
            j += 1
        elif ch in _CN_SCALE_FULL:
            scale = _CN_SCALE_FULL[ch]
            if scale >= 10000:   # 万、亿：封存当前段
                total += (c if c else 1) * scale
            else:                # 十、百、千
                total += (c if c else 1) * scale
            c = 0
            has = True
            j += 1
        elif ch in ('幾', '几'):
            # 幾 作占位符，结算此前累积 + 『几』
            val = total + c
            return (str(val) if val else "") + "几", j + 1
        else:
            break
    if not has:
        return None, i
    return str(total + c), j


def normalize_numerals_full(t: str) -> str:
    """增强版数词归一化：解析多位数中文数词为阿拉伯数字。

    例：十八→18、二十→20、两万→20000、三百→300、十三→13、十幾→10几、8块→8块。
    防误伤：「十分」「十月」里的『十』不转（保留原字）。
    仅用于原稿忠诚度比对，不影响 detector 与 LLM prompt。
    """
    if not t:
        return t
    res: list[str] = []
    i = 0
    n = len(t)
    while i < n:
        if _is_cn_num_start(t, i):
            seg, i = _parse_cn_number_segment(t, i)
            res.append(seg)
        else:
            res.append(t[i])
            i += 1
    return "".join(res)
