"""speech_error_detector.detect — 子模块包"""

# 中文数字 → 阿拉伯数字字符级映射（用于归一化比较）
CN_DIGIT_MAP = {
    '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
    '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
}


def normalize_numerals(t: str) -> str:
    """将中文数字字符统一转为阿拉伯数字，使 八==8 在比较时视为相同。"""
    return "".join(CN_DIGIT_MAP.get(ch, ch) for ch in t)
