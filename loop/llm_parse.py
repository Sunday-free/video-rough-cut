"""LLM 响应 JSON 解析工具。

统一处理 markdown 代码块、单引号、尾部逗号等脏数据，替代散落在
agent_review_loop / llm_judge 中的多份同质解析逻辑。
"""

import json
import re

_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _strip_code_block(text: str) -> str:
    text = text.strip()
    m = _BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    return text


def parse_json_object(text: str, default=None) -> dict | None:
    """解析 LLM 响应为 dict。失败时返回 default（默认 None）。"""
    t = _strip_code_block(text)
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    try:
        obj = json.loads(t.replace("'", '"'))
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    if m:
        try:
            obj = json.loads(m.group(0).replace("'", '"'))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return default


def parse_json_array(text: str, expected: int | None = None) -> list:
    """解析 LLM 响应为 list。

    - 支持 markdown 代码块 / 单引号 / 尾部逗号修复
    - 顶层为非数组对象时，包装为 [obj]
    - expected 给定且解析失败时，返回 [{"index":i,"confirmed":False,...}] 兜底
      （verify 类响应解析失败时对每一项默认驳回）

    expected=None 且解析失败时返回 []。
    """
    t = _strip_code_block(text)
    obj = None
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        pass
    if obj is None:
        cleaned = t.replace("'", '"')
        cleaned = _TRAILING_COMMA_RE.sub(r"\1", cleaned)
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            pass
    if obj is None:
        m = re.search(r"\[.*\]", t, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    if obj is None:
        return _verify_fallback(expected)
    if isinstance(obj, list):
        return obj
    # 解析为非数组对象
    if expected is not None:
        return _verify_fallback(expected)
    return [obj]


def _verify_fallback(expected: int | None) -> list:
    if expected is None:
        return []
    return [
        {"index": i, "confirmed": False, "reason": "JSON 解析失败，默认驳回"}
        for i in range(expected)
    ]
