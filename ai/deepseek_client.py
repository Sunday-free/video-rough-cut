"""
deepseek_client.py — 通过 OpenAI SDK 调用 DeepSeek Chat Completions API。

替换原来的 Coze 工作流：用更便宜的 DeepSeek 直接做口播粗剪分析。
API Key / 超时等配置统一从 speech_error_detector.config 加载。

思考模式: 默认 disabled（关闭思考，加速 3-5 倍）。
  文档: https://api-docs.deepseek.com/zh-cn/guides/thinking_mode
  开启: extra_body={"thinking": {"type": "enabled"}}
  关闭: extra_body={"thinking": {"type": "disabled"}}
"""
from openai import OpenAI

from speech_error_detector.config import DEEPSEEK_API_KEY


# API Key / 超时：统一由 config 加载（环境变量 > .env > 默认值）
REQUEST_TIMEOUT = 180

# OpenAI 兼容客户端（单例复用连接池）
_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
            timeout=REQUEST_TIMEOUT,
        )
    return _client


def deepseek_chat(
    system: str,
    user: str,
    model: str,
    temperature: float = 0.0,
    enable_thinking: bool | None = None,
) -> str:
    """
    调用 DeepSeek Chat Completions，返回模型输出的文本。

    Args:
        system:          系统提示（角色 + 剪辑规则）
        user:            用户消息（待分析的数据，如 utterance 列表）
        model:           模型名
        temperature:     采样温度，0 更稳定、更确定
        enable_thinking: 是否开启思考模式。None=跟随全局设置，True/False=覆盖全局

    Returns:
        模型回复的纯文本内容（通常为 JSON 数组字符串）
    """
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "缺少 DeepSeek API Key：请设置环境变量 DEEPSEEK_API_KEY，"
            "或在 loop/deepseek_client.py 中配置默认值。"
        )

    client = _get_client()

    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        stream=False,
    )

    # 思考模式控制：优先用调用级参数，否则用全局设置
    if enable_thinking:
        kwargs["reasoning_effort"] = "high"
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    else:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        raise RuntimeError(f"DeepSeek 请求失败: {e}") from e

    try:
        return resp.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError) as e:
        raise RuntimeError(f"DeepSeek 返回结构异常: {resp}") from e
