"""
aliyun_client.py — 通过 OpenAI SDK 调用阿里云百炼（DashScope）通义千问 Chat Completions API。

这是 deepseek_client.py 的平替：函数签名与 deepseek_chat 保持一致，便于整体切换。
deepseek_client.py 代码文件仍保留（未删除），仅在各调用方改用本模块。

API Key：环境变量 DASHSCOPE_API_KEY（也兼容从项目根 / 同目录的 .env 读取 DASHSCOPE_API_KEY=...）
Base URL：默认标准兼容端点；华北2（北京）等业务空间请设置环境变量 DASHSCOPE_BASE_URL 覆盖，
          形如 https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
模型：    默认 qwen-plus，可用环境变量 ALIYUN_MODEL 或调用时 model= 覆盖（如 qwen-max / qwen3-xxx）

思考模式：enable_thinking 仅对支持思考的模型（qwen3 系列）生效；默认关闭，
          对 qwen-plus 等不支持思考的模型传 False 即可（不附加任何 thinking 参数）。
"""
from openai import OpenAI

from speech_error_detector.config import (
    DASHSCOPE_API_KEY,
)


# API Key / 端点 / 超时：统一由 config 加载（环境变量 > .env > 默认值）
ALIYUN_API_KEY = DASHSCOPE_API_KEY
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 超时（秒）
REQUEST_TIMEOUT = 180

# OpenAI 兼容客户端（单例复用连接池）
_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=ALIYUN_API_KEY,
            base_url=BASE_URL,
            timeout=REQUEST_TIMEOUT,
        )
    return _client


def aliyun_chat(
    system: str,
    user: str,
    model: str,
    temperature: float = 0.0,
    enable_thinking: bool | None = None,
) -> str:
    """
    调用通义千问 Chat Completions，返回模型输出的文本。

    签名与 deepseek_chat 对齐：
        system / user / model / temperature / enable_thinking -> 纯文本 content

    Args:
        system:          系统提示（角色 + 规则）
        user:            用户消息（待分析数据）
        model:           模型名（默认 qwen-plus）
        temperature:     采样温度，0 更稳定
        enable_thinking: 是否开启思考（仅 qwen3 系列生效）；默认关闭

    Returns:
        模型回复的纯文本内容（通常为 JSON 数组/对象字符串）
    """
    if not ALIYUN_API_KEY:
        raise RuntimeError(
            "缺少阿里云 DashScope API Key：请设置环境变量 DASHSCOPE_API_KEY，"
            "或在 .env 中配置 DASHSCOPE_API_KEY=..."
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

    # 思考模式：仅开启时附加（支持 qwen3 系列）；关闭/默认不附加，兼容 qwen-plus
    if enable_thinking:
        kwargs["extra_body"] = {"enable_thinking": True}
    else:
        kwargs["extra_body"] = {"enable_thinking": False}

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        raise RuntimeError(f"阿里云 DashScope 请求失败: {e}") from e

    try:
        return resp.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError) as e:
        raise RuntimeError(f"阿里云 DashScope 返回结构异常: {resp}") from e
