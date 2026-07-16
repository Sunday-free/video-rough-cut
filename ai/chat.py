"""
chat.py — 统一 LLM 调用入口（按模型名自动路由到对应 client）。

路由规则：
  - 模型名以 "deepseek" 开头（如 "deepseek-v4-pro"、"deepseek-chat"）
    → 走 DeepSeek client（speech_error_detector.ai.deepseek_client）
  - 其余（如 "qwen3.7-plus"、"qwen3.7-max"、qwen-plus/qwen-max、qwen3 系列）
    → 走阿里云百炼 / 通义千问 client（speech_error_detector.ai.aliyun_client）

默认模型：qwen-max（可用环境变量 LLM_MODEL 覆盖）。

外部统一调用本模块的 chat()，不要在业务代码里直接 import 两个 client，
这样切换 / 新增模型只改这里即可。
"""
import os

from speech_error_detector.ai.deepseek_client import deepseek_chat
from speech_error_detector.ai.aliyun_client import aliyun_chat

# 默认模型（环境变量 LLM_MODEL 可覆盖；缺省 qwen3.7-max）
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "qwen3.7-max")


def _resolve_backend(model: str):
    """按模型名前缀选择后端 client。"""
    if model.startswith("deepseek"):
        return deepseek_chat
    # 其余默认走阿里百炼（通义千问）：qwen-plus / qwen-max / qwen3-xxx 等
    return aliyun_chat


def chat(
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    enable_thinking: bool | None = None,
) -> str:
    """
    统一 LLM 调用入口，签名与 deepseek_chat / aliyun_chat 完全一致。

    按 model 名称自动路由到对应后端 client：
      - "deepseek-*"   → DeepSeek
      - "qwen-*" 等    → 阿里云百炼（通义千问）

    Args:
        system:          系统提示
        user:            用户消息
        model:           模型名（默认 qwen-max）
        temperature:     采样温度，0 更稳定
        enable_thinking: 是否开启思考（按后端生效：DeepSeek 用 thinking 开关、
                         通义 qwen3 系列用 enable_thinking 开关；关闭/默认对
                         qwen-plus 等不附加额外参数）

    Returns:
        模型回复的纯文本内容
    """
    backend = _resolve_backend(model)
    return backend(
        system,
        user,
        model=model,
        temperature=temperature,
        enable_thinking=enable_thinking,
    )
