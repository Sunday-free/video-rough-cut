"""
volcengine_transcriber.py — 火山引擎 Seed ASR 2.0 转录模块

对应原 skill 脚本: volcengine_transcribe.sh (已改为纯 Python 实现)
功能: 提交音频 → 异步轮询 → 返回转录结果
"""

import base64
import json
import ssl
import time
import uuid
from pathlib import Path
from datetime import datetime

import urllib.request
import urllib.error


# ============================================================
#  SSL 配置（解决代理/VPN 环境下的证书问题）
# ============================================================

_ssl_no_verify_ctx = ssl.create_default_context()
_ssl_no_verify_ctx.check_hostname = False
_ssl_no_verify_ctx.verify_mode = ssl.CERT_NONE


# ============================================================
#  API 配置
# ============================================================

VOLCENGINE_SUBMIT_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
VOLCENGINE_QUERY_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"
VOLCENGINE_RESOURCE_ID = "volc.seedasr.auc"


def get_api_key() -> str:
    """获取火山引擎 API Key：环境变量 > .env 文件"""
    key = __import__("os").environ.get("VOLCENGINE_API_KEY", "")
    if not key:
        env_paths = [
            Path(__file__).resolve().parents[2] / ".env",
            Path("/Users/wangjianmin/Desktop/auto-generate-video/.env"),
        ]
        for p in env_paths:
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("VOLCENGINE_API_KEY=") and not line.startswith("#"):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
                if key:
                    break
    return key


# ============================================================
#  ASR 误识别修正映射表（hard code）
#  格式: "ASR 给出的错误词" → "期望的正确词"
#  后续如需新增，直接在这里加一行即可。
# ============================================================
_ASR_FIX_MAP: dict[str, str] = {
    "ming": "man",   # 火山 ASR 常将「man」误识别为「ming」
}


def _fix_asr_misrecognition(query_resp: dict, fix_map: dict[str, str] | None = None) -> None:
    """通用 ASR 误识别修正：根据 fix_map 对 utterances text/words 做就地替换。

    fix_map 格式: {"错误词": "正确词", ...}，默认使用模块级 _ASR_FIX_MAP。
    匹配为整词边界（\\b），不区分大小写；词级别 words 修正时保持原大小写风格。
    """
    import re
    if fix_map is None:
        fix_map = _ASR_FIX_MAP
    if not fix_map:
        return

    # 预编译正则 + 目标映射
    patterns: list[tuple[re.Pattern, str]] = []
    for wrong, correct in fix_map.items():
        patterns.append((re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE), correct))

    for utt in query_resp.get("result", {}).get("utterances", []):
        txt = utt.get("text", "")
        if not txt:
            continue
        # 整句级别替换
        for pat, correct in patterns:
            txt = pat.sub(correct, txt)
        utt["text"] = txt

        # 词级别 words
        words = utt.get("words", [])
        for w in words:
            w_text = w.get("text", "")
            for pat, correct in patterns:
                if pat.fullmatch(w_text):
                    w["text"] = correct[0].upper() + correct[1:] if w_text and w_text[0].isupper() else correct
                    break


def transcribe(
    audio_path: Path,
    output_dir: Path,
    language: str = "zh-CN",
) -> tuple[Path, dict]:
    """
    调用火山引擎录音文件识别 2.0（Seed ASR v3 异步模式）。
    
    Args:
        audio_path: 音频文件路径 (mp3/wav/m4a/aac)
        output_dir: 输出目录
        language: 识别语言，默认 "zh-CN"（如粤语 "zh-CN" 也走普通话模型，
                  火山不支持粤语专线时可保持默认；如有粤语模型 endpoint 在此切换）
    
    Returns:
        (result_json_path, meta_dict)
        
    输出文件:
        - volcengine_result.json (ASR 原始返回)
        - volcengine_asr_meta.json (元信息: provider/model/attempts/elapsed 等)
    """
    result_json = output_dir / "volcengine_result.json"
    meta_json = output_dir / "volcengine_asr_meta.json"

    # 缓存检查
    if result_json.exists() and meta_json.exists():
        print(f"   📂 发现火山转录缓存，跳过")
        with open(meta_json, "r") as f:
            meta = json.load(f)
        return result_json, meta

    # 获取 API Key
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "未找到火山引擎 API Key。\n"
            "请设置环境变量 VOLCENGINE_API_KEY 或在 .env 中配置。"
        )

    # 读取音频并 base64 编码
    audio_bytes = audio_path.read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

    ext = audio_path.suffix.lstrip(".").lower()
    fmt = ext if ext in ("mp3", "wav", "m4a", "aac", "ogg", "opus") else "mp3"

    # 构造请求体
    request_id = uuid.uuid4().hex
    body = {
        "user": {"uid": "auto-generate-video"},
        "audio": {
            "data": audio_b64,
            "format": fmt,
            "codec": "raw",
            "rate": 16000,
            "bits": 16,
            "channel": 1,
            "language": language,
        },
        "request": {
            "model_name": "bigmodel",
            "enable_itn": True,
            "enable_punc": True,
            "show_utterances": True,
        },
    }
    body_data = json.dumps(body).encode("utf-8")

    print(f"🎤 调用火山引擎 Seed ASR 2.0 ...")
    print(f"   音频: {audio_path.name} ({len(audio_bytes) / (1024*1024):.1f} MB)")

    # 提交任务
    req = urllib.request.Request(
        VOLCENGINE_SUBMIT_URL,
        data=body_data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Api-Key": api_key,
            "X-Api-Resource-Id": VOLCENGINE_RESOURCE_ID,
            "X-Api-Request-Id": request_id,
            "X-Api-Sequence": "-1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120, context=_ssl_no_verify_ctx) as resp:
            submit_status = resp.headers.get("x-api-status-code", "")
            submit_body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"火山转录提交失败 (HTTP {e.code}): {err_body}")

    if submit_status != "20000000":
        raise RuntimeError(f"火山转录提交失败: status={submit_status}")

    print(f"   ✅ 任务已提交，等待转录完成...")

    # 轮询结果
    query_body = json.dumps({}).encode("utf-8")
    max_attempts = 180  # 最多等 6 分钟
    start_ms = int(time.time() * 1000)

    for attempt in range(1, max_attempts + 1):
        time.sleep(2)

        req = urllib.request.Request(
            VOLCENGINE_QUERY_URL,
            data=query_body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Api-Key": api_key,
                "X-Api-Resource-Id": VOLCENGINE_RESOURCE_ID,
                "X-Api-Request-Id": request_id,
                "X-Api-Sequence": "-1",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30, context=_ssl_no_verify_ctx) as resp:
                query_status = resp.headers.get("x-api-status-code", "")
                query_resp = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"查询失败 (HTTP {e.code})")

        # 检查 utterances 结果
        utterances = None
        if isinstance(query_resp.get("result"), dict):
            utterances = query_resp["result"].get("utterances")
        elif isinstance(query_resp.get("utterances"), list):
            utterances = query_resp["utterances"]

        if utterances and len(utterances) > 0:
            # Hard code: 火山 ASR 识别错误修复
            _fix_asr_misrecognition(query_resp)

            # 保存结果
            with open(result_json, "w", encoding="utf-8") as f:
                json.dump(query_resp, f, ensure_ascii=False, indent=2)

            end_ms = int(time.time() * 1000)
            meta = {
                "provider": "volcengine",
                "api": "api/v3/auc/bigmodel",
                "model": "Seed ASR 2.0",
                "language": language,
                "resource_id": VOLCENGINE_RESOURCE_ID,
                "request_id": request_id,
                "attempts": attempt,
                "elapsed_ms": end_ms - start_ms,
                "utterances": len(utterances),
                "generated_at": datetime.now().isoformat(),
            }
            with open(meta_json, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            print(f"   ✅ 转录完成: {len(utterances)} 段语音, "
                  f"耗时 {(end_ms - start_ms) // 1000}s")
            return result_json, meta

        # 进行中状态
        if query_status in ("20000001", "20000002", ""):
            if attempt % 15 == 0:
                print(f"   ⏳ 已等待 {attempt * 2}s...")
            continue

        raise RuntimeError(f"转录失败: status={query_status}")

    raise RuntimeError("转录超时（6分钟），任务未完成")
