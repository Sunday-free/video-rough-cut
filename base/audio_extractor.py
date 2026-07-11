"""
audio_extractor.py — 从视频中提取音频 & 获取视频信息（ffmpeg / ffprobe）
"""

import json
import os
import subprocess


def extract_audio(video_path: str, output_dir: str, suffix: str = "0") -> str:
    """
    从视频中提取 16kHz 单声道 MP3（用于 Coze ASR）。

    Returns:
        audio_path: 生成的 .mp3 文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"audio_{suffix}.mp3")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "64k",
        output_path, "-y",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def get_video_info(video_path: str) -> dict:
    """
    用 ffprobe 获取视频信息。

    Returns:
        {"duration": float, "width": int, "height": int}
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    duration = float(data["format"]["duration"])

    # 找到视频流
    width, height = 1080, 1920
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            width = int(stream.get("width", 1080))
            height = int(stream.get("height", 1920))
            break

    return {"duration": duration, "width": width, "height": height}


def get_videos_info(video_paths: list[str]) -> list[dict]:
    """获取多个视频的信息列表。"""
    return [get_video_info(p) for p in video_paths]
