#!/usr/bin/env python3
"""
review_server.py — 剪口播审核服务器（review_server.js 的 Python 移植版）

功能（与原 Node 版一致）:
  1. 提供静态文件服务（review.html, video.mp4），支持 Range 请求（视频拖动播放）
  2. POST /api/save-selection  — 持久化人工选中状态到 saved_selection.json
  3. GET  /api/load-selection  — 读取已保存的选中状态
  4. POST /api/cut            — 接收删除片段列表，执行剪辑并写 cut_done.json

剪辑策略:
  - 优先调用 剪口播/scripts/cut_video.sh（如有）
  - 否则用内置 FFmpeg 精确剪辑（filter_complex + acrossfade 消除接缝）
  - filter_complex 失败时回退到「分段切割 + concat demuxer 拼接」

用法:
  python -m speech_error_detector.server.review_server [port] [video_file]
  python -m speech_error_detector.server.review_server 8899 "/path/红姐.mp4" \
      --work-dir "/path/.../3_审核"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

# ── 项目路径：保证脚本既能 `python review_server.py` 也能 `python -m` 运行 ──
_THIS_DIR = Path(__file__).resolve().parent          # .../speech_error_detector
_PROJECT_ROOT = _THIS_DIR.parents[1]                 # .../output
_DEFAULT_CUT_SCRIPT = _PROJECT_ROOT / "剪口播" / "scripts" / "cut_video.sh"

MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
}


# ============================================================
#  FFmpeg / FFprobe 封装
# ============================================================
def ffprobe_duration(path: str) -> float:
    """用 ffprobe 读取媒体时长（秒），失败返回 0.0。"""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", f"file:{path}"],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
        return float(out) if out else 0.0
    except (subprocess.SubprocessError, ValueError, OSError):
        return 0.0


def _cached_encoder() -> dict:
    """检测可用的硬件编码器（跨平台），带缓存。"""
    if hasattr(_cached_encoder, "_cache"):
        return _cached_encoder._cache  # type: ignore[attr-defined]

    platform = sys.platform
    candidates: list[tuple[str, str, str]] = []
    if platform == "darwin":
        candidates.append(("h264_videotoolbox", "-q:v 60", "VideoToolbox (macOS)"))
    elif platform == "win32":
        candidates.append(("h264_nvenc", "-preset p4 -cq 20", "NVENC (NVIDIA)"))
        candidates.append(("h264_qsv", "-global_quality 20", "QSV (Intel)"))
        candidates.append(("h264_amf", "-quality balanced", "AMF (AMD)"))
    else:  # Linux
        candidates.append(("h264_nvenc", "-preset p4 -cq 20", "NVENC (NVIDIA)"))
        candidates.append(("h264_vaapi", "-qp 20", "VAAPI (Linux)"))

    # 软件编码兜底
    candidates.append(("libx264", "-preset fast -crf 18", "x264 (软件)"))

    probe_out = ""
    try:
        probe_out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        probe_out = ""

    chosen = candidates[-1]  # 默认软件编码
    for name, args, label in candidates:
        if name in probe_out:
            chosen = (name, args, label)
            print(f"🎯 检测到编码器: {label}")
            break

    result = {"name": chosen[0], "args": chosen[1], "label": chosen[2]}
    _cached_encoder._cache = result  # type: ignore[attr-defined]
    return result


def _build_keep_segments(delete_list: list[dict], duration: float,
                         audio_offset: float) -> tuple[list[dict], list[dict]]:
    """根据删除片段计算合并后的删除段与保留段。"""
    MERGE_GAP = 0.2  # 间隙 < 200ms 的相邻删除段合并，避免碎片

    expanded = sorted(
        (
            {
                "start": max(0.0, seg["start"] - audio_offset),
                "end": min(duration, seg["end"] - audio_offset),
            }
            for seg in delete_list
            if "start" in seg and "end" in seg
        ),
        key=lambda s: s["start"],
    )

    merged: list[dict] = []
    for seg in expanded:
        if not merged or seg["start"] > merged[-1]["end"] + MERGE_GAP:
            merged.append(dict(seg))
        else:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])

    keep: list[dict] = []
    cursor = 0.0
    for del_seg in merged:
        if del_seg["start"] > cursor:
            keep.append({"start": cursor, "end": del_seg["start"]})
        cursor = del_seg["end"]
    if cursor < duration:
        keep.append({"start": cursor, "end": duration})

    return merged, keep


def execute_ffmpeg_cut(input_path: str, delete_list: list[dict],
                       output_path: str, work_dir: Path) -> None:
    """内置 FFmpeg 精确剪辑（filter_complex + acrossfade 消除接缝）。"""
    # 优化参数（与原版一致；BUFFER 仅作参考，精确边界不扩展）
    BUFFER_MS = 120      # 删除范围前后各扩展 120ms（默认未启用）
    CROSSFADE_MS = 30    # 音频淡入淡出 30ms
    print(f"⚙️  优化参数: 扩展范围={BUFFER_MS}ms, 音频crossfade={CROSSFADE_MS}ms")

    # 检测音频偏移量（MP3 编码引入的延迟）
    audio_offset = 0.0
    for cand in (work_dir.parent / "1_转录" / "audio.mp3",
                 work_dir / "1_转录" / "audio.mp3"):
        if cand.exists():
            try:
                off = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=start_time",
                     "-of", "csv=p=0", str(cand)],
                    capture_output=True, text=True, timeout=30,
                ).stdout.strip()
                audio_offset = float(off) if off else 0.0
                if audio_offset > 0:
                    print(f"🔧 检测到音频偏移: {audio_offset:.3f}s，自动补偿")
                break
            except (subprocess.SubprocessError, ValueError, OSError):
                pass

    duration = ffprobe_duration(input_path)
    crossfade_sec = CROSSFADE_MS / 1000.0

    merged, keep = _build_keep_segments(delete_list, duration, audio_offset)

    print(f"保留 {len(keep)} 个片段，删除 {len(merged)} 个片段")
    if not keep:
        raise RuntimeError("没有可保留的片段（删除列表覆盖整段视频）")

    # 生成 filter_complex
    filters: list[str] = []
    vconcat = ""
    for i, seg in enumerate(keep):
        s = f"{seg['start']:.3f}"
        e = f"{seg['end']:.3f}"
        filters.append(f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS[v{i}]")
        filters.append(f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[a{i}]")
        vconcat += f"[v{i}]"
    filters.append(f"{vconcat}concat=n={len(keep)}:v=1:a=0[outv]")

    if len(keep) == 1:
        filters.append("[a0]anull[outa]")
    else:
        current = "a0"
        for i in range(1, len(keep)):
            nxt = f"a{i}"
            out_lbl = "outa" if i == len(keep) - 1 else f"amid{i}"
            filters.append(
                f"[{current}][{nxt}]acrossfade=d={crossfade_sec:.3f}:c1=tri:c2=tri[{out_lbl}]"
            )
            current = out_lbl

    filter_complex = ";".join(filters)
    enc = _cached_encoder()
    print(f"✂️  执行 FFmpeg 精确剪辑（{enc['label']}）...")

    cmd = [
        "ffmpeg", "-y", "-i", f"file:{input_path}",
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", enc["name"], *enc["args"].split(),
        "-c:a", "aac", "-b:a", "192k",
        f"file:{output_path}",
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
    except subprocess.CalledProcessError as err:
        print("⚠️  FFmpeg filter_complex 失败，尝试分段方案...")
        print(err.stderr[-2000:] if err.stderr else "")
        execute_ffmpeg_cut_fallback(input_path, keep, output_path)
    except subprocess.SubprocessError as err:
        raise RuntimeError(f"ffmpeg 执行失败: {err}") from err


def execute_ffmpeg_cut_fallback(input_path: str, keep_segments: list[dict],
                                output_path: str) -> None:
    """备用方案：分段切割（-ss/-t）+ concat demuxer 无损拼接。"""
    tmp_dir = Path(tempfile.mkdtemp(prefix="tmp_cut_"))
    try:
        enc = _cached_encoder()
        part_files: list[Path] = []
        for i, seg in enumerate(keep_segments):
            part = tmp_dir / f"part{i:04d}.mp4"
            seg_dur = seg["end"] - seg["start"]
            cmd = [
                "ffmpeg", "-y", "-ss", f"{seg['start']:.3f}", "-i", f"file:{input_path}",
                "-t", f"{seg_dur:.3f}",
                "-c:v", enc["name"], *enc["args"].split(),
                "-c:a", "aac", "-b:a", "128k", "-avoid_negative_ts", "make_zero",
                str(part),
            ]
            print(f"切割片段 {i + 1}/{len(keep_segments)}: "
                  f"{seg['start']:.2f}s - {seg['end']:.2f}s")
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
            part_files.append(part)

        list_file = tmp_dir / "list.txt"
        list_file.write_text(
            "\n".join(f"file '{p.resolve()}'" for p in part_files), encoding="utf-8"
        )
        concat_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file), "-c", "copy", f"file:{output_path}",
        ]
        print("合并片段...")
        subprocess.run(concat_cmd, capture_output=True, text=True, check=True, timeout=600)
        print(f"✅ 输出: {output_path}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================
#  HTTP 请求处理器
# ============================================================
class ReviewHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: "ReviewServer"  # 类型提示，便于访问配置

    # ── 工具 ──
    def _send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self._send_cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or "0")
        return self.rfile.read(length) if length > 0 else b""

    # ── HTTP 方法路由 ──
    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(200)
        self._send_cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/load-selection":
            self._handle_load_selection()
            return
        self._handle_static(path)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        body = self._read_body()
        if path == "/api/save-selection":
            self._handle_save_selection(body)
        elif path == "/api/cut":
            self._handle_cut(body)
        else:
            self.send_error(404, "Not Found")

    # ── API: 保存选中状态 ──
    def _handle_save_selection(self, body: bytes) -> None:
        try:
            (self.server.work_dir / "saved_selection.json").write_bytes(body)
            self._send_json(200, {"success": True})
        except OSError as e:
            self._send_json(500, {"error": str(e)})

    # ── API: 读取选中状态 ──
    def _handle_load_selection(self) -> None:
        f = self.server.work_dir / "saved_selection.json"
        if f.exists():
            data = f.read_bytes()
            self.send_response(200)
            self._send_cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._send_json(404, {"error": "no saved selection"})

    # ── API: 执行剪辑 ──
    def _handle_cut(self, body: bytes) -> None:
        try:
            delete_list = json.loads(body.decode("utf-8"))
            if not isinstance(delete_list, list):
                raise ValueError("删除列表必须是数组")

            # ── 静音保留：删除静音 gap 时向内收缩 SILENCE_KEEP_DURATION 秒，
            #    保留句间「呼吸感」（如 0.1s）。仅对落在静音 gap 内的片段生效。 ──
            keep = self.server.silence_keep_duration or 0.0
            words_json = self.server.words_json
            if keep > 0 and words_json and Path(words_json).exists():
                try:
                    _wlist = json.loads(Path(words_json).read_text(encoding="utf-8"))
                    _gaps = [(round(g["start"], 4), round(g["end"], 4))
                             for g in _wlist if g.get("isGap")]
                    if _gaps:
                        def _in_gap(s: float, e: float) -> bool:
                            mid = (s + e) / 2.0
                            return any(gs - 0.02 <= mid <= ge + 0.02
                                       for (gs, ge) in _gaps)

                        # 先把时间上连续的「纯 gap」删除片段合并成一个块，
                        # 避免长停顿被拆成多条 gap 时，SILENCE_KEEP_DURATION
                        # 被每条 gap 重复扣减（最终只保留一次呼吸感）。
                        segs = sorted(
                            ({"start": float(seg["start"]), "end": float(seg["end"])}
                             for seg in delete_list),
                            key=lambda x: (x["start"], x["end"]),
                        )
                        merged: list[dict] = []
                        i = 0
                        n = len(segs)
                        while i < n:
                            if _in_gap(segs[i]["start"], segs[i]["end"]):
                                j = i
                                while (j + 1 < n
                                       and _in_gap(segs[j + 1]["start"], segs[j + 1]["end"])
                                       and segs[j + 1]["start"] - segs[j]["end"] < 0.05):
                                    j += 1
                                block_end = segs[j]["end"]
                                # 整段停顿仅保留 keep 秒呼吸感（在尾端）
                                ne = block_end - keep
                                if ne > segs[i]["start"]:
                                    merged.append({
                                        "start": round(segs[i]["start"], 3),
                                        "end": round(ne, 3),
                                    })
                                i = j + 1
                            else:
                                merged.append({
                                    "start": round(segs[i]["start"], 3),
                                    "end": round(segs[i]["end"], 3),
                                })
                                i += 1
                        delete_list = merged
                        print(f"🫁 静音保留: 连续静音已合并，整段向内收缩 {keep}s"
                              f"（句间呼吸感）")
                except Exception as _e:  # noqa: BLE001
                    print(f"⚠️ 静音保留处理跳过: {_e}")

            work_dir = self.server.work_dir
            video_file = self.server.video_file

            # 保存删除列表
            (work_dir / "delete_segments.json").write_text(
                json.dumps(delete_list, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"📝 保存 {len(delete_list)} 个删除片段")

            base = Path(video_file).stem
            output_file = work_dir / f"{base}_cut.mp4"

            cut_script = self.server.cut_script
            if cut_script and Path(cut_script).exists():
                print("🎬 调用 cut_video.sh...")
                subprocess.run(
                    ["bash", str(cut_script),
                     str(Path(video_file).resolve()),
                     str(work_dir / "delete_segments.json"),
                     str(output_file.resolve())],
                    cwd=str(work_dir), check=True,
                )
            else:
                print("🎬 执行剪辑（内置 FFmpeg）...")
                execute_ffmpeg_cut(
                    str(Path(video_file).resolve()),
                    delete_list, str(output_file.resolve()), work_dir,
                )

            # 计算剪辑前后时长
            original_duration = ffprobe_duration(str(Path(video_file).resolve()))
            new_duration = ffprobe_duration(str(output_file.resolve()))
            deleted_duration = original_duration - new_duration
            saved_percent = (
                (deleted_duration / original_duration * 100) if original_duration > 0 else 0.0
            )

            cut_done = {
                "success": True,
                "output": str(output_file.resolve()),
                "originalDuration": round(original_duration, 2),
                "newDuration": round(new_duration, 2),
                "deletedDuration": round(deleted_duration, 2),
                "savedPercent": round(saved_percent, 1),
                "completedAt": datetime.now().isoformat(),
                "nextStep": "Agent 基于剪后视频重新转写，AI 校对后再写入最终 subtitles.srt。",
            }
            (work_dir / "cut_done.json").write_text(
                json.dumps(cut_done, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._send_json(200, {**cut_done, "message": f"剪辑完成: {output_file}"})

        except Exception as err:  # noqa: BLE001
            print(f"❌ 剪辑失败: {err}")
            self._send_json(500, {"success": False, "error": str(err)})

    # ── 静态文件服务（含 Range 支持） ──
    def _handle_static(self, path: str) -> None:
        if path in ("/", ""):
            path = "/review.html"
        rel = unquote(path).lstrip("/")

        # 允许符号链接指向工作目录外的文件（如 video.mp4 → 真实视频），
        # 但仍拦截通过 URL "../" 注入的路径穿越。
        base = self.server.work_dir.resolve()
        requested = (base / rel)
        target = requested.resolve()
        inside = str(target).startswith(str(base) + os.sep) or str(target) == str(base)
        if not inside and not requested.is_symlink():
            self.send_error(403, "Forbidden")
            return
        if not target.exists() or not target.is_file():
            self.send_error(404, "Not Found")
            return

        ext = target.suffix.lower()
        content_type = MIME_TYPES.get(ext, "application/octet-stream")
        size = target.stat().st_size

        range_header = self.headers.get("Range")
        m = re.match(r"bytes=(\d+)-(\d*)", range_header or "") if range_header else None
        if m and ext in (".mp4", ".mp3"):
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else size - 1
            end = min(end, size - 1)
            if start > end:
                self.send_error(416, "Requested Range Not Satisfiable")
                return
            length = end - start + 1
            self.send_response(206)
            self._send_cors()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            with open(target, "rb") as f:
                f.seek(start)
                self.wfile.write(f.read(length))
            return

        self.send_response(200)
        self._send_cors()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with open(target, "rb") as f:
            shutil.copyfileobj(f, self.wfile)


class ReviewServer(ThreadingHTTPServer):
    """携带配置（工作目录、视频文件、cut 脚本路径）的 HTTP 服务器。"""

    allow_reuse_address = True   # 允许 TIME_WAIT 状态下立即重绑，避免退出后短时间内重启报错

    def __init__(self, host: str, port: int, work_dir: Path,
                 video_file: str, cut_script: Path | None,
                 words_json: Path | None = None,
                 silence_keep_duration: float = 0.0):
        super().__init__((host, port), ReviewHandler)
        self.work_dir = work_dir
        self.video_file = video_file
        self.cut_script = cut_script
        self.words_json = words_json
        # 删除静音时保留的「呼吸感」时长（秒）。>0 时向内收缩静音片段边界。
        self.silence_keep_duration = silence_keep_duration


# ============================================================
#  入口
# ============================================================
def find_video_file(work_dir: Path) -> str:
    for f in sorted(work_dir.iterdir()):
        if f.suffix.lower() == ".mp4":
            return str(f)
    return "source.mp4"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="剪口播审核服务器（Python 版，等价于 review_server.js）"
    )
    parser.add_argument("port", nargs="?", default="8899", help="监听端口（默认 8899）")
    parser.add_argument("video", nargs="?", default=None,
                        help="视频文件路径（默认自动检测工作目录下的 .mp4）")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    parser.add_argument("--work-dir", default=None,
                        help="审核目录（默认当前目录，需含 review.html）")
    parser.add_argument("--cut-script", default=str(_DEFAULT_CUT_SCRIPT),
                        help="cut_video.sh 路径（默认 剪口播/scripts/cut_video.sh）")
    parser.add_argument("--words-json", default=None,
                        help="subtitles_words.json 路径；提供后删除静音时保留 SILENCE_KEEP_DURATION 呼吸感")
    parser.add_argument("--silence-keep-duration", type=float, default=0.0,
                        help="删除静音时保留的秒数（默认 0，不保留；如 0.1 句间留呼吸感）")
    args = parser.parse_args(argv)

    work_dir = Path(args.work_dir).resolve() if args.work_dir else Path.cwd().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    video_file = args.video or find_video_file(work_dir)
    cut_script = Path(args.cut_script) if args.cut_script else None
    words_json = Path(args.words_json) if args.words_json else None

    keep_info = (
        f"{args.silence_keep_duration}s" if args.silence_keep_duration > 0
        else "关闭（删除整段静音）"
    )

    port = int(args.port)
    server = ReviewServer(
        args.host, port, work_dir, video_file, cut_script,
        words_json=words_json,
        silence_keep_duration=args.silence_keep_duration,
    )

    try:
        print(f"""
🎬 审核服务器已启动（Python 版）
📍 地址: http://localhost:{port}
📂 工作目录: {work_dir}
📹 视频: {video_file}
✂️  剪辑脚本: {cut_script if (cut_script and cut_script.exists()) else '内置 FFmpeg'}
🫁 静音保留: {keep_info}

操作说明:
1. 在网页中审核选择要删除的片段
2. 点击「🎬 执行剪辑」按钮
3. 等待剪辑完成
        """)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 服务器已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
