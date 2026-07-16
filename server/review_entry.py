"""
review_entry.py — Review 审核环境入口。

从 pipeline.py 抽离，包含：
  - _find_video, _compute_preselect_set, _generate_review_html, _start_review_server
  - run_review (人工审核入口)
"""

import json
import shutil
import subprocess
from pathlib import Path

from speech_error_detector.assemble.subtitle_generator import SILENCE_GAP_THRESHOLD
from speech_error_detector.assemble.assemble import iter_gap_runs
from speech_error_detector.utils.paths import detect_agent_dir
from speech_error_detector.server.review_server import ReviewServer

_THIS_DIR = Path(__file__).resolve().parent  # .../speech_error_detector/server


# ============================================================
#  Review 辅助函数
# ============================================================

def _find_video(base_dir: Path) -> Path | None:
    """在 base_dir 下查找真实视频文件（排除 3_审核 与符号链接）。"""
    for p in sorted(base_dir.rglob("*.mp4")):
        if "3_审核" in p.parts or p.is_symlink():
            continue
        return p
    return None


def _compute_preselect_set(
    words: list,
    full_selected: list,
    silence_gap_threshold: float,
) -> tuple[list[int], int, int]:
    """autoselect=True 时计算预选集。

    规则：
    - 片头/片尾静音：无条件预勾；
    - 片中静音：连续 gap 合并后合计 ≥ silence_gap_threshold 才预勾；
    - 口误/重复/语气词：检测器已判定，直接预勾（人工只负责取消）；
    - 兜底：开头/结尾静音强制预选；夹在两个已选词之间的孤立静音 gap 也一并删除。

    Returns: (preselect 列表, 静音预勾数, 片中短静音跳过数)
    """
    _gap_flags = [w.get("isGap") for w in words]
    first_speech = next((i for i, g in enumerate(_gap_flags) if not g), len(words))
    last_speech = next(
        (len(words) - 1 - i for i, g in enumerate(reversed(_gap_flags)) if not g),
        -1,
    )

    # 基于连续 gap 合并，算出应预勾的静音索引集合
    gap_preselect: set[int] = set()
    for gi, gj, combined in iter_gap_runs(words):
        is_opening = gj <= first_speech
        is_closing = gi > last_speech
        if is_opening or is_closing or combined >= silence_gap_threshold:
            for k in range(gi, gj):
                gap_preselect.add(k)

    preselect_set: set[int] = set()
    skipped = 0
    sil_pre = 0
    for idx in full_selected:
        w = words[idx] if 0 <= idx < len(words) else None
        if w is None:
            # 越界 idx（如 assemble 结尾补尾的虚拟 gap）：视为片尾静音，预勾
            preselect_set.add(idx)
            sil_pre += 1
            continue
        if not w.get("isGap"):
            # 口误/重复/语气词：检测器已判定，直接预勾，人工只负责取消
            preselect_set.add(idx)
        elif idx in gap_preselect:
            # 静音（含片头/片尾 + 合并后 ≥ 阈值的片中长停顿）：预勾
            preselect_set.add(idx)
            sil_pre += 1
        else:
            skipped += 1

    # 兜底扫描：无论候选池(full_selected)是否已含头尾静音，强制把开头/结尾
    # 的静音 gap 选入预选集。这些 gap 一定落在说话内容之外，应无条件删除，
    # 不依赖上游候选池是否包含它们（避免候选池缺失时漏选结尾静音）。
    for i, e in enumerate(words):
        if e.get("isGap") and (i < first_speech or i > last_speech) \
                and i not in preselect_set:
            preselect_set.add(i)
            sil_pre += 1

    # 兜底2: 夹在两个已选(待删)词之间的静音 gap 也一并删除。
    # 否则删掉两侧说错内容后，中间会留下一段孤立停顿（cut_video.sh 仅在
    # 间隔<0.2s 时自动合并，说错内容之间的较长停顿仍会保留）。
    for i, e in enumerate(words):
        if e.get("isGap") and i not in preselect_set:
            l = i - 1
            while l >= 0 and words[l].get("isGap"):
                l -= 1
            r = i + 1
            while r < len(words) and words[r].get("isGap"):
                r += 1
            if l >= 0 and r < len(words) and l in preselect_set \
                    and r in preselect_set:
                preselect_set.add(i)
                sil_pre += 1

    # 兜底3: 句尾删词后露出的短静音，也在 full_selected 中，
    #       不论时长都必须预选。判定方式：gap 紧邻一个已删的非 gap 词。
    _full = set(full_selected)
    for idx in full_selected:
        if idx in preselect_set:
            continue
        if idx < 0 or idx >= len(words):
            continue
        if not words[idx].get("isGap"):
            continue
        left_ok = idx > 0 and not words[idx - 1].get("isGap") and (idx - 1) in _full
        right_ok = (idx + 1 < len(words)
                    and not words[idx + 1].get("isGap") and (idx + 1) in _full)
        if left_ok or right_ok:
            preselect_set.add(idx)
            sil_pre += 1
            skipped -= 1

    return sorted(preselect_set), sil_pre, skipped


def _parse_range(s) -> tuple[int, int] | None:
    """把 "a-b" 形式的字级 range 字符串解析成 (a, b)。"""
    try:
        a, b = str(s).split("-")
        return int(a), int(b)
    except Exception:
        return None


def _entry_indices(entry: dict) -> tuple[str | None, list[tuple[int, int]]]:
    """从一条决策条目提取 (类型, 受影响的字级索引区间列表)。

    类型归并（按前端 tab 映射，用户确认）：
      - inter_repeat             → 句间重复
      - intra_repeat             → 句内重复
      - fragment                 → 残句
      - partial（保头删尾）       → 句间重复（区间算法与 fragment 同构，但 tab 归「句间重复」）
      - misread / resay          → 残句（V3 读稿错误：残句重说，整句删）

    即：partial → 句间重复；resay → 残句。前端只需 inter_repeat / fragment 两类即可承载。
    """
    det = entry.get("detect", entry) or {}
    dec = entry.get("decision", {}) or {}
    typ = det.get("type") or det.get("dimension")

    # V3 读稿错误检测：dimension=misread，归并到独立「误读(misread)」tab。
    #   resay（残句重说）整句删 → 用整句 range。
    if typ == "misread":
        r = _parse_range(det.get("range"))
        return "misread", ([r] if r else [])

    if typ == "intra_repeat":
        dr = dec.get("delete_ranges")
        out: list[tuple[int, int]] = []
        if isinstance(dr, list):
            for pr in dr:
                if isinstance(pr, (list, tuple)) and len(pr) == 2:
                    try:
                        out.append((int(pr[0]), int(pr[1])))
                    except Exception:
                        pass
        if out:
            return typ, out
        r = _parse_range(det.get("range"))
        return typ, ([r] if r else [])

    # fragment（残句）/ partial（保头删尾）：区间算法一致。
    #   partial 按用户映射归「句间重复」tab，fragment 归「残句」tab。
    if typ in ("fragment", "partial"):
        r = _parse_range(det.get("range"))
        tail = None
        if dec.get("mode") == "keep_head" and dec.get("keep_head") and r:
            kh = dec["keep_head"]
            try:
                tail = [(int(kh[1]) + 1, r[1])]
            except Exception:
                tail = [r]
        out_ranges = tail if tail is not None else ([r] if r else [])
        return ("inter_repeat" if typ == "partial" else "fragment"), out_ranges

    if typ == "inter_repeat":
        r = _parse_range(det.get("sent_a_range")) or _parse_range(det.get("range"))
        return typ, ([r] if r else [])

    return typ, []


def _build_word_categories(analysis_dir: Path) -> dict[int, str]:
    """构建 {字级索引: 类型} 映射，类型取值 inter_repeat/intra_repeat/fragment。

    数据来源（权威、已应用的决策）：
      - detect_repeat/judge_decisions_{inter,intra,fragment}.json（status == "applied"）
      - review_loop_decisions.json（decision.confirmed == True）

    优先级：intra_repeat > fragment > inter_repeat（先标更精确/更具体的，
    同一索引已被标注则不覆盖），避免同一句被 inter 与 fragment 双重计数。
    """
    detect_dir = analysis_dir / "detect_repeat"
    cats: dict[int, str] = {}

    def mark(ranges: list[tuple[int, int]], typ: str) -> None:
        for r in ranges:
            if not r:
                continue
            a, b = r
            for i in range(a, b + 1):
                cats.setdefault(i, typ)

    # 收集所有「已应用」的决策条目
    applied: list[dict] = []
    for name in ("inter", "intra", "fragment", "partial"):
        p = detect_dir / f"judge_decisions_{name}.json"
        if not p.exists():
            p = detect_dir / f"decisions_{name}.json"  # 兼容旧格式
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                data = []
            for e in data if isinstance(data, list) else []:
                if isinstance(e, dict) and e.get("status", "applied") == "applied":
                    applied.append(e)

    loop_p = detect_agent_dir(analysis_dir) / "review_loop_decisions.json"
    if not loop_p.exists():  # 兼容旧数据（曾放在 analysis_dir 根）
        loop_p = analysis_dir / "review_loop_decisions.json"
    if loop_p.exists():
        try:
            loop_data = json.loads(loop_p.read_text(encoding="utf-8"))
        except Exception:
            loop_data = []
        for e in loop_data if isinstance(loop_data, list) else []:
            if isinstance(e, dict) and "detect" in e:
                applied.append(e)  # 包含 ✅确认/📌仅标注/❌驳回，全部在页面标注

    # 按优先级分类后逐类标注（先精确/具体，first-wins）
    # 归并映射在 _entry_indices 内完成：partial→句间重复；misread(resay)→误读。
    buckets: dict[str, list[list[tuple[int, int]]]] = {
        "intra_repeat": [], "fragment": [], "inter_repeat": [], "misread": [],
    }
    for e in applied:
        typ, ranges = _entry_indices(e)
        if typ in buckets and ranges:
            buckets[typ].append(ranges)

    for typ in ("intra_repeat", "fragment", "inter_repeat", "misread"):
        for ranges in buckets[typ]:
            mark(ranges, typ)

    return cats


def _generate_review_html(
    words_json: Path, review_auto: Path, video: Path, review_dir: Path,
    categories_json: Path | None = None,
) -> None:
    """调用 generate_review.js 生成 review.html（需要 node）。"""
    gen_script = _THIS_DIR / "generate_review.js"
    if not (gen_script.exists() and shutil.which("node")):
        print("⚠️ 未找到 node 或 generate_review.js，跳过生成 review.html")
        print("   （请手动确保 3_审核/review.html 存在）")
        return
    try:
        cmd = ["node", str(gen_script.resolve()),
               str(words_json.resolve()), str(review_auto.resolve()),
               str(video.resolve())]
        if categories_json is not None:
            cmd.append(str(categories_json.resolve()))
        subprocess.run(cmd, cwd=str(review_dir), check=True)
        print(f"  已生成: review.html（含预选 {len(json.loads(review_auto.read_text(encoding='utf-8')))} 项）")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ generate_review.js 执行失败: {e}")


def _start_review_server(
    host: str, port: int, review_dir: Path, video: Path,
    cut_script_path: Path | None, words_json: Path, silence_keep_duration: float,
) -> None:
    """启动审核服务器（端口被占则顺延到下一个空闲端口）。阻塞直到 Ctrl+C。"""
    server = None
    used_port = None
    for try_port in range(port, port + 11):
        try:
            server = ReviewServer(
                host, try_port, review_dir, str(video), cut_script_path,
                words_json=words_json,
                silence_keep_duration=silence_keep_duration,
            )
            used_port = try_port
            break
        except OSError as e:
            if e.errno == 48:  # Address already in use
                print(f"⚠️ 端口 {try_port} 被占用，尝试 {try_port + 1} ...")
                continue
            raise
    if server is None:
        print(f"❌ 端口 {port}~{port + 10} 均被占用，请先释放或改端口。")
        return
    port = used_port
    try:
        print(f"""
🎬 审核服务器已启动
📍 地址: http://localhost:{port}/review.html
📂 审核目录: {review_dir}
🫁 静音保留: {silence_keep_duration}s（句间呼吸感）
💡 网页中审核勾选 → 点击「执行剪辑」

按 Ctrl+C 停止服务
        """)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 审核服务已停止")
    finally:
        server.server_close()


# ============================================================
#  Review 入口
# ============================================================

def run_review(
    base_dir: str | Path,
    port: int = 8899,
    host: str = "0.0.0.0",
    autoselect: bool = True,
    silence_gap_threshold: float = SILENCE_GAP_THRESHOLD,
    silence_keep_duration: float = 0.1,
    video_file: str | None = None,
    cut_script: str | None = None,
    serve: bool = True,
) -> None:
    """
    Review 操作：把 pipeline 产出的 auto_selected.json 转换为人工审核环境。

    流程:
      1. 读取 2_分析/auto_selected.json（完整删除候选）。
      2. 按 autoselect 计算「预选集」：
         - autoselect=True : 只预选 SILENCE_GAP_THRESHOLD 以上的静音片段
                              （口误/重复/语气词仅展示，不预勾选，由人工决定）。
         - autoselect=False: 预选全部候选（等价于旧的自动全选）。
      3. 生成 3_审核/（auto_selected.json 预选副本 + video.mp4 符号链接 + review.html）。
      4. 启动审核服务器（review_server），人工在网页中审核并点击剪辑；
         剪辑时按 silence_keep_duration 在静音 gap 内保留呼吸感。

    Args:
        base_dir: 数据根目录（含 1_转录、2_分析）
        port:     审核服务器端口（默认 8899）
        autoselect: 仅自动预选 SILENCE_GAP_THRESHOLD 以上的静音片段
        silence_gap_threshold: 静音自动预选阈值（秒）
        silence_keep_duration: 删除静音时保留的呼吸感秒数（默认 0.1）
        video_file: 视频路径（默认自动探测 base_dir 下 .mp4）
        cut_script: cut_video.sh 路径
        serve:     True=生成后启动服务器并阻塞；False=仅准备目录
    """
    base_dir = Path(base_dir)
    analysis_dir = base_dir / "2_分析"
    review_dir = base_dir / "3_审核"
    # 每次运行前清空旧的审核目录，避免上次 run 残留的 review.html /
    # auto_selected.json / word_categories.json 干扰本轮审核结果
    if review_dir.exists():
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)

    words_json = base_dir / "1_转录" / "subtitles_words.json"
    auto_selected_src = analysis_dir / "auto_selected.json"

    print("=" * 60)
    print("  Review 操作（生成审核环境）")
    print("=" * 60)
    print(f"  数据目录: {base_dir}")
    print(f"  审核目录: {review_dir}")

    if not words_json.exists():
        print("❌ 找不到 subtitles_words.json，请先完成转写步骤。")
        return
    if not auto_selected_src.exists():
        print("❌ 找不到 2_分析/auto_selected.json，请先运行完整 pipeline 检测:")
        print("   python -m speech_error_detector.test.run_speech_pipeline   # 或修改 test/run_speech_pipeline.py 的 dirs 常量")
        return

    words = json.loads(words_json.read_text(encoding="utf-8"))
    full_selected = json.loads(auto_selected_src.read_text(encoding="utf-8"))

    # --- 计算预选集 ---
    if autoselect:
        preselect, sil_pre, skipped = _compute_preselect_set(
            words, full_selected, silence_gap_threshold,
        )
        print(f"  autoselect=True: 预选 {len(preselect)} 项 "
              f"（含口误/重复/语气词全勾 + 静音 {sil_pre} 个"
              f"（片头/片尾及合并后≥{silence_gap_threshold}s 的片中长停顿）；"
              f"{skipped} 个片中短静音(<{silence_gap_threshold}s)留待人工勾选")
    else:
        preselect = list(full_selected)
        print(f"  autoselect=False: 预选全部 {len(preselect)} 个候选")

    # 写入审核目录的预选副本（供 generate_review.js 嵌入 review.html）
    review_auto = review_dir / "auto_selected.json"
    review_auto.write_text(
        json.dumps(preselect, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  已写入: {review_auto.name} ({len(preselect)} 项)")

    # --- 构建「字级索引 → 错误类型」映射（供 review 页三类分类计数）---
    word_categories = _build_word_categories(analysis_dir)
    categories_json = review_dir / "word_categories.json"
    categories_json.write_text(
        json.dumps({str(k): v for k, v in sorted(word_categories.items())},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    _cat_stat = {t: sum(1 for v in word_categories.values() if v == t)
                 for t in ("inter_repeat", "intra_repeat", "fragment", "misread")}
    print(f"  已写入: {categories_json.name} "
          f"（句间重复 {_cat_stat['inter_repeat']} / "
          f"句内重复 {_cat_stat['intra_repeat']} / "
          f"残句 {_cat_stat['fragment']} / "
          f"误读 {_cat_stat['misread']} 词）")

    # --- 视频文件 ---
    video = Path(video_file) if video_file else _find_video(base_dir)
    if not video or not video.exists():
        print("❌ 未找到视频文件，请用 --video 指定。")
        return
    print(f"  视频: {video}")

    # --- 生成 review.html ---
    _generate_review_html(words_json, review_auto, video, review_dir, categories_json)

    if not serve:
        print(f"\n✅ 审核目录已准备: {review_dir}")
        print(f"   启动服务器: python -m speech_error_detector.server.review_server "
              f"{port} \"{video}\" --work-dir \"{review_dir}\" "
              f"--words-json \"{words_json}\" --silence-keep-duration {silence_keep_duration}")
        return

    # --- 启动审核服务器 ---
    cut_script_path = Path(cut_script) if cut_script else None
    _start_review_server(
        host, port, review_dir, video, cut_script_path, words_json, silence_keep_duration,
    )
