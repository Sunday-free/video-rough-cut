"""
paths.py — 2_分析 目录下的子目录布局（集中管理，避免产物平铺混乱）

布局（analysis_dir = .../剪口播/2_分析）：
  sentences.txt / readable.txt / original_script.txt   ← 核心 I/O（顶层）
  auto_selected.json / updated_sentences.txt /
  口误标注字幕.txt / 口误分析.md                         ← 最终交付物（顶层）
  decisions.json                                        ← 合并删除索引（sorted 整数数组，顶层）
  detect/   ← 机械检测 + LLM 研判 + 跨轮去重（detect_round_*.json / detect_history.json / judge_decisions_*.json）
  loop/     ← Agent 循环审查（review_loop_decisions.json / loop_round*_*.txt）
  debug/    ← 调试备份（sentences_from_*_judged.txt）

各 helper 返回对应子目录 Path，并在首次访问时自动 mkdir（幂等）。
"""

from pathlib import Path


def _sub(analysis_dir: Path, name: str) -> Path:
    p = Path(analysis_dir) / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def detect_dir(analysis_dir: Path) -> Path:
    return _sub(analysis_dir, "detect")


def loop_dir(analysis_dir: Path) -> Path:
    return _sub(analysis_dir, "loop")


def debug_dir(analysis_dir: Path) -> Path:
    return _sub(analysis_dir, "debug")


def make_subdirs(analysis_dir: Path) -> None:
    """一次性创建全部子目录（pipeline 启动清空前调用）。"""
    detect_dir(analysis_dir)
    loop_dir(analysis_dir)
    debug_dir(analysis_dir)
