"""
test_llm_judge.py — 直接用 detect_judge_data/ 里已拼好的 judge_prompt_*.txt 调真实 LLM，
验证 agent 返回值（不走 detect → build_judge_prompt 全流程，直接把文件内容当 user prompt 喂给 agent）。

文件名约定: judge_prompt_{detector}[_round{N}].txt  → detector ∈ {inter, intra, fragment}
对应 system prompt 取自 llm_judge.PROMPTS[detector]（round 后缀仅用于区分轮次，不影响 system prompt）。

每个文件可能含**多条** user prompt，用分隔线 `==== 检测 #N ====` 隔开；本脚本会自动切分，
逐条独立调用 LLM（与 run_llm_judge 的逐条调用语义一致，每条返回单个 JSON 对象）。

运行参数已写死在下方常量区（见 RUN_*），要改直接改那里：
  python3 -m speech_error_detector.test.test_llm_judge

默认目录: speech_error_detector/test/detect_judge_data
"""
import json
import re
from pathlib import Path

from speech_error_detector.detect_repeat.llm_judge import PROMPTS
from speech_error_detector.ai.chat import chat
from speech_error_detector.config import DEFAULT_MODEL
from speech_error_detector.ai.llm_parse import parse_json_object, parse_json_array

DEFAULT_DIR = Path(__file__).resolve().parent / "detect_judge_data"
PREFIX = "judge_prompt_"

# 分隔线约定（与 llm_judge.run_llm_judge 合并落盘格式一致）:
#   "\n\n" + "="*50 + "  检测 #N  " + "="*50 + "\n\n"
_SPLIT_RE = re.compile(r"\n*={2,}\s*检测 #\d+\s*={2,}\n*")

# ---- 写死的运行参数（改这里即可）----
RUN_DIR = DEFAULT_DIR                      # 含 judge_prompt_*.txt 的目录
RUN_OUT_DIR = Path(__file__).resolve().parent / "test_output_detect_judge"  # 结果输出目录
RUN_MODEL = DEFAULT_MODEL
RUN_TEMPERATURE = 0.1
RUN_THINKING = None                        # enable_thinking: True / False，None=不传（由模型决定）
RUN_LIST_ONLY = False                      # True=只列出将测文件/prompt 数与 system prompt，不调 LLM


def split_prompts(text: str) -> list[str]:
    """按分隔线把文件内容切分为多个独立 user prompt。无分隔线则返回单条。"""
    parts = _SPLIT_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def parse_single(resp: str) -> dict:
    """解析单条 LLM 响应：优先单个 JSON 对象；兼容历史数组格式（取 [0]）。"""
    obj = parse_json_object(resp)
    if isinstance(obj, dict):
        return obj
    arr = parse_json_array(resp)
    if isinstance(arr, list) and arr:
        first = arr[0]
        return first if isinstance(first, dict) else {}
    return {}


def resolve_detector(stem_suffix: str) -> str:
    """judge_prompt_fragment_round1 → fragment（去掉 _roundN 后缀）。"""
    return re.sub(r"_round\d+$", "", stem_suffix)


def main():
    d = Path(str(RUN_DIR))
    files = sorted(d.glob(f"{PREFIX}*.txt"))
    if not files:
        print(f"[!] 在 {d} 下未找到 {PREFIX}*.txt")
        return

    plan = []
    for fp in files:
        suffix = fp.stem[len(PREFIX):]
        det = resolve_detector(suffix)
        system = PROMPTS.get(det)
        plan.append((fp, det, system, suffix))

    if RUN_LIST_ONLY:
        print(f"目录: {d}")
        for fp, det, system, suffix in plan:
            prompts = split_prompts(fp.read_text(encoding="utf-8"))
            ok = "✓ 有对应 system prompt" if system else "✗ 无对应 system prompt(将回退 inter)"
            print(f"  {fp.name}  → detector={det}  prompts={len(prompts)}  {ok}")
        return

    thinking = RUN_THINKING
    out_dir = RUN_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for fp, det, system, suffix in plan:
        raw = fp.read_text(encoding="utf-8")
        prompts = split_prompts(raw)
        system_prompt = system if system else PROMPTS["inter"]
        print("\n" + "=" * 70)
        print(f"[{det}] 文件: {fp.name}  prompts={len(prompts)}  system={'有' if system else '无→回退 inter'}")
        print("=" * 70)

        results = []
        for k, user in enumerate(prompts):
            print(f"\n--- prompt #{k} (共 {len(prompts)} 条) ---")
            try:
                resp = chat(
                    system=system_prompt,
                    user=user,
                    model=RUN_MODEL,
                    temperature=RUN_TEMPERATURE,
                    enable_thinking=thinking,
                )
            except Exception as e:
                print(f"[!] LLM 调用失败: {e}")
                results.append({"_error": str(e)})
                continue

            print("----- RAW RESPONSE -----")
            print(resp)

            parsed = parse_single(resp)
            results.append(parsed)
            print(f"💾 prompt #{k} 解析为对象: {json.dumps(parsed, ensure_ascii=False)[:200]}")

        # 写出：合并为 list（每条一个对象），文件名带 suffix 以免不同轮次互相覆盖
        out_name = f"judge_response_{suffix}.json"
        (out_dir / out_name).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n💾 全部 {len(results)} 条已保存到 {out_dir / out_name}")


if __name__ == "__main__":
    main()
