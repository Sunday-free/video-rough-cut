"""
config.py — speech_error_detector 统一配置中心。

集中加载所有环境变量（含 .env 文件），业务代码统一从这里取配置，
不要再到处写 os.environ.get(...) / getenv(...)。

加载优先级：进程环境变量  >  .env 文件  >  内置默认值。

.env 候选路径（按文件出现的先后顺序扫描，先扫到的优先；均不覆盖已存在的进程环境变量）：
  - <项目根>/speech_error_detector/.env
  - <项目根>/output/.env
  - /Users/wangjianmin/Desktop/auto-generate-video/.env   （历史项目兜底，保留兼容）
"""
import os
from enum import Enum
from pathlib import Path

# ---------------------------------------------------------------------------
# .env 候选路径
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent          # speech_error_detector/
_ENV_CANDIDATES = [
    _PROJECT_ROOT / ".env",                              # speech_error_detector/.env
    _PROJECT_ROOT.parent / ".env",                       # output/.env
]


# 进程环境变量优先，.env 仅在未设置时补全
_env_cache: dict[str, str] = {}


def _load_env_files() -> None:
    """把各 .env 文件里的键值装入 _env_cache（不覆盖进程环境变量）。"""
    for p in _ENV_CANDIDATES:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k not in os.environ and k not in _env_cache:
                _env_cache[k] = v


_load_env_files()


def getenv(name: str, default: str = "") -> str:
    """取配置：进程环境变量  >  .env  >  默认值。"""
    if name in os.environ:
        return os.environ[name]
    return _env_cache.get(name, default)


# ---------------------------------------------------------------------------
# 支持的模型清单
# ---------------------------------------------------------------------------
class SUPPORTED_MODELS(str, Enum):
    """当前支持的所有 LLM 模型（继承自 str，其值即模型名字符串，可直接当模型名使用）"""

    # DeepSeek client（deepseek_client.deepseek_chat）
    DEEPSEEK_V4_PRO = "deepseek-v4-pro"
    DEEPSEEK_V4_FLASH = "deepseek-v4-flash"
    DEEPSEEK_CHAT = "deepseek-chat"
    DEEPSEEK_REASONER = "deepseek-reasoner"

    # 阿里云百炼 / 通义千问 client（aliyun_client.aliyun_chat）
    QWEN_3_7_MAX = "qwen3.7-max"
    QWEN_3_7_PLUS = "qwen3.7-plus"
    QWEN_MAX = "qwen-max"
    QWEN_PLUS = "qwen-plus"
    QWEN_TURBO = "qwen-turbo"


# 全部模型枚举成员的扁平列表（便于遍历 / 校验）
ALL_SUPPORTED_MODELS = list(SUPPORTED_MODELS)

# ---------------------------------------------------------------------------
# LLM（默认模型：环境变量 LLM_MODEL 可覆盖）
# ---------------------------------------------------------------------------
DEFAULT_MODEL = getenv("LLM_MODEL", SUPPORTED_MODELS.QWEN_3_7_MAX)

# ---------------------------------------------------------------------------
# DeepSeek
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY = getenv("DEEPSEEK_API_KEY", "")

# ---------------------------------------------------------------------------
# 阿里云百炼 / 通义千问
# ---------------------------------------------------------------------------
DASHSCOPE_API_KEY = getenv("DASHSCOPE_API_KEY", "")

# ---------------------------------------------------------------------------
# 火山引擎 Seed ASR
# ---------------------------------------------------------------------------
VOLCENGINE_API_KEY = getenv("VOLCENGINE_API_KEY", "")

# ---------------------------------------------------------------------------
# 视频
# ---------------------------------------------------------------------------
VDUR = float(getenv("VDUR", "0"))


# ===========================================================================
# A. 项目目录布局（顶层目录名 / 文件名常量，单一来源）
# ===========================================================================
TRANSCRIPT_DIR = "1_转录"
ANALYSIS_DIR = "2_分析"
REVIEW_DIR = "3_审核"

WORDS_JSON = "subtitles_words.json"
VOLC_RESULT_JSON = "volcengine_result.json"
VOLC_ASR_META_JSON = "volcengine_asr_meta.json"
ORIGINAL_SCRIPT_FILE = "original_script.txt"
SENTENCES_FILE = "sentences.txt"
SENTENCES_ORIGIN_FILE = "sentences_origin.txt"
READABLE_FILE = "readable.txt"
AUDIO_FILE = "audio.mp3"


# ===========================================================================
# B. 默认模型（消除各处字面量不一致）
# ===========================================================================
# 机械检测(detect_repeat)默认模型。run_speech_pipeline.py 用 QWEN_3_7_PLUS（生产实测跑通），
# 取代 speech_pipeline 原字面量 "deepseek-v4-pro" 与 run 脚本的不一致。
DEFAULT_DETECT_REPEAT_MODEL = SUPPORTED_MODELS.QWEN_3_7_PLUS


# ===========================================================================
# C. 运行参数默认（统一 run_pipeline 与 run_speech_pipeline 的冲突取值）
# ===========================================================================
# 采用 run_speech_pipeline.py 的生产实测值。
SILENCE_THRESH = 0.9
SILENCE_KEEP_DURATION = 0.5
SILENCE_GAP_THRESHOLD = 0.5      # 断句静音阈值（秒），subtitle_generator 句子切分用
ENABLE_DEEPSEEK_THINKING = False
SPLIT_MODE = "hybrid"
USE_ORIGINAL_SCRIPT = True
MAX_DET_ROUNDS = 5                 # 机械检测多轮收敛上限（与 detect_loop 对齐）
MAX_LOOP_ROUNDS = 3                # Agent 循环审查「请求轮数」默认
AGENT_MAX_ROUNDS = 5               # Agent 循环硬上限（防无限循环，语义独立，不混 MAX_LOOP_ROUNDS）
VIDEO_DURATION = 0.0
LANGUAGE = "zh-CN"


# ===========================================================================
# D. 算法参数（共享阈值，单一来源）
# ===========================================================================
CONTEXT_RADIUS = 3
MIN_RATIO = 0.4                                              # script_window 模糊匹配相似度门槛
MIN_SCRIPT_DYN_MAX = 60                                      # 原稿窗口动态上限下限（dyn_max = max(60, 最长句*2)）
INTRA_MAX_GAP_DIGIT = 8                                      # detect_intra：数字相邻最大间隔
INTRA_MAX_GAP_OTHER = 3                                      # detect_intra：其余相邻最大间隔
PARTIAL_MIN_OVERLAP = 10                                     # detect_partial：部分重叠最小长度
DEFAULT_TEMPERATURE = 0.0                                    # 各 client / Agent 默认温度
LLM_JUDGE_TEMPERATURE = 0.1                                  # llm_judge 研判温度
VOLC_MAX_POLL_ATTEMPTS = 180                                 # 火山 ASR 轮询上限
