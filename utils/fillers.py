"""统一语气词 / 虚词 / 动词字符集 —— 全工程唯一来源。

历史上 MODAL_CHARS / _FILLERS / _FILLER / _SAFETY_FILLER / _VERB_CHARS 在
detect_intra / detect_inter / mechanical_seed / review_loop_specialized /
agent_prompts 等多处被**重复手写且口径互不一致**。本模块为唯一事实来源，
所有调用方一律从此导入，禁止再手写字符集。

集合说明
--------
MODAL_CHARS    : 最全语气词/叹词。合并了历史上所有副本（detect_intra.MODAL_CHARS、
                 detect_inter._FILLERS、mechanical_seed._FILLER、前端 JS 正则）的全部字符，
                 去重后得到最齐全版本。
FUNCTION_CHARS : 虚词（非语气词）。模糊匹配时一并剥除以比对核心实义。
VERB_CHARS     : 动词字。判断短句是否含谓语；统一自原 mechanical_seed._VERB_CHARS
                 （刻意不含「能」，避免「人工智能」等名词被误判为动词）。
FILLER_CHARS   : MODAL_CHARS | FUNCTION_CHARS。剥除语气词+虚词后的核心文本，用于
                 子串 / 前缀比对。
MODAL_LIST_ZH  : 由 MODAL_CHARS 派生（"、" 连接）的中文列举串，专供 LLM prompt 展示。
                 本身不是独立来源——改 MODAL_CHARS 即同步更新，禁止手抄。
"""

# 最全语气词 / 叹词 —— 单一有序来源（覆盖所有历史版本，去重）。
# 检测用的集合(MODAL_CHARS)与 prompt 展示串(MODAL_LIST_ZH)均由此派生，
# 改这一个字符串即全工程同步，禁止再另写一份。
# 前三段为普通话，末段为粤语（句末语气词 / 叹词）。
_MODAL_CHARS_STR = (
    "啊呢吧嘛呀哇呐嘞咯哦呃嗯啦哈哟昂咦喂哼嗬嘘咂噻呗啧喏呸嗷嘢"
    + "额唔哎唉诶嗨嗐噢喔吖嗳撒咧啰喽吔署嘶"
    + "㗎喇咩啩㖞啫咋添晒噃啵喎嗱嘩"
)
MODAL_CHARS = set(_MODAL_CHARS_STR)

# 虚词（非语气词，模糊匹配时一并剥除以比对核心实义）
# 含粤语虚词：嘅(的) 咗(了) 咁(这么) 啲(些) 乜(什么) 嘢(东西) 噉(这样) 喺(在) 俾(给) 同(和)
FUNCTION_CHARS = set("那把给了的么着过哪吗在嘅咗咁啲乜嘢噉喺俾同")

# 动词字（判断短句是否含谓语；统一自原 mechanical_seed._VERB_CHARS）
# 注意：刻意不含「能」，否则「人工智能」等名词会被误判为含动词。
# 含粤语动词：係(是) 喺(在) 嚟(来) 講(说) 睇(看) 食(吃) 飲(喝) 行(走) 攞(拿) 搵(找) 瞓(睡) 企(站) 郁(动) 畀(给)
VERB_CHARS = set(
    "是来有在让给做说要去到成变打看想知觉得把被叫应会算觉上下发回走跑飞"
    + "涨跌买开合写读听吃喝拿带推拉提收放存进出生死活改修建选排留保"
    + "係喺嚟講睇食飲行攞搵瞓企郁畀"
)

# 语气词 + 虚词（剥除后用于子串 / 前缀比对）
FILLER_CHARS = MODAL_CHARS | FUNCTION_CHARS

# 给 LLM prompt 用的中文列举：由唯一来源 _MODAL_CHARS_STR 派生（"、" 连接），
# 保留原手写的好读顺序，且避免再手抄一份导致漂移。
MODAL_LIST_ZH = "、".join(_MODAL_CHARS_STR)

# 给 prompt 用的「单字叠读」举例（由 MODAL_CHARS 派生，仅用于说明
# 「语气词重复也正常」；不参与检测，故取少量示例即可，且保证都来自 MODAL_CHARS）
_MODAL_REPEAT_SAMPLES = [c for c in "啊呢呃哟啦哈" if c in MODAL_CHARS][:3]
MODAL_LIST_ZH_REPEAT = "、".join(f"{c}{c}" for c in _MODAL_REPEAT_SAMPLES)
