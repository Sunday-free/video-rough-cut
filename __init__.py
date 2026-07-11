"""
speech_error_detector — 口误检测系统

包结构（按功能模块分子包）:
- pipeline.py:            主入口，串联检测→研判→循环审查→装配→字幕（保留在包根，供 run_koubo.py / SKILL.md 直接引用）
- base/:                  基础依赖（句子 IO、路径配置、音频提取、转录）
    sentence_io.py          sentences.txt 读写
    paths.py                输出目录配置（detect_dir/loop_dir/debug_dir；judge 产物并入 detect/）
    audio_extractor.py      音频提取
    volcengine_transcriber.py  火山引擎转录
- detect/:                机械检测 + LLM 研判
    detect_intra.py / detect_inter.py / detect_fragment.py   （句内/句间重复、残句）
    llm_judge.py            LLM 候选精判（确认/排除误报）
- loop/:                  Agent 循环审查 + LLM 客户端
    agent_review_loop.py    主循环编排
    agent_prompts.py       Detect/Verify prompt 构建
    agent_apply.py         删除应用与摘要
    deepseek_client.py     OpenAI 兼容 LLM 客户端
    llm_parse.py           JSON 对象/数组解析
- assemble/:              决策合并与字幕输出
    assemble.py            合并所有决策 → auto_selected.json + 报告
    subtitle_generator.py  字幕轴 → 句子列表
    annotated_subtitle.py  带标注字幕 txt
- server/:                审核服务器
    review_server.py       剪口播审核服务（静态文件 + 剪辑）

使用方法:
    python -m speech_error_detector.pipeline --base-dir <数据目录>
"""

__version__ = "1.1.0"
