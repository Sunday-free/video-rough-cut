"""
speech_error_detector — 口误检测系统

包结构（按功能模块分子包）:
- speech_pipeline.py:     主入口，串联检测→研判→循环审查(V3)→装配→字幕（run_pipeline 编程入口；完整运行入口见 test/run_speech_pipeline.py）
- base/:                  基础依赖（句子 IO、路径配置、音频提取、转录）
    sentence_io.py          sentences.txt 读写
    paths.py                输出目录配置（detect_dir/loop_dir；judge 产物并入 detect_repeat/）
    audio_extractor.py      音频提取
    volcengine_transcriber.py  火山引擎转录
- detect_repeat/:                机械检测 + LLM 研判
    detect_intra.py / detect_inter.py / detect_fragment.py / detect_partial.py
                          （句内重复 / 句间整句重复 / 残句整句删 / 句间部分删保头删尾）
    llm_judge.py            LLM 候选精判（确认/排除误报）
- detect_agent/:           V3 读稿错误检测循环（pipeline 唯一使用；检测+确认 双 Agent，对照原稿找"说错"）
    review_loop.py         V3 主循环
    prompts.py             V3 Detect/Confirm prompt
- llm/:                   LLM 客户端/解析（deepseek_client, llm_parse）
    deepseek_client.py     OpenAI 兼容 LLM 客户端
    llm_parse.py           JSON 对象/数组解析
- assemble/:              决策合并与字幕输出
    assemble.py            合并所有决策 → auto_selected.json + 报告
    subtitle_generator.py  字幕轴 → 句子列表
    annotated_subtitle.py  带标注字幕 txt
- server/:                审核服务器
    review_server.py       剪口播审核服务（静态文件 + 剪辑）

使用方法:
    python -m speech_error_detector.test.run_speech_pipeline   # 或编程: from speech_error_detector.speech_pipeline import run_pipeline
"""

__version__ = "1.1.0"
