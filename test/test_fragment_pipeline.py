"""
测试 fragment 多轮检测 - 红姐 30-36
模拟 detect_loop 中 fragment 的机械检测 + 逐轮应用（无 LLM judge，直接采纳所有 finding）
"""
import os
import sys

# 本文件位于 speech_error_detector/test/，需将 output/ 根目录加入 sys.path
# 以便 `import speech_error_detector.detect.detect_fragment` 可用
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# fmt: off
SENTENCES = [
    {"idx":30,"range":"531-562","text":"那会呢我还没有完全地琢磨明白但是呢这句话可是给我镇住了啊"},
    {"idx":31,"range":"564-566","text":"寄到"},
    {"idx":32,"range":"568-568","text":"县"},
    {"idx":33,"range":"570-579","text":"我呢一直是记到现在"},
    {"idx":34,"range":"581-601","text":"后来呢他给我呃在复盘以前做的对照实验"},
    {"idx":35,"range":"605-626","text":"但是这句话那可是把我镇住了我一直记到现在"},
    {"idx":36,"range":"628-652","text":"后来呢他给我复盘以前做的对照实验我亲眼看见"},
]
# fmt: on

# 期望最终结果（句 idx -> 期望文本；空串表示整句删除）
EXPECTED = {
    30: "那会呢我还没有完全地琢磨明白",
    31: "",
    32: "",
    33: "",
    34: "",
    35: "但是这句话那可是把我镇住了我一直记到现在",
    36: "后来呢他给我复盘以前做的对照实验我亲眼看见",
}

from speech_error_detector.detect.detect_fragment import detect_fragment

# ===== 福总 整篇句子（来自 2026-07-07_福总/2_分析/sentences.txt）=====
# fmt: off
FUZONG_SENTENCES = [
    {"idx":0,"range":"1-28","text":'4月23日明天周四的行情不用猜了我直接呢把答案甩给你们'},
    {"idx":1,"range":"30-53","text":'要是周四行情呢跟我说的差半个字你尽管过来取关我'},
    {"idx":2,"range":"55-61","text":'怎么骂我我都认'},
    {"idx":3,"range":"63-82","text":'今天呢a股表现用炸裂来形容啊毫不夸张'},
    {"idx":4,"range":"84-107","text":'早盘低开以后资金啊如同开闸的洪水般疯狂涌入啊'},
    {"idx":5,"range":"109-131","text":'沪指一路强势攀升全天成交额直接突破1.1万亿'},
    {"idx":6,"range":"133-138","text":'量能大幅释放'},
    {"idx":7,"range":"139-159","text":'收盘时呢两市超2900只个股飘红赚钱效应拉满啊'},
    {"idx":8,"range":"161-177","text":'所有看空的声音呢都被市场狠狠压制'},
    {"idx":9,"range":"179-193","text":'家人们我太懂你们此刻的心情了'},
    {"idx":10,"range":"195-230","text":'持仓的朋友既开心是又忐忑生怕明天行情跳水把到手的利润又吐回去'},
    {"idx":11,"range":"232-252","text":'踏空的朋友们急得直跺脚想追高又怕被套牢'},
    {"idx":12,"range":"254-266","text":'选择观望呢又怕错过后续行情'},
    {"idx":13,"range":"268-293","text":'不管你是哪种情况都别慌静下心来把我的这条视频看完'},
    {"idx":14,"range":"298-315","text":'老粉啊都清楚我的每次提醒呢都恰到好处'},
    {"idx":15,"range":"317-357","text":'我从来不是靠吹牛全凭硬核逻辑和真实数据始终站在散户的立场上说真话讲实情'},
    {"idx":16,"range":"359-399","text":'觉得我实在的朋友来点个小红心留下一句红红火火祝大家2026年呢账户飘红一路高升'},
    {"idx":17,"range":"401-467","text":'回归盘面核心啊今天的上涨呢绝非一日游行情也不是主力诱多陷阱而是大级别牛市主升浪的加速阶段三大硬核逻辑让你心里彻底有底'},
    {"idx":18,"range":"472-477","text":'第一技术面'},
    {"idx":19,"range":"479-498","text":'沪指成功突破4100点强压力位'},
    {"idx":20,"range":"500-510","text":'彻底打开上方的上涨空间'},
    {"idx":21,"range":"512-520","text":'均线呢呈现多头排列'},
    {"idx":22,"range":"522-536","text":'macdkdj指标全线金叉'},
    {"idx":23,"range":"539-546","text":'上行趋势已然走牛'},
    {"idx":24,"range":"548-562","text":'趋势一旦形成就不会轻易反转'},
    {"idx":25,"range":"564-585","text":'今天的上涨是趋势的延续'},
    {"idx":26,"range":"589-607","text":'明天只会顺势前行不会出现深度调整'},
    {"idx":27,"range":"609-615","text":'来第二个资金面'},
    {"idx":28,"range":"617-648","text":'今夜两市成交额呢超2.5万亿市场资金供给充足承接力度极强'},
    {"idx":29,"range":"650-683","text":'盘中即便出现小幅回调呢也立刻有大量资金进场抄底将指数拉回正轨'},
    {"idx":30,"range":"685-714","text":'北向资金持续净流入内资果断加仓散户呢也纷纷跑步进场'},
    {"idx":31,"range":"716-746","text":'三方资金形成强大合力明天盘面呢只会更强根本就没有下跌空间'},
    {"idx":32,"range":"750-756","text":'第三个是情绪面'},
    {"idx":33,"range":"758-793","text":'市场情绪呢已经进入亢奋状态超2900只个股上涨赚钱效应拉满市场信心十足啊'},
    {"idx":34,"range":"795-816","text":'在这种情绪氛围下任何回调都是绝佳的上车机会'},
    {"idx":35,"range":"818-829","text":'不会出现恐慌性砸盘的情况'},
    {"idx":36,"range":"831-850","text":'接下来散户最关心的核心主线我直接给大家'},
    {"idx":37,"range":"852-873","text":'一AI通讯半导体依旧是本轮牛市的绝对主线'},
    {"idx":38,"range":"875-905","text":'昨天的震荡调整正是绝佳的上车机会千万别被短期洗盘洗下车啊'},
    {"idx":39,"range":"907-922","text":'拿住核心龙头才能吃到最大的收益'},
    {"idx":40,"range":"926-931","text":''},
    {"idx":41,"range":"933-950","text":''},
    {"idx":42,"range":"954-987","text":'二人工智能新能源车是今日资金调仓的核心方向也是接下来的补涨属性'},
    {"idx":43,"range":"989-1004","text":'踏空的朋友重点关注这两个板块'},
    {"idx":44,"range":"1006-1053","text":'三锂矿电池材料固态电池板块前期啊调整充分业绩确定性强安全边际高值得重点布局'},
    {"idx":45,"range":"1058-1073","text":'以上呢就是我对明天行情的全部观点'},
    {"idx":46,"range":"1075-1091","text":'明天的具体操作思路'},
    {"idx":47,"range":"1093-1131","text":'和周四的核心布局方向我都整理在了商品橱窗的白皮书里大家呢直接去橱窗领取就ok'},
]
# fmt: on

# ===== 红姐 整篇句子（来自 2026-07-07_红姐/2_分析/sentences.txt）=====
# fmt: off
HONGJIE_SENTENCES = [
    {"idx":0,"range":"2-37","text":'假如你手里攥着10万块股价10块的时候你一把梭哈你全砸进去了'},
    {"idx":1,"range":"39-52","text":'结果呢刚买完就一路往下砸'},
    {"idx":2,"range":"54-62","text":'这个时候你咋整啊'},
    {"idx":3,"range":"64-84","text":'我敢说啊绝大多数的散户第一反应就是补仓'},
    {"idx":4,"range":"86-97","text":'寻思啊哎多买点咱们'},
    {"idx":5,"range":"99-103","text":'摊薄成本'},
    {"idx":6,"range":"105-122","text":'网上一大堆大v也都是这么教的啊'},
    {"idx":7,"range":"124-137","text":'管这招啊叫做无限补仓法'},
    {"idx":8,"range":"139-167","text":'但是今天我呢掏心窝子跟大伙说啊这套思路压根它就不对'},
    {"idx":9,"range":"169-174","text":'错得非常离谱'},
    {"idx":10,"range":"176-195","text":'下跌趋势一旦走出来根本不像你们想的啊'},
    {"idx":11,"range":"197-208","text":'跌个20个点就到底反弹了'},
    {"idx":12,"range":"210-223","text":'哎不知道这话是谁传出来的'},
    {"idx":13,"range":"225-241","text":'只有真在股市里挨过毒打的人那才明白'},
    {"idx":14,"range":"243-249","text":'底下还有底呢'},
    {"idx":15,"range":"251-270","text":'跌完一波还有一波抄底根本就看不着头'},
    {"idx":16,"range":"272-285","text":'今天呢我给大伙整四步法子啊'},
    {"idx":17,"range":"287-306","text":'能一点点把套牢的仓位哎给它盘活'},
    {"idx":18,"range":"308-331","text":'这套操作啊你得熬时间有耐心的老铁你好好听啊'},
    {"idx":19,"range":"333-342","text":'干货我全撂这了啊'},
    {"idx":20,"range":"344-372","text":'我是在2014年牛市启动就入市到现在呢整整是呃炒了12年了'},
    {"idx":21,"range":"374-399","text":'呃刚炒股那会呢我也认准越跌越买别人恐惧我贪婪'},
    {"idx":22,"range":"401-415","text":'哎股价越便宜哎越值得加仓'},
    {"idx":23,"range":"417-432","text":'可是这个想法本身它就是个误区'},
    {"idx":24,"range":"434-448","text":'谁规定股价便宜就值得下手啊'},
    {"idx":25,"range":"450-469","text":'直到2019年我碰到了一位实打实厉害的游资大佬'},
    {"idx":26,"range":"471-502","text":'当时呢她直接点醒我她说小红啊你一门心思地摊低成本'},
    {"idx":27,"range":"504-508","text":'你咋不想想'},
    {"idx":28,"range":"510-520","text":'你手里现金流的主动权'},
    {"idx":29,"range":"522-529","text":'全交到主力手里了'},
    {"idx":30,"range":"531-562","text":'那会呢我还没有完全地琢磨明白'},
    {"idx":31,"range":"564-566","text":''},
    {"idx":32,"range":"568-568","text":''},
    {"idx":33,"range":"570-579","text":''},
    {"idx":34,"range":"581-601","text":''},
    {"idx":35,"range":"605-626","text":'但是这句话那可是把我镇住了我一直记到现在'},
    {"idx":36,"range":"628-652","text":'后来呢他给我复盘以前做的对照实验我亲眼看见'},
    {"idx":37,"range":"654-686","text":'同样10块钱建的底仓有的人死扛硬拿熬老长时间才勉强回本'},
    {"idx":38,"range":"688-695","text":'会滚动操作的直接'},
    {"idx":39,"range":"697-704","text":'赚了七成收益呀'},
    {"idx":40,"range":"706-742","text":'打那之后我就彻底的放弃了无脑补仓照着大佬教的战法去实操到现在呢'},
    {"idx":41,"range":"744-754","text":'再也没有深度套牢过'},
    {"idx":42,"range":"756-776","text":'今天呢免费分享给大伙你呀可得好好珍惜啊'},
    {"idx":43,"range":"778-805","text":'视频呢你给它收藏好开盘前收盘后你反复的多看几遍'},
    {"idx":44,"range":"807-823","text":'咱闲话不多唠直接上四步实操'},
    {"idx":45,"range":"825-849","text":'第一步啊减仓留住活钱啊止损不是让你全清仓'},
    {"idx":46,"range":"851-882","text":'好比股价从10块跌到8块先卖出5000股手里立马回笼4万现金'},
    {"idx":47,"range":"884-897","text":'同时呢还留5000股底仓不动'},
    {"idx":48,"range":"899-903","text":'为啥这么干'},
    {"idx":49,"range":"905-975","text":'下跌趋势成型跌20个点往往只是开胃小菜玩久了你就会懂后面很大概率还有下跌空间把现金攥手里这不叫认输是给自己留足子弹啊'},
    {"idx":50,"range":"977-979","text":'第二步'},
    {"idx":51,"range":"981-992","text":'大跌低位重新吸筹'},
    {"idx":52,"range":"994-1027","text":'等股价跌到5块账户直接腰斩的时候绝大多数散户心态早就崩了'},
    {"idx":53,"range":"1029-1042","text":'拿不住也不知道该咋操作了'},
    {"idx":54,"range":"1044-1054","text":'这个时候啊你得沉住气'},
    {"idx":55,"range":"1056-1076","text":'反倒适合重新布局拿之前撤出来的4万块'},
    {"idx":56,"range":"1078-1087","text":'全在五块附近接回来'},
    {"idx":57,"range":"1089-1099","text":'原先剩5500底仓这会'},
    {"idx":58,"range":"1101-1113","text":'能再买8000股总持仓啊'},
    {"idx":59,"range":"1115-1143","text":'直接一万三千股当初给你一块10块钱进场的人手里依旧是一万股'},
    {"idx":60,"range":"1145-1153","text":'这就是滚动解套'},
    {"idx":61,"range":"1155-1177","text":'最关键的一步下跌途中持续增加持股数量啊'},
    {"idx":62,"range":"1180-1198","text":'第三步反弹碰压力位兑现利润啊'},
    {"idx":63,"range":"1200-1251","text":'股价从5块反弹回8块这个位置一般全是套牢盘压力贼大直接把低位加仓的8000股全部卖掉8000'},
    {"idx":64,"range":"1253-1261","text":''},
    {"idx":65,"range":"1263-1268","text":''},
    {"idx":66,"range":"1271-1284","text":'八块成交到手6万4的资金'},
    {"idx":67,"range":"1286-1298","text":'第一轮滚动操作到此完事'},
    {"idx":68,"range":"1300-1318","text":'本金呢基本上全部回笼底仓丝毫没动'},
    {"idx":69,"range":"1320-1327","text":'白赚了一波差价啊'},
    {"idx":70,"range":"1329-1352","text":'第四步回踩再度低吸反复轮动放大收益'},
    {"idx":71,"range":"1354-1365","text":'你仔细地寻思寻思啊'},
    {"idx":72,"range":"1367-1385","text":'就算是后面股价重新涨回原来的价位'},
    {"idx":73,"range":"1387-1412","text":'那些死扛不动的人一万股刚勉强回本你中途来回滚动'},
    {"idx":74,"range":"1414-1426","text":'做差价早就赚的盆满钵满啦'},
    {"idx":75,"range":"1428-1447","text":'最后呢我再多絮叨两句实操细节啊'},
    {"idx":76,"range":"1449-1463","text":'真实盘面不会这么规整啊'},
    {"idx":77,"range":"1465-1485","text":'会来回的震荡还总出假跌破假拉升的套路'},
    {"idx":78,"range":"1487-1518","text":'你呀别死板盯着十块八块五块这个固定的价格生搬硬套啊'},
    {"idx":79,"range":"1520-1561","text":'我讲这些呢不是让你记死数字是吃透滚动加仓啊不断地增厚筹码的底层逻辑'},
    {"idx":80,"range":"1563-1603","text":'你把这套逻辑你给它摸透以后被套的仓位你呀都能够从容地把控仓位节奏'},
]
# fmt: on

MAX_ROUNDS = 6
cur = [dict(s) for s in SENTENCES]  # 深拷贝

print("=" * 70)
print("fragment 多轮检测 — 红姐 30-36 片段")
print("=" * 70)

for det_round in range(1, MAX_ROUNDS + 1):
    print(f"\n{'─' * 50}")
    print(f"第 {det_round} 轮 ─ 当前句子 ({sum(1 for s in cur if s['text'].strip())} 个有内容):")
    for s in cur:
        marker = "  " if s["text"].strip() else "❌"
        print(f"  {marker} 句{s['idx']:>2} [{s['range']:>7}] ➜ {s['text'][:45]}{'...' if len(s['text'])>45 else ''}")

    # 模拟 pipeline: 过滤空句子后送检测
    _detect_sents = [s for s in cur if s.get("text", "").strip()]
    findings = detect_fragment(_detect_sents)

    if not findings:
        print(f"\n  ✅ 无新发现，收敛！")
        break

    print(f"\n  ⚡ 本轮回发现 {len(findings)} 处:")
    for f in findings:
        print(f"    [{f['subtype']}] 句{f['sent_idx']} "
              f"head=『{f.get('head_text','')[:20]}』"
              f"tail=『{f.get('tail_text','')[:20]}』"
              f" | {f['decision_hint'][:60]}...")

    # 模拟 judge 全批准 + 逐条应用
    applied_this_round = 0
    for f in findings:
        sid = f["sent_idx"]
        sent = next(s for s in cur if s["idx"] == sid)
        old_text = sent["text"]

        if "跨句头-头口误" in f.get("subtype", ""):
            # 整句删(mode=full)
            sent["text"] = ""
        elif "跨句尾部重叠" in f.get("subtype", ""):
            # 整句删(mode=full)
            sent["text"] = ""
        elif "前句尾与后句头重叠(头体重叠" in f.get("subtype", ""):
            # keep_head: 保留 head_text（前句独有头部），删除尾部
            ht = f.get("head_text", "")
            if ht and old_text.startswith(ht):
                sent["text"] = ht
        elif "残句(被后句接续重说)" in f.get("subtype", ""):
            ht = f.get("head_text", "")
            if ht:
                sent["text"] = ht
            elif f.get("overlap_len", 0) >= len(old_text):
                sent["text"] = ""
        elif "句尾句头重叠+长停顿" in f.get("subtype", ""):
            ht = f.get("head_text", "")
            if ht:
                sent["text"] = ht
        elif "极短孤立句" in f.get("subtype", ""):
            sent["text"] = ""
        elif "孤立编号" in f.get("subtype", ""):
            sent["text"] = ""

        if sent["text"] != old_text:
            applied_this_round += 1
            delta = len(old_text) - len(sent["text"])
            print(f"    ✓ 应用 句{sid}: 删除 {delta} 字 ➜ 『{sent['text'][:30]}』")

    print(f"\n  本轮回应用 {applied_this_round} 条，删除"
          f" {sum(len(s['text']) for s in SENTENCES) - sum(len(s['text']) for s in cur)} 字")

print(f"\n{'=' * 70}")
print("最终结果:")
print("=" * 70)
for s in cur:
    if s["text"].strip():
        print(f"  句{s['idx']:>2} [{s['range']:>7}] ➜ {s['text']}")
    else:
        print(f"  句{s['idx']:>2} [{s['range']:>7}] ➜ (已删除)")

def run_full_detect(name: str, sentences: list[dict], max_rounds: int = MAX_ROUNDS) -> None:
    """对整篇句子跑多轮 fragment 检测并应用（无 LLM judge，直接采纳所有 finding）。

    与上面 30-36 片段使用完全相同的迭代逻辑，仅用于把整篇语料放进检测、
    观察机械检测在真实长文本上的表现，不做断言。
    """
    cur = [dict(s) for s in sentences]
    print(f"\n{'=' * 70}")
    print(f"整篇检测 — {name}（共 {len(cur)} 句，{sum(1 for s in cur if s['text'].strip())} 句有内容）")
    print("=" * 70)

    for det_round in range(1, max_rounds + 1):
        _detect_sents = [s for s in cur if s.get("text", "").strip()]
        findings = detect_fragment(_detect_sents)
        if not findings:
            print(f"  ✅ 第 {det_round} 轮无新发现，收敛！")
            break

        print(f"  第 {det_round} 轮回发现 {len(findings)} 处:")
        for f in findings:
            print(f"    [{f['subtype']}] 句{f['sent_idx']} "
                  f"head=『{f.get('head_text','')[:18]}』"
                  f"tail=『{f.get('tail_text','')[:18]}』")
        applied = 0
        for f in findings:
            sid = f["sent_idx"]
            sent = next(s for s in cur if s["idx"] == sid)
            old = sent["text"]
            if "跨句头-头口误" in f.get("subtype", ""):
                sent["text"] = ""
            elif "跨句尾部重叠" in f.get("subtype", ""):
                sent["text"] = ""
            elif "前句尾与后句头重叠(头体重叠" in f.get("subtype", ""):
                ht = f.get("head_text", "")
                if ht and old.startswith(ht):
                    sent["text"] = ht
            elif "残句(被后句接续重说)" in f.get("subtype", ""):
                ht = f.get("head_text", "")
                if ht:
                    sent["text"] = ht
                elif f.get("overlap_len", 0) >= len(old):
                    sent["text"] = ""
            elif "句尾句头重叠+长停顿" in f.get("subtype", ""):
                ht = f.get("head_text", "")
                if ht:
                    sent["text"] = ht
            elif "极短孤立句" in f.get("subtype", ""):
                sent["text"] = ""
            elif "孤立编号" in f.get("subtype", ""):
                sent["text"] = ""
            if sent["text"] != old:
                applied += 1
        print(f"  → 应用 {applied} 条，剩余 {sum(1 for s in cur if s['text'].strip())} 句有内容")

    kept = sum(1 for s in cur if s["text"].strip())
    deleted = sum(1 for s in cur if not s["text"].strip())
    print(f"  📊 结果：保留 {kept} 句，删除 {deleted} 句")


run_full_detect("福总", FUZONG_SENTENCES)
run_full_detect("红姐", HONGJIE_SENTENCES)


# === 校验最终结果是否符合 EXPECTED ===
failures = []
for s in cur:
    exp = EXPECTED.get(s["idx"])
    if exp is None:
        continue
    if s["text"] != exp:
        failures.append(
            f"  句{s['idx']}: 期望『{exp}』 实际『{s['text']}』"
        )
if failures:
    print("\n❌ 结果与 EXPECTED 不符:")
    print("\n".join(failures))
    raise SystemExit(1)
print("\n✅ 结果与 EXPECTED 完全一致")
