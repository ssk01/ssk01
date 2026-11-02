"""Keep measured tables and explanatory examples in the sole tutorial README."""
import csv
import re
import statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def update():
    summary=list(csv.DictReader((ROOT/'results/summary.csv').open()))
    stats={r['needle']:r for r in csv.DictReader((ROOT/'results/key-statistics.csv').open())}
    selections=list(csv.DictReader((ROOT/'results/bucket-selection.csv').open()))
    lookup={(r['config'],r['needle'],r['shape'],r['impl']):r for r in summary}
    def get(k,shape='contains',impl='simd',config='manual'):return lookup[(config,k,shape,impl)]
    def ms(k,shape='contains',impl='simd',config='manual'):return float(get(k,shape,impl,config)['median_ms'])
    def ratio(k,shape='contains'):return ms(k,shape,'scalar')/ms(k,shape)
    text='### youtube 的耗时与工作量\n\n'
    text+='| 形状 | 标量 ms | SIMD ms | 加速比 | 标量尝试起点数 | 标量字节比较数 |\n|---|---:|---:|---:|---:|---:|\n'
    for shape in ['prefix','suffix','contains']:
        r=get('youtube',shape)
        text+=f"| {shape} | {ms('youtube',shape,'scalar'):.4f} | {ms('youtube',shape):.4f} | {ratio('youtube',shape):.2f}× | {int(r['scalar_starts']):,} | {int(r['scalar_compares']):,} |\n"
    r=get('youtube');p=get('youtube','prefix');t=get('youtube','suffix')
    text+=f"\n标量 prefix 平均每行比较 **{int(p['scalar_compares'])/51204:.2f} 个字节**，suffix 是 **{int(t['scalar_compares'])/51204:.2f} 个字节**。它们已经很快结束，SIMD 的固定装载与归约成本很难摊薄。contains 的标量则尝试 **{int(r['scalar_starts']):,} 个起点**；SIMD 用 **{int(r['blocks']):,} 个完整块**过滤，再验证 **{int(r['candidates']):,} 个候选**，另有 **{int(r['tail_starts']):,} 个尾部起点**。contains 省掉的是大量逐起点工作，这正是三种形状加速幅度不同的直接原因。\n"
    text+='\n### 典型词的候选数与耗时\n\n'
    text+='搜索范围累计每行开头到首次匹配末尾的字节数；未命中的行计入全文，再除以正文总字节数。它描述提前退出，不是实际内存读取量。\n\n'
    text+='| key | 首字节占比 | 完整模式命中行占比 | 逻辑搜索范围 | SIMD 候选数 | SIMD ms | 标量 / SIMD |\n|---|---:|---:|---:|---:|---:|---:|\n'
    keys=list(dict.fromkeys(['youtube','YouTube','you','yes','the',max(stats,key=lambda k:int(get(k)['candidates']))]))
    for k in keys:
        s=stats[k]
        text+=f"| {k} | {float(s['anchor_pct']):.3f}% | {float(s['contains_hit_pct']):.2f}% | {float(s['logical_search_span_pct']):.2f}% | {int(get(k)['candidates']):,} | {ms(k):.3f} | {ratio(k):.2f}× |\n"
    text+=f"\n`you` 和 `yes` 共享首字节。`you` 更常提前命中，SIMD 耗时更短（{ms('you'):.3f} / {ms('yes'):.3f} ms），但加速比接近（{ratio('you'):.2f}× / {ratio('yes'):.2f}×）：标量也从提前退出中受益。\n"
    text+='\n完整选词、计数区间和词池大小见 [bucket-selection.csv](results/bucket-selection.csv)；图中已标出每桶三个词。\n'
    bucket_rows=[]
    for d in ['first_byte','whole_word']:
        for b in range(1,11):
            ks=[r['needle'] for r in selections if r['distribution']==d and int(r['bucket'])==b]
            for shape in ['prefix','suffix','contains']:
                values=[ratio(k,shape) for k in ks]
                bucket_rows.append(dict(distribution=d,bucket=b,shape=shape,sampled_words=len(ks),
                                        median_speedup=statistics.median(values),min_speedup=min(values),max_speedup=max(values)))
    with (ROOT/'results/bucket-performance.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=bucket_rows[0].keys(), lineterminator="\n");w.writeheader();w.writerows(bucket_rows)
    bm={(r['distribution'],r['bucket'],r['shape']):r['median_speedup'] for r in bucket_rows}
    text+='\n### 分桶结果\n\n'
    text+=f"首字节频率第 1 桶与第 10 桶的 contains 加速比中位数为 {bm[('first_byte',1,'contains')]:.2f}× / {bm[('first_byte',10,'contains')]:.2f}×。这两端分别抽自首字节稀疏和密集的词池，用来观察过滤候选的难易程度。全文词频第 1 桶与第 10 桶对应 {bm[('whole_word',1,'contains')]:.2f}× / {bm[('whole_word',10,'contains')]:.2f}×，观察的是完整词出现频率这一不同维度。每桶只有三个词，词长、命中位置和验证长度也会变化；应结合逐词图与实际工作量理解曲线。分桶汇总见 [bucket-performance.csv](results/bucket-performance.csv)。\n"
    text+='\n### 自动向量化与 libc 对照\n\n| key | 手写标量 ms | 手写 SIMD ms | 系统 NEON memchr + 验证 ms | 默认 -O3 标量 / SIMD ms |\n|---|---:|---:|---:|---:|\n'
    for k in ['youtube','YouTube','you','the']:
        text+=f"| {k} | {ms(k,impl='scalar'):.3f} | {ms(k):.3f} | {ms(k,impl='memchr'):.3f} | {ms(k,impl='scalar',config='auto'):.3f} / {ms(k,config='auto'):.3f} |\n"
    text+=f'\n默认优化也可能改变 lane 提取与辅助代码，表中测的是完整编译产物。全部 {len(stats)} 个 key 的逐项数据与三轮范围保存在 `summary.csv`，分桶统计只汇总每桶抽中的三个词。\n'
    p=ROOT/'README.md';s=p.read_text();start='<!-- MEASURED_RESULTS_START -->';end='<!-- MEASURED_RESULTS_END -->'
    s=re.sub(r'实际测量 \d+ 个 key',f'实际测量 {len(stats)} 个 key',s)
    s=re.sub(r'全部 \d+ 个测试词',f'全部 {len(stats)} 个测试词',s)
    head='## 首字符过滤与候选验证成本'
    old='## contains 的性能：首字节过滤效率与候选验证成本'
    if old in s:s=s.replace(old,head)
    a=s.index(head);b=s.index('## 首字符频率与完整词频的作用',a)
    figure=re.search(r'!\[[^\]]*\]\([^)]*\)',s[a:b]).group()
    explanation=head+'\n\n**首字符稀疏时，contains 通常获得更大的 SIMD 加速；留下的候选越多，验证成本也越高。** 下面是主实验 64 个词的总体分布，控制词频和词长后的比较见下一节。\n\n'+figure+'\n\n'
    explanation+='左图横轴是首字符占比，纵轴是标量耗时 / SIMD 耗时。右图横轴是完整 SIMD 块内的候选验证总数（百万次），纵轴是 SIMD 耗时（ms）。两图颜色均表示完整模式命中的评论比例。\n\n'
    explanation+=f"`YouTube` 和 `youtube` 的扫描量相近，但候选数分别为 1.26 万和 27.6 万，SIMD 耗时分别为 {ms('YouTube'):.2f} 和 {ms('youtube'):.2f} ms，加速比分别为 {ratio('YouTube'):.2f}× 和 {ratio('youtube'):.2f}×：较少的候选留下了更大的过滤收益。\n\n"
    explanation+=f"同以 `t` 开头的 `the` 和 `tutorial` 在左图横坐标相同。`the` 经常提前命中，SIMD 耗时更短（{ms('the'):.2f} / {ms('tutorial'):.2f} ms），但标量也少做了工作，加速比分别为 {ratio('the'):.2f}× / {ratio('tutorial'):.2f}×。因此要分别观察绝对耗时和相对收益。候选之外的块扫描、mask 提取和尾部处理也计入总耗时。\n\n"
    s=s[:a]+explanation+s[b:]
    before,rest=s.split(start,1);_,after=rest.split(end,1);p.write_text(before+start+'\n'+text+'\n'+end+after)
if __name__=='__main__':update()
