"""Extra real-word experiment: matched frequency/length groups and fixed-anchor groups."""
import collections,csv,hashlib,statistics,subprocess
from pathlib import Path
import run
ROOT=Path(__file__).resolve().parent

def select_groups(words,bytefreq):
    groups=[]
    pool={w:c for w,c in words.items() if w.isascii() and w.isalpha()}
    # Eight different anchors spanning each pool's available byte frequencies.
    for length in (5,7):
        for lo,hi in ((10,19),(50,99),(200,399)):
            by=collections.defaultdict(list)
            for w,c in pool.items():
                if len(w)==length and lo<=c<=hi:by[w[0]].append(w)
            anchors=sorted(by,key=lambda x:(bytefreq[ord(x)],x))
            picked=[anchors[round(i*(len(anchors)-1)/7)] for i in range(8)]
            assert len(set(picked))==8
            for anchor in picked:
                w=min(by[anchor],key=lambda w:hashlib.sha256(('matched-v1:'+w).encode()).digest())
                groups.append(dict(kind='vary_anchor',group=f'L{length}_F{lo}-{hi}',needle=w,length=length,word_count=pool[w],anchor_count=bytefreq[ord(anchor)]))
    # Same length and first byte, spanning the available full-word frequencies.
    for anchor in ('s','t','c'):
        candidates=sorted((w for w in pool if len(w)==5 and w[0]==anchor and pool[w]>=5),key=lambda w:(pool[w],w))
        for i in range(8):
            w=candidates[round(i*(len(candidates)-1)/7)]
            groups.append(dict(kind='vary_word',group=f'L5_anchor_{anchor}',needle=w,length=5,word_count=pool[w],anchor_count=bytefreq[ord(anchor)]))
    return groups

def measure():
    rows,blob=run.sample()
    words={r['word']:int(r['count']) for r in csv.DictReader(open(ROOT/'results/word-frequency.csv'))}
    freq=collections.Counter(blob)
    groups=select_groups(words,freq);run.save_csv(ROOT/'results/grouped-selection.csv',groups)
    keys=list(dict.fromkeys(g['needle'] for g in groups))
    stats={}
    for k in keys:
        n=k.encode();hits=sum(n in row for row in rows)
        span=sum((p+len(n)) if (p:=row.find(n))>=0 else len(row) for row in rows)
        stats[k]=dict(anchor_pct=100*freq[n[0]]/len(blob),hit_pct=100*hits/len(rows),scan_pct=100*span/len(blob))
    collected=collections.defaultdict(list)
    for rep in range(1,4):
        order=keys if rep%2 else list(reversed(keys))
        result=subprocess.run(['./build/demo','blob','build/comments.bin','build/lengths.bin',*order],cwd=ROOT,text=True,capture_output=True,check=True)
        (ROOT/f'results/grouped-{rep}.csv').write_text(result.stdout)
        for r in csv.DictReader(line for line in result.stdout.splitlines() if not line.startswith('#')):
            k=r['needle'];n=k.encode();shape=r['shape']
            expected=sum((row.startswith(n) if shape=='prefix' else row.endswith(n) if shape=='suffix' else n in row) for row in rows)
            assert int(r['hits'])==expected
            collected[k,shape,r['impl']].append(r)
        print(f'Grouped run {rep}/3: {len(keys)} keys; oracle counts verified',flush=True)
    summary=[]
    for g in groups:
        k=g['needle'];r=dict(g,**stats[k])
        for impl in ('scalar','simd','memchr'):
            rs=collected[k,'contains',impl];values=[float(x['ms']) for x in rs]
            r[impl+'_ms']=statistics.median(values);r[impl+'_min_ms']=min(values);r[impl+'_max_ms']=max(values)
        for field in ('candidates','blocks','active_blocks','tail_starts','scalar_starts','scalar_compares'):
            r[field]=int(collected[k,'contains','simd'][0][field])
        r['speedup']=r['scalar_ms']/r['simd_ms'];summary.append(r)
    run.save_csv(ROOT/'results/grouped-summary.csv',summary)
    publish()

def place_labels(fig, axes):
    # Place annotations after layout so every selected word stays readable.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for ax in axes:
        occupied=[]
        for label in ax.texts:
            for dx,dy in [(4,8),(4,-18),(-85,8),(-85,-18),(4,26),(4,-36),(-85,26),(-85,-36),(4,44)]:
                label.set_position((dx,dy))
                box=label.get_window_extent(renderer).expanded(1.04,1.15)
                if not any(box.overlaps(other) for other in occupied):
                    break
            occupied.append(box)

def publish():
    import plot
    plt=plot.plt
    rs=list(csv.DictReader(open(ROOT/'results/grouped-summary.csv')))
    f=lambda r,k:float(r[k])
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,3,figsize=(18,10),layout='constrained')
    groups=list(dict.fromkeys(r['group'] for r in rs if r['kind']=='vary_anchor'))
    for ax,group in zip(axs.flat,groups):
        entries=[r for r in rs if r['group']==group]
        for i,r in enumerate(entries):
            ax.scatter(f(r,'anchor_pct'),f(r,'speedup'),c=[f(r,'scan_pct')],cmap='viridis',vmin=90,vmax=100,s=50)
            ax.annotate(r['needle']+' ('+r['word_count']+')',(f(r,'anchor_pct'),f(r,'speedup')),xytext=(3,8 if i%2==0 else -18),textcoords='offset points',fontsize=8)
        ax.set_ylim(0,11);ax.set_xlim(.001,20);ax.set_xscale('log');ax.set_title(f"{entries[0]['length']} bytes; word count {group.split('_F')[1]}");ax.set_xlabel('First-byte frequency (%)');ax.set_ylabel('Scalar / SIMD time');ax.grid(alpha=.2);ax.margins(.25)
    fig.colorbar(plt.cm.ScalarMappable(norm=plot.Normalize(90,100),cmap='viridis'),ax=list(axs.flat),label='Actual search span (%)',shrink=.8)
    fig.suptitle('Matched word length and frequency',fontsize=14)
    place_labels(fig,list(axs.flat))
    plot.save_figure(fig,'matched-anchor','matched-anchor.png')
    fig,axs=plt.subplots(1,3,figsize=(18,6),layout='constrained')
    for ax,group in zip(axs,dict.fromkeys(r['group'] for r in rs if r['kind']=='vary_word')):
        entries=[r for r in rs if r['group']==group]
        for i,r in enumerate(entries):
            ax.scatter(f(r,'word_count'),f(r,'speedup'),c=[f(r,'scan_pct')],cmap='viridis',vmin=0,vmax=100,s=50)
            ax.annotate(r['needle'],(f(r,'word_count'),f(r,'speedup')),xytext=(3,8 if i%2==0 else -18),textcoords='offset points',fontsize=8)
        ax.set_xscale('log');ax.set_title(f"5 bytes; first byte '{entries[0]['needle'][0]}'");ax.set_xlabel('Whole-word occurrences');ax.set_ylabel('Scalar / SIMD time');ax.grid(alpha=.2);ax.margins(.25)
    fig.colorbar(plt.cm.ScalarMappable(norm=plot.Normalize(0,100),cmap='viridis'),ax=list(axs),label='Actual search span (%)',shrink=.8)
    fig.suptitle('Matched word length and first byte',fontsize=14)
    place_labels(fig,list(axs))
    plot.save_figure(fig,'matched-word','matched-word.png')
    body='## 控制词频和词长后的分组比较\n\n'
    body+='**固定词长、限制完整词频后，首字符稀疏的组内端点仍获得更大的 SIMD 收益。**\n\n补充实验选 72 个真实词，分九组、每组八词：六组固定词长（5 或 7 字节）和词频范围（10–19、50–99、200–399 次）；三组固定五字节长度和首字符（`s`、`t`、`c`）。选词只依据词表与字节计数。\n\n'
    body+='每词测量七条路径、独立运行三轮，第二轮反转词序。下面展示 contains 的中位数结果；逐轮计时、命中率和工作量见 [grouped-summary.csv](results/grouped-summary.csv) 与 `results/grouped-{1,2,3}.csv`。\n\n'
    body+='### 固定词频范围与词长\n\n**六组都显示：首字符稀疏的一端约加速 8–9 倍，密集的一端约 1.3 倍。**\n\n![固定词长与词频范围，比较首字符频率](results/matched-anchor.png)\n\n'
    body+='每格一组，括号内为完整词次数。横轴是首字符频率，纵轴是加速比；颜色表示搜索范围占正文的比例。六格坐标尺度相同。\n\n| 组 | 完整词次数范围 | 命中行比例范围 | 实际搜索范围 | 最稀疏首字符的词 / 加速比 | 最密集首字符的词 / 加速比 |\n|---|---:|---:|---:|---|---|\n'
    for group in groups:
        es=[r for r in rs if r['group']==group];a=min(es,key=lambda r:f(r,'anchor_pct'));b=max(es,key=lambda r:f(r,'anchor_pct'))
        rang=lambda k,fmt:f'{min(f(r,k) for r in es):{fmt}}–{max(f(r,k) for r in es):{fmt}}'
        body+=f"| {es[0]['length']} 字节 / {group.split('_F')[1]} 次 | {rang('word_count','.0f')} | {rang('hit_pct','.2f')}% | {rang('scan_pct','.2f')}% | `{a['needle']}` / {f(a,'speedup'):.2f}× | `{b['needle']}` / {f(b,'speedup'):.2f}× |\n"
    body+='\n六组的搜索范围都接近整份正文，组内差异主要表现为首字符过滤后留下多少候选。\n\n'
    body+='### 固定首字符与词长\n\n**这三组中，完整词频越高并未带来持续上升的加速比。**\n\n![固定首字符与词长，比较完整词频](results/matched-word.png)\n\n每格固定一个首字符，横轴为完整词次数，纵轴为加速比，颜色表示搜索范围。三格纵轴范围不同，比较数值时请读刻度。下表列出各组词频最低和最高的词。\n\n| 组 | 词（次数） | 命中行比例 | 搜索范围 | 标量 ms | SIMD ms | 加速比 |\n|---|---|---:|---:|---:|---:|---:|\n'
    for group in dict.fromkeys(r['group'] for r in rs if r['kind']=='vary_word'):
        es=[r for r in rs if r['group']==group]
        for r in (min(es,key=lambda r:f(r,'word_count')),max(es,key=lambda r:f(r,'word_count'))):
            body+=f"| 首字符 `{r['needle'][0]}`，5 字节 | `{r['needle']}` ({r['word_count']}) | {f(r,'hit_pct'):.2f}% | {f(r,'scan_pct'):.2f}% | {f(r,'scalar_ms'):.3f} | {f(r,'simd_ms'):.3f} | {f(r,'speedup'):.2f}× |\n"
    bykey={r['needle']:r for r in rs};a=bykey['tails'];b=bykey['think']
    body+=f"\n`think` 比 `tails` 少扫描约 {f(a,'scan_pct')-f(b,'scan_pct'):.0f} 个百分点的正文，标量耗时下降，SIMD 耗时接近，因此加速比从 {f(a,'speedup'):.2f}× 降到 {f(b,'speedup'):.2f}×。扫描量下降与相对收益增加是两件事。\n"
    body+='\n### 适用范围与局限\n\n分组限制了词长和词频，后续字符与命中位置仍有差异。完整词按可见文本分词统计，contains 则匹配原始字节子串；表中用实际命中率和搜索范围补足这一区别。\n'
    p=ROOT/'README.md';s=p.read_text();start='<!-- GROUPED_START -->';end='<!-- GROUPED_END -->'
    if start in s:
        before,rest=s.split(start,1);_,after=rest.split(end,1);s=before+start+'\n'+body+'\n'+end+after
    else:s=s.replace('## 运行方法与测量口径',start+'\n'+body+'\n'+end+'\n\n## 运行方法与测量口径')
    # Use the files just rendered; content hashes avoid stale image caches.
    for stem in ('matched-anchor','matched-word'):
        filename=next((ROOT/'results').glob(stem+'-*.png')).name;s=s.replace('results/'+stem+'.png','results/'+filename)
    p.write_text(s)

if __name__=='__main__':
    import sys
    publish() if '--plots-only' in sys.argv else measure()
