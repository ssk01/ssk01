"""Plots: population buckets, sampled speedups, and two separate cost mechanisms."""
import csv
import hashlib
import io
import re
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'build/mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
ROOT=Path(__file__).resolve().parent

def read(name):return list(csv.DictReader((ROOT/'results'/name).open()))
def save_figure(fig, stem, legacy=None):
    buffer=io.BytesIO()
    fig.savefig(buffer,format='png',dpi=170)
    data=buffer.getvalue()
    filename=stem+'-'+hashlib.sha256(data).hexdigest()[:12]+'.png'
    (ROOT/'results'/filename).write_bytes(data)
    p=ROOT/'README.md'; text=p.read_text()
    names=[re.escape(stem)+r'-[0-9a-f]+\.png']
    if legacy: names.append(re.escape(legacy))
    text=re.sub(r'results/(?:'+'|'.join(names)+r')', 'results/'+filename, text)
    p.write_text(text)
    for old in (ROOT/'results').glob(stem+'-*.png'):
        if old.name!=filename:old.unlink()
    if legacy:(ROOT/'results'/legacy).unlink(missing_ok=True)
    plt.close(fig)
    return filename

def draw():
    stats=read('key-statistics.csv');summary=read('summary.csv');selection=read('bucket-selection.csv')
    lookup={(r['config'],r['needle'],r['shape'],r['impl']):r for r in summary}
    def get(k,shape='contains',impl='simd'):return lookup[('manual',k,shape,impl)]
    def speed(k,shape='contains'):return float(get(k,shape,'scalar')['median_ms'])/float(get(k,shape)['median_ms'])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fbfaf7','axes.facecolor':'#fbfaf7'})
    fig,axs=plt.subplots(2,2,figsize=(18,10),layout='constrained')
    names={'first_byte':'First-byte frequency','whole_word':'Whole-word frequency'}
    for row,d in enumerate(names):
        groups=[[r for r in selection if r['distribution']==d and int(r['bucket'])==b] for b in range(1,11)]
        ax=axs[row,0];ax.bar(range(1,11),[int(g[0]['eligible_words']) for g in groups],color='#367c89')
        ax.set_yscale('log');ax.set_ylabel('Eligible distinct words (log scale)');ax.set_title(names[d]+': vocabulary distribution')
        ax=axs[row,1]
        for b,g in enumerate(groups,1):
            ax.scatter([b-.12,b,b+.12][:len(g)],[float(r['probability'])*100 for r in g],color='#bd7843',s=32)
        ax.set_yscale('log');ax.set_ylabel('First byte / raw bytes (%)' if d=='first_byte' else 'Whole word / visible tokens (%)');ax.set_title('3 deterministic samples from each bucket')
        for ax in axs[row]:ax.set_xticks(range(1,11));ax.set_xlabel('Log-frequency bucket: rare (1) -> common (10)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        labels=[str(b)+'\n'+'\n'.join(r['needle'] for r in g) for b,g in enumerate(groups,1)]
        axs[row,1].set_xticks(range(1,11),labels,fontsize=8)
        axs[row,1].set_xlabel('Bucket and selected words (points: left / middle / right)')
    fig.suptitle('Word selection: two frequency measures',fontsize=16)
    save_figure(fig,'frequency-buckets','frequencies.png')

    colors={'prefix':'#bd7843','suffix':'#89839f','contains':'#367c89'}
    # Give every sampled word its own labelled row, so individual dots are identifiable.
    fig,axs=plt.subplots(1,2,figsize=(16,15),layout='constrained')
    for ax,d in zip(axs,names):
        entries=[(f"B{int(r['bucket']):02d}  {r['needle']}",r['needle'],int(r['bucket'])) for r in selection if r['distribution']==d]
        if d=='first_byte':entries += [('CTRL  '+k,k,11) for k in ['youtube','YouTube','you','yes']]
        ys=np.arange(len(entries))
        for row,(_,key,bucket) in enumerate(entries):
            if bucket%2:ax.axhspan(row-.5,row+.5,color='#e9e6df',alpha=.5,zorder=0)
        for shape,marker in [('prefix','o'),('suffix','s'),('contains','D')]:
            ax.scatter([speed(k,shape) for _,k,_ in entries],ys,color=colors[shape],marker=marker,s=37,label=shape,zorder=3)
        ax.set_yticks(ys,[label for label,_,_ in entries],fontsize=10)
        ax.set_ylim(len(entries)-.5,-.5);ax.set_xscale('log');ax.axvline(1,color='#333',ls='--',lw=1)
        ax.set_xticks([.5,1,2,4,8,16],['.5x','1x','2x','4x','8x','16x'])
        ax.set_xlim(.48,16);ax.set_title('Selected by '+names[d].lower()+'\nB01–B10: low to high frequency',pad=14)
        ax.set_xlabel('Scalar time / SIMD time; right of 1x = SIMD faster');ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right')
    fig.suptitle('SIMD speedup for each word',fontsize=15)
    save_figure(fig,'selected-word-speedups','selected-words.png')

    fig,axs=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    norm=Normalize(0,100);cmap='viridis';c=[float(r['contains_hit_pct']) for r in stats]
    scatter=axs[0].scatter([float(r['anchor_pct']) for r in stats],[speed(r['needle']) for r in stats],c=c,cmap=cmap,norm=norm,s=45,edgecolors='white',lw=.4)
    axs[0].set_xscale('log');axs[0].set_xlabel('First-byte frequency in raw text (%) [log scale]');axs[0].set_ylabel('Contains scalar time / SIMD time');axs[0].axhline(1,color='#777',ls='--',lw=1);axs[0].set_title('First-byte frequency vs. speedup')
    axs[1].scatter([int(get(r['needle'])['candidates'])/1e6 for r in stats],[float(get(r['needle'])['median_ms']) for r in stats],c=c,cmap=cmap,norm=norm,s=45,edgecolors='white',lw=.4)
    axs[1].set_xlabel('Verified SIMD candidates per whole scan (millions)');axs[1].set_ylabel('SIMD contains latency (ms)');axs[1].set_title('Candidate count vs. SIMD time')
    labels=['youtube','YouTube','the']
    maxkey=max(stats,key=lambda r:int(get(r['needle'])['candidates']))['needle']
    labels=list(dict.fromkeys(labels+[maxkey]))
    for k in labels:
        r=next(r for r in stats if r['needle']==k);m=get(k)
        for ax,x,y in [(axs[0],float(r['anchor_pct']),speed(k)),(axs[1],int(m['candidates'])/1e6,float(m['median_ms']))]:
            offset={'youtube':(8,10),'YouTube':(8,-18),'the':(8,-18)}.get(k,(-25,12))
            ax.annotate(k,(x,y),xytext=offset,textcoords='offset points',fontsize=9)
    for ax in axs:ax.grid(alpha=.15);ax.margins(.17)
    cb=fig.colorbar(scatter,ax=axs,shrink=.8,pad=.02);cb.set_label('Rows containing full pattern (%)')
    fig.suptitle('Contains: filtering and verification',fontsize=15)
    save_figure(fig,'candidate-mechanism','mechanism.png')
if __name__=='__main__':draw()
