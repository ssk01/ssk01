#!/usr/bin/env python3
import argparse,json,os,platform,statistics,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('binary',type=Path);p.add_argument('--out',type=Path,default=Path('results'));p.add_argument('--repeats',type=int,default=5);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
env={**os.environ,'VECLIB_MAXIMUM_THREADS':'1','LIBXSMM_VERBOSE':'0'}
subprocess.run([str(a.binary.resolve()),'verify'],check=True,env=env)
shapes=[(112,112,512),(128,128,512),(144,144,512),(160,160,512),(176,176,512),(192,192,512),(256,256,512),(384,384,512),(512,512,512)]
jobs=[(f'exec-{m}-{n}-{k}',['exec',str(m),str(n),str(k)]) for m,n,k in shapes]+[(f'cache-{c}',['cache',str(c),'20'])for c in [0,2,8]]
samples={name:[]for name,_ in jobs}
for r in range(a.repeats):
 order=jobs[r:]+jobs[:r]
 if r%2:order=order[::-1]
 for name,args in order:
  if args[0]=='exec':args=[*args,str(r)]
  x=subprocess.run([str(a.binary.resolve()),*args],check=True,env=env,text=True,capture_output=True)
  (a.out/f'{name}-{r}.log').write_text(x.stdout+x.stderr)
  samples[name].append(json.loads(x.stdout));print(r,name,flush=True)
for name in ['cache-0','cache-2','cache-8']:
 for s in samples[name]:
  assert s['requests']==s['hits']+s['misses']==160
  assert s['misses']==(0 if name=='cache-8' else 160)
assert len({s['checksum'] for n in ['cache-0','cache-2','cache-8'] for s in samples[n]})==1
medians={name:{k:statistics.median(s[k]for s in ss)if isinstance(v,(int,float))else v for k,v in ss[0].items()}for name,ss in samples.items()}
result={'environment':{'platform':platform.platform(),'cpu':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),'repeats':a.repeats,'VECLIB_MAXIMUM_THREADS':'1','libxsmm_sha':'3057b239c16994d307589e40e45f40857ad5f5b8'},'medians':medians,'samples':samples}
(a.out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(medians,indent=2))
