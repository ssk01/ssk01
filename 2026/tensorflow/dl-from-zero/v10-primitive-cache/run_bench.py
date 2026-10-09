#!/usr/bin/env python3
"""Five isolated processes per case; alternating case order across repeats."""
import json, os, statistics, subprocess, sys
exe=sys.argv[1]
cases=[(mode,backend,cap,work) for mode in ('primitive','tf') for work in ('fixed','mixed') for backend,cap in [('jit',0),('jit',2),('jit',8),('accelerate',0)]]
runs={}
for repeat in range(5):
    for case in (cases if repeat%2==0 else list(reversed(cases))):
        mode,backend,cap,work=case
        result=subprocess.check_output([exe,mode,backend,str(cap),'20',work],env={**os.environ,'VECLIB_MAXIMUM_THREADS':'1'},text=True)
        runs.setdefault('/'.join(map(str,case)),[]).append(json.loads(result))
medians={key:{field:statistics.median(row[field] for row in rows) for field in ('hits','misses','generated','generated_bytes','prepare_us','execute_us','total_us','max_error')} for key,rows in runs.items()}
print(json.dumps({'repeats':5,'threads':1,'runs':runs,'medians':medians},indent=2))
