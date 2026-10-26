"""Controlled local PageRank ablations; timings are warm-query medians."""
from pagerank_demo import *
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import platform
from pathlib import Path

STATE = {}

def buffer(a):
    ctype = ctypes.c_double if a.dtype == np.float64 else ctypes.c_int64
    b = mp.RawArray(ctype, int(a.size))
    np.frombuffer(b, dtype=a.dtype)[:] = a.ravel()
    return b


def init_shared(src, dst, out, rank, result, n, p):
    STATE.update(src=np.frombuffer(src, dtype=np.int64), dst=np.frombuffer(dst, dtype=np.int64),
                 out=np.frombuffer(out, dtype=np.int64), rank=np.frombuffer(rank, dtype=np.float64),
                 result=np.frombuffer(result, dtype=np.float64).reshape(p,n), n=n, p=p)


def worker(args):
    i, mode, supplied_rank = args
    s = STATE
    lo = len(s['src'])*i//s['p']; hi = len(s['src'])*(i+1)//s['p']
    src, dst = s['src'][lo:hi], s['dst'][lo:hi]
    rank = s['rank'] if supplied_rank is None else supplied_rank
    start = time.perf_counter()
    result = np.bincount(dst, weights=rank[src]/s['out'][src], minlength=s['n'])
    compute = time.perf_counter()-start
    if mode == 'shared':
        s['result'][i] = result
        return compute
    return result, compute


def naive_profile(args):
    start=time.perf_counter()
    result=_local_chunk(args)
    return result, time.perf_counter()-start


def _local_chunk(args):
    src,dst,out,rank=args
    return np.bincount(dst,weights=rank[src]/out[src],minlength=len(out))


def benchmark(src,dst,out,matrix,p,mode,repeats=5,iterations=20):
    n=len(out)
    chunks=list(zip(np.array_split(src,p),np.array_split(dst,p)))
    start=time.perf_counter()
    if mode == 'csr_threads':
        blocks=[matrix[n*i//p:n*(i+1)//p] for i in range(p)]
        executor=ThreadPoolExecutor(max_workers=p)
    else:
        rb=buffer(np.full(n,1/n)); ob=buffer(np.zeros((p,n)))
        sb,db,deg=buffer(src),buffer(dst),buffer(out)
        shared_rank=np.frombuffer(rb,dtype=np.float64)
        shared_result=np.frombuffer(ob,dtype=np.float64).reshape(p,n)
        executor=ProcessPoolExecutor(max_workers=p,mp_context=mp.get_context('spawn'),
            initializer=init_shared,initargs=(sb,db,deg,rb,ob,n,p))
    timings=[]; profiles=[]
    with executor as pool:
        for trial in range(repeats+1):
            rank=np.full(n,1/n)
            wait_s=reduce_s=update_s=compute_critical_s=0.
            begin=time.perf_counter()
            for _ in range(iterations):
                t=time.perf_counter()
                if mode == 'naive':
                    parts=list(pool.map(naive_profile,((s,d,out,rank) for s,d in chunks)))
                elif mode == 'csr_threads':
                    parts=list(pool.map(lambda b: b @ rank,blocks))
                else:
                    if mode=='shared': shared_rank[:]=rank
                    parts=list(pool.map(worker,((i,mode,None if mode=='shared' else rank) for i in range(p))))
                wait_s+=time.perf_counter()-t
                t=time.perf_counter()
                if mode == 'csr_threads': incoming=np.concatenate(parts)
                elif mode == 'shared':
                    incoming=shared_result.sum(axis=0)
                    compute_critical_s+=max(parts)
                else:
                    incoming=np.zeros(n)
                    for values,elapsed in parts: incoming+=values
                    compute_critical_s+=max(x[1] for x in parts)
                reduce_s+=time.perf_counter()-t
                t=time.perf_counter()
                rank=.15/n+.85*(incoming+rank[out==0].sum()/n)
                update_s+=time.perf_counter()-t
            elapsed=time.perf_counter()-begin
            if trial==0: setup_warm=time.perf_counter()-start
            else:
                timings.append(elapsed)
                profiles.append({'dispatch_wait_s':wait_s,'reduce_s':reduce_s,'update_s':update_s,
                                 'max_worker_compute_sum_s':compute_critical_s})
    np.testing.assert_allclose(rank,pagerank_sparse(matrix,out,iterations),rtol=1e-11,atol=1e-13)
    return {'mode':mode,'workers':p,'median_s':float(np.median(timings)),'samples_s':timings,
            'setup_and_warmup_s':setup_warm,'profile_median':{k:float(np.median([x[k] for x in profiles])) for k in profiles[0]}}


def main():
    all_results=[]
    for n in (8000,80000,400000):
        src,dst,out,matrix=make_graph(n,16)
        row={'nodes':n,'edges':len(src),'iterations':20,'runs':[]}
        for label,fn in [('numpy',lambda:pagerank_numpy(src,dst,out)),('csr',lambda:pagerank_sparse(matrix,out))]:
            elapsed,rank,samples=timed(fn,repeats=5)
            row[label]={'median_s':elapsed,'samples_s':samples}
        print('SIZE',n,'BASELINES',row['numpy']['median_s'],row['csr']['median_s'],flush=True)
        for mode in ('naive','resident','shared','csr_threads'):
            for p in (1,2,4):
                r=benchmark(src,dst,out,matrix,p,mode)
                row['runs'].append(r)
                print(mode,p,round(r['median_s']*1000,3),'ms',flush=True)
        all_results.append(row)
        Path(__file__).with_name('diagnosis_results.json').write_text(json.dumps({'platform':platform.platform(),
          'numpy':np.__version__,'cases':all_results},indent=2))

if __name__=='__main__': main()
