"""Run controlled in-memory layout and connected-component experiments."""
from paper_algorithms import *
import json,platform
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def reference_components(src,dst,n):
    graph=csr_matrix((np.ones(len(src)),(src,dst)),shape=(n,n))
    _,labels=connected_components(graph,directed=False)
    return canonical(labels)


def main():
    root=Path(__file__).resolve().parent
    # Compile before any measured preprocessing.
    hilbert_keys(np.array([0],np.uint32),np.array([1],np.uint32))
    cases=[]
    for n in (8192,262144,1048576):
        rng=np.random.default_rng(42)
        src=np.repeat(np.arange(n,dtype=np.uint32),8)
        dst=rng.integers(n,size=len(src),dtype=np.uint32)
        start=time.perf_counter(); keys=hilbert_keys(src,dst); encoding=time.perf_counter()-start
        start=time.perf_counter(); order=np.argsort(keys,kind='stable'); sorting=time.perf_counter()-start
        start=time.perf_counter(); hs=np.ascontiguousarray(src[order]);hd=np.ascontiguousarray(dst[order]);packing=time.perf_counter()-start
        results={'vertex':[],'hilbert':[]}
        paper_pagerank(src,dst,n);paper_pagerank(hs,hd,n)
        # Alternate order to reduce consistent warm-cache/frequency advantages.
        for rep in range(5):
            for label,s,d in ([('vertex',src,dst),('hilbert',hs,hd)] if rep%2==0 else [('hilbert',hs,hd),('vertex',src,dst)]):
                start=time.perf_counter();r=paper_pagerank(s,d,n);results[label].append(time.perf_counter()-start)
        vr=paper_pagerank(src,dst,n);hr=paper_pagerank(hs,hd,n)
        np.testing.assert_allclose(vr,hr,rtol=3e-6,atol=2e-6)
        # Independent float64 sparse formulation with exactly the paper's initialization/reset.
        mat=csr_matrix((np.full(len(src),1/8),(dst,src)),shape=(n,n))
        ref=np.zeros(n)
        alpha=float(np.float32(.85));reset=float(np.float32(1)-np.float32(.85))
        for _ in range(20): ref=reset+alpha*(mat@ref)
        np.testing.assert_allclose(vr,ref,rtol=3e-6,atol=2e-6)
        case={'nodes':n,'edges':len(src),'encoding_s':encoding,'sorting_s':sorting,'repacking_s':packing,
              'preprocess_s':encoding+sorting+packing,'max_abs_rank_difference':float(abs(vr-hr).max()),
              **{k:{'samples_s':v,'median_s':float(np.median(v))} for k,v in results.items()}}
        cases.append(case);print('LAYOUT',case,flush=True)
    (root/'layout_results.json').write_text(json.dumps({'platform':platform.platform(),'cases':cases},indent=2))
    # Correctness corner cases: isolates, duplicates, self-loops, disconnected groups.
    for n,s,d in [(5,[0,0,1,3],[1,1,1,3]),(4,[],[]),(1,[0],[0])]:
        s=np.array(s,dtype=np.int64);d=np.array(d,dtype=np.int64)
        ref=reference_components(s,d,n)
        np.testing.assert_array_equal(label_propagation(s,d,n)[0],ref)
        np.testing.assert_array_equal(union_find(s,d,n)[0],ref)
        for p in (1,2,4):np.testing.assert_array_equal(parallel_labels(s,d,n,p,repeats=1)[0],ref)
    cc=[]
    n=8192
    s=np.arange(n-1,dtype=np.int64);d=s+1
    rng=np.random.default_rng(42)
    for name,order in [('path_forward',np.arange(len(s))),('path_reverse',np.arange(len(s))[::-1]),('path_shuffled',rng.permutation(len(s)))]:
        src=np.ascontiguousarray(s[order]);dst=np.ascontiguousarray(d[order])
        row={'name':name,'nodes':n,'edges':len(src)}
        for label,fn in [('label_propagation',label_propagation),('union_find',union_find)]:
            result,stats=measure(lambda:fn(src,dst,n),repeats=3)
            np.testing.assert_array_equal(result[0],np.zeros(n,dtype=np.int64))
            row[label]={**stats,'scans':int(result[1]),'edge_visits':int(result[2])}
        cc.append(row);print('CC',row,flush=True)
    n=60000;src,dst=components_graph(n=n)
    ref=reference_components(src,dst,n)
    row={'name':'20_components_shortcuts','nodes':n,'edges':len(src)}
    for label,fn in [('label_propagation',label_propagation),('union_find',union_find)]:
        result,stats=measure(lambda:fn(src,dst,n))
        np.testing.assert_array_equal(result[0],ref)
        row[label]={**stats,'scans':int(result[1]),'edge_visits':int(result[2])}
    row['parallel']=[]
    for p in (1,2,4):
        labels,stats=parallel_labels(src,dst,n,p)
        np.testing.assert_array_equal(labels,ref)
        row['parallel'].append(stats)
    cc.append(row);print('CC',row,flush=True)
    (root/'components_results.json').write_text(json.dumps({'platform':platform.platform(),'cases':cc,'checks':'scipy reference + empty/isolated/duplicate/self-loop cases passed'},indent=2))

if __name__=='__main__': main()
