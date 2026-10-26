"""Single-threaded teaching ports of COST algorithms, plus one parallel control.

Sources: frankmcsherry/COST at d5bef75f976ee5880811305621a2883d499f3e65.
See references/author-cost/LICENSE and author_sources.md. These are teaching
ports, not the original benchmark: in-memory arrays, Numba, no huge pages/I/O.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from numba import njit


@njit(cache=True)
def hilbert_keys(src, dst):
    """32-bit coordinates -> 64-bit Hilbert distances, matching author's bit code.

    Reflection/rotation at each scale distinguishes this from Morton bit interleave.
    """
    keys = np.empty(len(src), np.uint64)
    for k in range(len(src)):
        x, y = np.int64(src[k]), np.int64(dst[k])
        d = np.uint64(0)
        for s in range(31, -1, -1):
            rx, ry = (x >> s) & 1, (y >> s) & 1
            d += np.uint64((3 * rx) ^ ry) << np.uint64(2 * s)
            if ry == 0:
                if rx:
                    x, y = ((1 << s) - 1 - y) & 0xffffffff, ((1 << s) - 1 - x) & 0xffffffff
                else:
                    x, y = y, x
        keys[k] = d
    return keys


@njit(cache=True)
def paper_pagerank(src, dst, n, iterations=20):
    """Matches paper's zero initialization/reset=0.15 convention (not sum=1).

    One contribution per source per round, same compiled kernel for both orders.
    Test graphs have positive outdegree for every vertex, avoiding dangling ambiguity.
    """
    a = np.zeros(n, np.float32)
    b = np.zeros(n, np.float32)
    degree = np.zeros(n, np.float32)
    alpha = np.float32(.85)
    for e in range(len(src)):
        degree[src[e]] += np.float32(1)
    for _ in range(iterations):
        for v in range(n):
            b[v] = alpha * a[v] / degree[v] if degree[v] else np.float32(0)
            a[v] = np.float32(1) - alpha
        for e in range(len(src)):
            a[dst[e]] += b[src[e]]
    return a


@njit(cache=True)
def label_propagation(src, dst, n):
    """Paper's in-place (asynchronous) full scans until no label changes."""
    label = np.arange(n, dtype=np.int64)
    scans = 0
    changed = True
    while changed:
        changed = False
        scans += 1
        for i in range(len(src)):
            u, v = src[i], dst[i]
            if label[u] != label[v]:
                value = min(label[u], label[v])
                label[u], label[v] = value, value
                changed = True
    return label, scans, scans * len(src)


@njit(cache=True)
def union_find(src, dst, n):
    """Weighted union by rank, no path compression, as in the paper.

    Canonical per-vertex labels are included in runtime, unlike original count-only CLI.
    """
    root = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.uint8)
    steps = 0
    for i in range(len(src)):
        x, y = src[i], dst[i]
        while x != root[x]:
            x = root[x]; steps += 1
        while y != root[y]:
            y = root[y]; steps += 1
        if x != y:
            if rank[x] < rank[y]: root[x] = y
            elif rank[x] > rank[y]: root[y] = x
            else:
                root[y] = x
                rank[x] += 1
    labels = np.empty(n, dtype=np.int64)
    minimum = np.full(n, n, dtype=np.int64)
    for i in range(n):
        r = i
        while r != root[r]:
            r = root[r]; steps += 1
        labels[i] = r
        minimum[r] = min(minimum[r], i)
    for i in range(n): labels[i] = minimum[labels[i]]
    return labels, 1, len(src), steps


@njit(cache=True, nogil=True)
def sync_piece(src, dst, old):
    """Private output; old labels immutable during a round. Safe for threads."""
    result = old.copy()
    for i in range(len(src)):
        u, v = src[i], dst[i]
        result[u] = min(result[u], old[v])
        result[v] = min(result[v], old[u])
    return result


def parallel_labels(src, dst, n, workers=1, repeats=5):
    """Synchronous full scans: all kernels compiled; no Python-edge-loop baseline."""
    chunks = [(np.ascontiguousarray(s), np.ascontiguousarray(d))
              for s, d in zip(np.array_split(src, workers), np.array_split(dst, workers))]
    sync_piece(chunks[0][0], chunks[0][1], np.arange(n, dtype=np.int64))
    samples=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rep in range(repeats+1):
            start=time.perf_counter()
            label=np.arange(n, dtype=np.int64)
            scans=0
            while True:
                parts=list(pool.map(lambda edge: sync_piece(edge[0],edge[1],label),chunks))
                new=parts[0]
                for other in parts[1:]: np.minimum(new,other,out=new)
                scans+=1
                done=np.array_equal(label,new)
                label=new
                if done: break
            elapsed=time.perf_counter()-start
            if rep: samples.append(elapsed)
    return label, {'workers':workers,'median_s':float(np.median(samples)),
                   'samples_s':samples,'scans':scans,'edge_visits':scans*len(src)}


def components_graph(n=60000, groups=20, seed=42):
    """Separate components, each connected via a backbone plus random shortcuts."""
    assert n % groups == 0
    rng=np.random.default_rng(seed)
    width=n//groups
    s=[]; d=[]
    for g in range(groups):
        start=g*width
        s.extend(range(start,start+width-1));d.extend(range(start+1,start+width))
        s.extend((rng.integers(width,size=width*4)+start).tolist())
        d.extend((rng.integers(width,size=width*4)+start).tolist())
    permutation=rng.permutation(n)
    src=permutation[np.array(s)]
    dst=permutation[np.array(d)]
    order=rng.permutation(len(src))
    return np.ascontiguousarray(src[order]),np.ascontiguousarray(dst[order])


def measure(fn,repeats=5):
    fn()  # JIT and cache warmup excluded
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter(); result=fn();samples.append(time.perf_counter()-start)
    return result, {'median_s':float(np.median(samples)),'samples_s':samples}


def canonical(labels):
    labels=np.asarray(labels)
    minimum=np.full(len(labels),len(labels),dtype=np.int64)
    np.minimum.at(minimum,labels,np.arange(len(labels)))
    return minimum[labels]
