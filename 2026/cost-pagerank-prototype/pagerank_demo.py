"""Small local PageRank benchmark. macOS spawn-safe; no external datasets."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.sparse import csr_matrix


def make_graph(n=8000, degree=16, seed=42):
    rng = np.random.default_rng(seed)
    # Reserve the last vertex as a dangling node, but allow incoming edges to it.
    src = np.repeat(np.arange(n - 1), degree)
    dst = rng.integers(0, n, len(src))
    out = np.bincount(src, minlength=n)
    weights = 1.0 / out[src]
    matrix = csr_matrix((weights, (dst, src)), shape=(n, n))
    return src, dst, out, matrix


def pagerank_python(src, dst, out, iterations=20, alpha=0.85):
    n = len(out)
    edges = list(zip(src.tolist(), dst.tolist()))
    counts = out.tolist()
    rank = [1.0 / n] * n
    for _ in range(iterations):
        dangling = sum(rank[i] for i, d in enumerate(counts) if d == 0)
        nxt = [(1 - alpha) / n + alpha * dangling / n] * n
        for s, d in edges:
            nxt[d] += alpha * rank[s] / counts[s]
        rank = nxt
    return np.array(rank)


def pagerank_numpy(src, dst, out, iterations=20, alpha=0.85):
    n = len(out)
    rank = np.full(n, 1.0 / n)
    for _ in range(iterations):
        incoming = np.bincount(dst, weights=rank[src] / out[src], minlength=n)
        rank = (1 - alpha) / n + alpha * (incoming + rank[out == 0].sum() / n)
    return rank


def pagerank_sparse(matrix, out, iterations=20, alpha=0.85):
    n = len(out)
    rank = np.full(n, 1.0 / n)
    for _ in range(iterations):
        rank = (1 - alpha) / n + alpha * (matrix @ rank + rank[out == 0].sum() / n)
    return rank


def _edge_chunk(args):
    src, dst, out, rank = args
    return np.bincount(dst, weights=rank[src] / out[src], minlength=len(out))


def _ping(_):
    time.sleep(0.03)
    return os.getpid()


def parallel_trial(src, dst, out, workers, iterations=20, alpha=0.85):
    """Report startup, warm-query median, and total lifecycle separately.

    Deliberately simple IPC-heavy prototype, NOT a tuned parallel algorithm.
    Each iteration sends rank and graph chunks, then reduces dense partials.
    """
    n = len(out)
    chunks = list(zip(np.array_split(src, workers), np.array_split(dst, workers)))
    begin = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        pids = sorted(set(pool.map(_ping, range(workers * 2))))
        startup = time.perf_counter() - begin
        samples = []
        for _ in range(3):
            rank = np.full(n, 1.0 / n)
            start = time.perf_counter()
            for __ in range(iterations):
                parts = pool.map(_edge_chunk, ((s, d, out, rank) for s, d in chunks))
                incoming = np.zeros(n)
                for part in parts:
                    incoming += part
                rank = (1 - alpha) / n + alpha * (incoming + rank[out == 0].sum() / n)
            samples.append(time.perf_counter() - start)
    return {'workers': workers, 'startup_s': startup,
            'query_s': float(np.median(samples)), 'samples_s': samples,
            'lifecycle_s': time.perf_counter() - begin, 'observed_pids': pids}, rank


def timed(fn, repeats=3):
    fn()  # warm-up excluded
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples)), result, samples


if __name__ == '__main__':
    src, dst, out, matrix = make_graph()
    reference = pagerank_sparse(matrix, out)
    for workers in (1, 2, 4):
        stats, rank = parallel_trial(src, dst, out, workers)
        np.testing.assert_allclose(rank, reference, rtol=1e-11, atol=1e-13)
        print(stats)
