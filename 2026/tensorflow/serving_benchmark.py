"""
Serving Benchmark + Profile
- 加载 saved_model (等价于 TF Serving 内部执行路径)
- 模拟线上请求做压测
- tf.profiler 采集执行 trace
- 支持 Naive vs Compressed 双塔对比

用法:
    python serving_benchmark.py
"""

import os
import time
import json
import numpy as np
import tensorflow as tf

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(WORK_DIR, "saved_models", "20260714")
NAIVE_DIR = os.path.join(WORK_DIR, "saved_models", "compress_naive")
COMPRESSED_DIR = os.path.join(WORK_DIR, "saved_models", "compress_compressed")
PROFILE_DIR = os.path.join(WORK_DIR, "tb_logs", "serving_profile")

BATCH_SIZE = 128
WARMUP_BATCHES = 15
BENCH_BATCHES = 100
PROFILE_BATCHES = 30
COMPRESS_N_LIST = [10, 50, 100, 500, 1000, 2000]


def load_serving_fn():
    loaded = tf.saved_model.load(EXPORT_DIR)
    return loaded.signatures["serving_default"]


def mock_batch(batch_size=BATCH_SIZE):
    return (
        np.random.randn(batch_size, 3).astype(np.float32),
        np.random.randn(batch_size, 3).astype(np.float32),
    )


def run_batch(serve_fn, user_feat, item_feat):
    out = serve_fn(
        user_feat=tf.constant(user_feat),
        item_feat=tf.constant(item_feat),
    )
    return out[list(out.keys())[0]].numpy()


def warmup(serve_fn):
    print(f"Warmup ({WARMUP_BATCHES} batches)...")
    for _ in range(WARMUP_BATCHES):
        u, i = mock_batch()
        run_batch(serve_fn, u, i)


def benchmark(serve_fn):
    print(f"\nBenchmark ({BENCH_BATCHES} batches × {BATCH_SIZE}):")
    latencies = []
    for idx in range(BENCH_BATCHES):
        u, i = mock_batch()
        t0 = time.perf_counter()
        run_batch(serve_fn, u, i)
        latencies.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latencies)
    qps = BATCH_SIZE / (lat.mean() / 1000)
    print(f"  avg:    {lat.mean():.2f} ms")
    print(f"  p50:    {np.percentile(lat, 50):.2f} ms")
    print(f"  p90:    {np.percentile(lat, 90):.2f} ms")
    print(f"  p99:    {np.percentile(lat, 99):.2f} ms")
    print(f"  QPS:    {qps:,.0f}")
    return lat


def capture_profile(serve_fn):
    os.makedirs(PROFILE_DIR, exist_ok=True)
    print(f"\nProfile 采集 ({PROFILE_BATCHES} batches)...")

    tf.profiler.experimental.start(PROFILE_DIR)
    for _ in range(PROFILE_BATCHES):
        u, i = mock_batch()
        run_batch(serve_fn, u, i)
    tf.profiler.experimental.stop()

    print(f"  → {PROFILE_DIR}")


def analyze_profile():
    from tensorflow.tsl.profiler.protobuf import xplane_pb2

    xplane_path = None
    for root, dirs, files in os.walk(PROFILE_DIR):
        for f in files:
            if f.endswith('.xplane.pb'):
                xplane_path = os.path.join(root, f)
                break
    if not xplane_path:
        print("  未找到 xplane.pb")
        return

    xspace = xplane_pb2.XSpace()
    with open(xplane_path, 'rb') as f:
        xspace.ParseFromString(f.read())

    for plane in xspace.planes:
        if 'Task' in plane.name:
            continue
        meta_dict = {}
        for k in plane.event_metadata:
            meta_dict[k] = plane.event_metadata[k].name

        events = []
        for line in plane.lines:
            for ev in line.events:
                dur = ev.duration_ps
                if dur > 0:
                    events.append((line.name or '', dur, ev.metadata_id))

        events.sort(key=lambda x: -x[1])

        print(f"\n  [{plane.name}]  {len(events)} events, {sum(e[1] for e in events)/1e6:.1f} us total")

        # 按类别聚合
        from collections import defaultdict
        cat_totals = defaultdict(float)
        for line_name, dur, meta_id in events:
            name = meta_dict.get(meta_id, '')
            cat = name.split(':')[0] if ':' in name else name.split('/')[0] if '/' in name else name
            cat_totals[cat] += dur

        total = sum(cat_totals.values())
        print(f"\n  {'Category':<40} {'time_us':>8} {'pct':>6}")
        print(f"  {'-'*55}")
        for cat, dur in sorted(cat_totals.items(), key=lambda x: -x[1])[:12]:
            print(f"  {cat:<40} {dur/1e6:>8.1f} {dur/total*100:>5.1f}%")

        # Top 单个事件
        print(f"\n  Top 10 longest events:")
        print(f"  {'event':<55} {'dur_us':>8}")
        print(f"  {'-'*65}")
        for line_name, dur, meta_id in events[:10]:
            name = meta_dict.get(meta_id, f'id:{meta_id}')
            print(f"  {name:<55} {dur/1e6:>8.1f}")


def export_json():
    from tensorflow.core.protobuf import saved_model_pb2
    from google.protobuf import json_format
    sm = saved_model_pb2.SavedModel()
    pb_path = os.path.join(EXPORT_DIR, "saved_model.pb")
    with open(pb_path, "rb") as f:
        sm.ParseFromString(f.read())
    json_path = os.path.join(WORK_DIR, "saved_model.json")
    mg = sm.meta_graphs[0]
    d = {
        "tags": list(mg.meta_info_def.tags),
        "signatures": {},
        "graph_stats": {
            "node_count": len(mg.graph_def.node),
            "library_functions": [f.signature.name for f in mg.graph_def.library.function],
        },
    }
    for name, sig in mg.signature_def.items():
        d["signatures"][name] = json_format.MessageToDict(sig)
    with open(json_path, "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    return json_path


def main():
    print("=" * 60)
    print("Serving Benchmark + Profile")
    print(f"Model:  {EXPORT_DIR}")
    print(f"Batch:  {BATCH_SIZE}")
    print("=" * 60)

    serve_fn = load_serving_fn()
    warmup(serve_fn)
    benchmark(serve_fn)
    capture_profile(serve_fn)

    print("\n" + "=" * 60)
    print("Naive vs Compressed 对比 (saved_model)")
    print("=" * 60)

    naive_fn = tf.saved_model.load(NAIVE_DIR).signatures["serving_default"]
    comp_fn = tf.saved_model.load(COMPRESSED_DIR).signatures["serving_default"]

    print(f"\n{'N':>6} | {'Naive(ms)':>10} {'P99':>8} | {'Comp(ms)':>10} {'P99':>8} | {'Speedup':>7}")
    print("-" * 65)
    for N in COMPRESS_N_LIST:
        u1 = np.random.randn(1, 3).astype(np.float32)
        items = np.random.randn(N, 3).astype(np.float32)
        ut_n = tf.constant(np.tile(u1, (N, 1)))
        ut_c = tf.constant(u1)
        it = tf.constant(items)

        for _ in range(50):
            naive_fn(user_feat=ut_n, item_feat=it)
            comp_fn(user_feat=ut_c, item_feat=it)

        l_n, l_c = [], []
        for _ in range(200):
            t0 = time.perf_counter(); naive_fn(user_feat=ut_n, item_feat=it); l_n.append((time.perf_counter()-t0)*1000)
            t0 = time.perf_counter(); comp_fn(user_feat=ut_c, item_feat=it); l_c.append((time.perf_counter()-t0)*1000)

        l_n, l_c = np.array(l_n), np.array(l_c)
        print(f'{N:>6} | {l_n.mean():>10.3f} {np.percentile(l_n,99):>8.3f} | {l_c.mean():>10.3f} {np.percentile(l_c,99):>8.3f} | {l_n.mean()/l_c.mean():>6.2f}x')
    print()


if __name__ == "__main__":
    main()
