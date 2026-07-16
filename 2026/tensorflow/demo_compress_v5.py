"""
V5: tf.Graph 层面 DeepRec 风格压缩
====================================
在构建 tf.Graph 时，用 BFS 找到 user/item 的 boundary tensor，
在 boundary 处插 tile ops（expand_dims→tile→reshape），
整个压缩逻辑作为 graph 的一部分被导出。

运行:
    python demo_compress_v5.py
"""

import os
import time
import numpy as np
import tensorflow as tf
from collections import deque

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(WORK_DIR, "saved_models")

BENCH_N_LIST = [10, 50, 100, 500, 1000, 2000, 5000]
BENCH_WARMUP = 50
BENCH_ITERS = 200


def dense(x, units, activation=None, name=""):
    w = tf.compat.v1.get_variable(f"{name}/kernel", [x.shape[-1], units],
                                   initializer=tf.compat.v1.glorot_uniform_initializer())
    b = tf.compat.v1.get_variable(f"{name}/bias", [units],
                                   initializer=tf.zeros_initializer())
    y = tf.matmul(x, w) + b
    if activation is not None:
        y = activation(y)
    return y


def deeprec_find_boundary(ops_list, user_op_names, item_op_names):
    """在已构建的 graph ops 上执行 DeepRec 的 find_boundery_tensors 算法"""
    name_to_op = {op.name: op for op in ops_list}
    user_ops = [name_to_op[n] for n in user_op_names]
    item_ops = [name_to_op[n] for n in item_op_names]

    item_sets = set()
    q = deque(item_ops)
    processed = set()
    while q:
        op = q.popleft()
        if op in processed:
            continue
        processed.add(op)
        for t in op.outputs:
            for c in t.consumers():
                item_sets.add(c)
                q.append(c)

    user_sets = set()
    boundaries = []
    q = deque(user_ops)
    while q:
        op = q.popleft()
        for t in op.outputs:
            for c in t.consumers():
                if c in user_sets:
                    continue
                if c in item_sets:
                    boundaries.append(t)
                else:
                    user_sets.add(c)
                    q.append(c)

    return item_sets, user_sets, boundaries


def main():
    print("=" * 60)
    print("V5: tf.Graph 构建期 DeepRec 风格压缩")
    print("=" * 60)

    g = tf.Graph()
    with g.as_default():
        user_ph = tf.compat.v1.placeholder(tf.float32, [None, 3], name="user_feat")
        item_ph = tf.compat.v1.placeholder(tf.float32, [None, 3], name="item_feat")
        item_size = tf.compat.v1.placeholder(tf.int32, shape=[], name="item_size")

        # 构建压缩版图（tile 在图构建时就插进去）
        with tf.compat.v1.variable_scope("user_tower"):
            u = dense(user_ph, 256, activation=tf.nn.relu, name="user_d1")
            u = dense(u, 128, activation=tf.nn.relu, name="user_d2")
            u = dense(u, 64, name="user_emb")

        with tf.compat.v1.variable_scope("item_tower"):
            i = dense(item_ph, 256, activation=tf.nn.relu, name="item_d1")
            i = dense(i, 128, activation=tf.nn.relu, name="item_d2")
            i = dense(i, 64, name="item_emb")

        # user 只来一份 [1, 64]，tile 到 [N, 64]
        u_tiled = tf.tile(u, tf.stack([item_size, tf.constant(1, tf.int32)]), name="user_tiled")

        x = tf.concat([u_tiled, i], axis=1, name="concat")
        x = dense(x, 128, activation=tf.nn.relu, name="top_d1")
        x = dense(x, 64, activation=tf.nn.relu, name="top_d2")
        x = dense(x, 32, activation=tf.nn.relu, name="top_d3")
        logit = dense(x, 1, name="top_out")
        scores = tf.sigmoid(tf.squeeze(logit, axis=1), name="scores")

    # 验证 DeepRec 边界检测算法（在已构建的 graph 上跑）
    print("\nDeepRec find_boundery_tensors 算法验证:")
    ops = g.get_operations()
    user_op = g.get_operation_by_name("user_feat")
    item_op = g.get_operation_by_name("item_feat")
    item_sets, user_sets, boundaries = deeprec_find_boundary(
        ops, ["user_feat"], ["item_feat"])

    print(f"  item_sets: {len(item_sets)} ops (从 item_ph 可达的全部 op)")
    print(f"  user_sets: {len(user_sets)} ops (纯 user 路径)")
    print(f"  boundaries: {len(boundaries)} tensors (user 输出被 item 子图消费)")
    for t in boundaries:
        consumers = [c.name for c in t.consumers()]
        print(f"    {t.name} → {consumers}")

    # 导出
    print("\n导出 saved_model...")
    export_path = os.path.join(EXPORT_DIR, "compress_v5")
    with g.as_default():
        init = tf.compat.v1.global_variables_initializer()
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_path)
    with tf.compat.v1.Session(graph=g) as sess:
        sess.run(init)
        ti_u = tf.compat.v1.saved_model.utils.build_tensor_info(user_ph)
        ti_i = tf.compat.v1.saved_model.utils.build_tensor_info(item_ph)
        ti_s = tf.compat.v1.saved_model.utils.build_tensor_info(item_size)
        ti_o = tf.compat.v1.saved_model.utils.build_tensor_info(scores)
        sig = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
            inputs={"user_feat": ti_u, "item_feat": ti_i, "item_size": ti_s},
            outputs={"output_0": ti_o},
            method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME)
        builder.add_meta_graph_and_variables(
            sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
            signature_def_map={"serving_default": sig})
    builder.save()
    print(f"  → {export_path}")

    # Benchmark
    comp_fn = tf.saved_model.load(export_path).signatures["serving_default"]
    naive_fn = tf.saved_model.load(
        os.path.join(EXPORT_DIR, "compress_naive")).signatures["serving_default"]

    print(f"\n{'N':>6} | {'Naive(ms)':>10} {'P99':>8} | {'V5(ms)':>10} {'P99':>8} | {'Speedup':>7}")
    print("-" * 65)
    for N in BENCH_N_LIST:
        u1 = np.random.randn(1, 3).astype(np.float32)
        items = np.random.randn(N, 3).astype(np.float32)
        ut_n = tf.constant(np.tile(u1, (N, 1)))
        ut_c = tf.constant(u1)
        it = tf.constant(items)
        sz = tf.constant(N, dtype=tf.int32)

        for _ in range(BENCH_WARMUP):
            comp_fn(user_feat=ut_c, item_feat=it, item_size=sz)
            naive_fn(user_feat=ut_n, item_feat=it)

        l_n, l_c = [], []
        for _ in range(BENCH_ITERS):
            t0 = time.perf_counter(); naive_fn(user_feat=ut_n, item_feat=it); l_n.append((time.perf_counter()-t0)*1000)
            t0 = time.perf_counter(); comp_fn(user_feat=ut_c, item_feat=it, item_size=sz); l_c.append((time.perf_counter()-t0)*1000)
        l_n, l_c = np.array(l_n), np.array(l_c)
        print(f'{N:>6} | {l_n.mean():>10.3f} {np.percentile(l_n,99):>8.3f} | {l_c.mean():>10.3f} {np.percentile(l_c,99):>8.3f} | {l_n.mean()/l_c.mean():>6.2f}x')


if __name__ == "__main__":
    main()
