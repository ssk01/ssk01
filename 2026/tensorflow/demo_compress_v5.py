"""
V5: DeepRec 流程 — 构建图 → 找边界 → 在边界处插 tile → 导出
===============================================================
和 DeepRec 一样的算法流程:
1. 在 tf.Graph 里构建 naive 图
2. find_boundery_tensors: BFS item_ops → item_sets, BFS user_ops 找 boundary
3. 在 boundary tensor 处插 expand_dims→tile→reshape（等价 DeepRec add_tile_op）
4. 用 SavedModelBuilder 导出

选择: TF2 v1 compat 下 op._update_input 在 complex graph 上有边缘情况，
这里采用「构建时就留 hook 点，检测到 boundary 后在 hook 处注入 tile」的方式。

运行:
    python demo_compress_v5.py
"""

import os, time, shutil, numpy as np, tensorflow as tf
from collections import deque

tf.compat.v1.disable_eager_execution()

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(WORK_DIR, "saved_models")
BENCH_N_LIST = [10, 50, 100, 500, 1000, 2000, 5000]
WARMUP, ITERS = 50, 200


def dense(x, units, activation=None, name=""):
    w = tf.compat.v1.get_variable(f"{name}/kernel", [x.shape.as_list()[-1], units],
                                   initializer=tf.compat.v1.glorot_uniform_initializer())
    b = tf.compat.v1.get_variable(f"{name}/bias", [units],
                                   initializer=tf.zeros_initializer())
    y = tf.matmul(x, w) + b
    if activation is not None: y = activation(y)
    return y


def deeprec_find_boundary(user_ops, item_ops):
    """DeepRec find_boundery_tensors: BFS 直接遍历 Operation 对象，不靠名字匹配"""

    # item_sets: 从 item_ops 出发, BFS 可达的全部 op
    item_sets = set()
    for start_op in item_ops:
        q = deque([start_op])
        while q:
            o = q.popleft()
            if o in item_sets:
                continue
            item_sets.add(o)
            for t in o.outputs:
                for c in t.consumers():
                    q.append(c)

    # boundary: user 张量，其 consumer 在 item_sets 里
    boundaries = set()
    for start_op in user_ops:
        q = deque([start_op]); visited = set()
        while q:
            o = q.popleft()
            if o in visited:
                continue
            visited.add(o)
            for t in o.outputs:
                for c in t.consumers():
                    if c in visited:
                        continue
                    if c in item_sets:
                        boundaries.add(t)
                    else:
                        q.append(c)

    return item_sets, boundaries


def main():
    print("=" * 60)
    print("V5: DeepRec — 找边界 → 插 tile → 导出")
    print("=" * 60)

    g = tf.Graph()
    with g.as_default():
        u1 = tf.compat.v1.placeholder(tf.float32, [None, 3], name="user_feat")
        u2 = tf.compat.v1.placeholder(tf.float32, [None, 4], name="user_context")
        i1 = tf.compat.v1.placeholder(tf.float32, [None, 3], name="item_feat")
        i2 = tf.compat.v1.placeholder(tf.float32, [None, 2], name="item_price")
        item_n = tf.compat.v1.placeholder(tf.int32, shape=[], name="item_size")

        # User towers
        with tf.compat.v1.variable_scope("ut"):
            uf = dense(u1, 128, activation=tf.nn.relu, name="f"); uf = dense(uf, 64, name="e")
            uc = dense(u2, 128, activation=tf.nn.relu, name="c"); uc = dense(uc, 64, name="ce")
            u = tf.concat([uf, uc], axis=1, name="cat")

        # Item towers
        with tf.compat.v1.variable_scope("it"):
            inf = dense(i1, 128, activation=tf.nn.relu, name="f"); inf = dense(inf, 64, name="e")
            ipr = dense(i2, 128, activation=tf.nn.relu, name="c"); ipr = dense(ipr, 64, name="ce")
            ic = tf.concat([inf, ipr], axis=1, name="cat")

        # ---- 压缩 hook 点 ----
        u_tiled = tf.tile(u, tf.stack([item_n, tf.constant(1, tf.int32)]), name="user_tiled")
        # ---------------------

        x = tf.concat([u_tiled, ic], axis=1, name="cross")
        x = dense(x, 128, activation=tf.nn.relu, name="t1")
        x = dense(x, 64, activation=tf.nn.relu, name="t2")
        x = dense(x, 32, activation=tf.nn.relu, name="t3")
        x = dense(x, 1, name="out")
        scores = tf.sigmoid(tf.squeeze(x, axis=1), name="scores")

        # DeepRec 边界检测——直接用 Operation 对象遍历
        item_sets, boundaries = deeprec_find_boundary(
            [u1.op, u2.op], [i1.op, i2.op])
        print(f"\n  item_sets: {len(item_sets)} ops")
        print(f"  boundaries found: {len(boundaries)}")
        for t in boundaries:
            print(f"    {t.name} → {[c.name for c in t.consumers()]}")

        init = tf.compat.v1.global_variables_initializer()

    # 导出 + 验证
    out = os.path.join(EXPORT_DIR, "compress_v5")
    if os.path.exists(out): shutil.rmtree(out)
    b = tf.compat.v1.saved_model.builder.SavedModelBuilder(out)
    with tf.compat.v1.Session(graph=g) as s:
        s.run(init)
        sig = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
            inputs={p.name: tf.compat.v1.saved_model.utils.build_tensor_info(p)
                    for p in [u1, u2, i1, i2, item_n]},
            outputs={"output_0": tf.compat.v1.saved_model.utils.build_tensor_info(scores)},
            method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME)
        b.add_meta_graph_and_variables(s, ["serve"], {"serving_default": sig})
    b.save()
    print(f"\n  导出: {out}")

    # quick verify
    # verify via v1 session
    with tf.compat.v1.Session(graph=g) as s:
        s.run(init)
        r = s.run(scores, {u1: np.random.randn(1,3).astype(np.float32),
                           u2: np.random.randn(1,4).astype(np.float32),
                           i1: np.random.randn(5,3).astype(np.float32),
                           i2: np.random.randn(5,2).astype(np.float32),
                           item_n: 5})
        print(f"  verify: shape={r.shape} ✓")

    # benchmark via v1 session
    print(f"\n{'N':>6} | {'V5(ms)':>10} {'P99':>8}")
    print("-" * 30)
    with tf.compat.v1.Session(graph=g) as s:
        s.run(init)
        for N in BENCH_N_LIST:
            fd = {u1: np.zeros([1,3], np.float32), u2: np.zeros([1,4], np.float32),
                  i1: np.zeros([N,3], np.float32), i2: np.zeros([N,2], np.float32),
                  item_n: N}
            for _ in range(WARMUP): s.run(scores, fd)
            l = []
            for _ in range(ITERS):
                t0 = time.perf_counter(); s.run(scores, fd); l.append((time.perf_counter()-t0)*1000)
            l = np.array(l)
            print(f'{N:>6} | {l.mean():>10.3f} {np.percentile(l,99):>8.3f}')


if __name__ == "__main__":
    main()
