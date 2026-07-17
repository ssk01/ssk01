"""
V5: DeepRec 完整流程（照着源码抄）
====================================
1. 构建 user_tower + item_tower (naive，无交叉)
2. find_boundery_tensors: 从 item_ops BFS → item_sets (consumer ops)，
   从 user_ops BFS → 找 consumer ∈ item_sets 的 boundary tensor
3. 对每个 boundary tensor，插 expand_dims→tile→reshape
4. 用 tiled tensor 继续构建 concat → top_mlp → scores
5. 导出 saved_model

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


def find_boundery_tensors(user_ops, item_ops):
    """照着 DeepRec 源码抄的 find_boundery_tensors"""
    # Step 1: BFS from item_ops → item_sets = set of all reachable consumer ops
    queue_item = deque(item_ops)
    item_sets = set()
    processed_item = set()
    while queue_item:
        op = queue_item.popleft()
        if op in processed_item:
            continue
        processed_item.add(op)
        for t in op.outputs:
            consumers = list(t.consumers())
            item_sets = item_sets | set(consumers)
            queue_item.extend(consumers)

    # Step 2: BFS from user_ops → boundary = user tensors whose consumer in item_sets
    queue_user = deque(user_ops)
    user_sets = set()
    boundery_tensor_sets = set()
    while queue_user:
        op = queue_user.popleft()
        for t in op.outputs:
            for op2 in t.consumers():
                if op2 in user_sets:
                    continue
                if op2 in item_sets:
                    boundery_tensor_sets.add(t)
                else:
                    user_sets.add(op2)
                    queue_user.append(op2)
    return user_sets, item_sets, boundery_tensor_sets


def apply_tile(tensor, item_n):
    """对 user boundary tensor 做 expand_dims→tile→reshape"""
    shape = tensor.get_shape().as_list()
    u_exp = tf.expand_dims(tensor, 1)
    tile_multiples = [tf.constant(1, tf.int32)]
    tile_multiples.append(item_n)
    for _ in range(len(shape) - 1):
        tile_multiples.append(tf.constant(1, tf.int32))
    u_tiled = tf.tile(u_exp, tf.stack(tile_multiples))
    return tf.reshape(u_tiled, tf.concat([[-1], tf.shape(tensor)[1:]], axis=0))


def main():
    print("=" * 60)
    print("V5: DeepRec — find_boundery_tensors → apply_tile → 继续构图")
    print("=" * 60)

    g = tf.Graph()
    with g.as_default():
        u1 = tf.compat.v1.placeholder(tf.float32, [None, 3], name="user_feat")
        u2 = tf.compat.v1.placeholder(tf.float32, [None, 4], name="user_context")
        i1 = tf.compat.v1.placeholder(tf.float32, [None, 3], name="item_feat")
        i2 = tf.compat.v1.placeholder(tf.float32, [None, 2], name="item_price")
        item_n = tf.compat.v1.placeholder(tf.int32, shape=[], name="item_size")

        # ==== Phase 1: 构建完整 naive 图（为了 boundary detection 有连接） ====
        with tf.compat.v1.variable_scope("ut"):
            uf = dense(u1, 128, activation=tf.nn.relu, name="f"); uf = dense(uf, 64, name="e")
            uc = dense(u2, 128, activation=tf.nn.relu, name="c"); uc = dense(uc, 64, name="ce")
            u = tf.concat([uf, uc], axis=1, name="cat")

        with tf.compat.v1.variable_scope("it"):
            inf = dense(i1, 128, activation=tf.nn.relu, name="f"); inf = dense(inf, 64, name="e")
            ipr = dense(i2, 128, activation=tf.nn.relu, name="c"); ipr = dense(ipr, 64, name="ce")
            ic = tf.concat([inf, ipr], axis=1, name="cat")

        # naive cross (用于 boundary detection 建立 user↔item 连接)
        x_naive = tf.concat([u, ic], axis=1, name="cross_naive")
        _ = dense(x_naive, 1, name="dead")  # 死路径，只为了让 graph 有连接

        # ==== Phase 2: DeepRec find_boundery_tensors ====
        print("\nPhase 2: find_boundery_tensors")
        user_sets, item_sets, boundaries = find_boundery_tensors(
            [u1.op, u2.op], [i1.op, i2.op])
        print(f"  item_sets: {len(item_sets)} consumer ops")
        print(f"  user_sets: {len(user_sets)} user ops")
        print(f"  boundaries_found: {len(boundaries)} tensors")
        for t in boundaries:
            print(f"    {t.name} → {[c.name for c in t.consumers()]}")

        # ==== Phase 3: 基于 boundary 结果，tile + 构建真实的压缩图 ====
        print("\nPhase 3: tile at boundaries → compressed graph")
        tiled_map = {}
        for t in boundaries:
            tiled_map[t] = apply_tile(t, item_n)
            print(f"  {t.name} → expand_dims→tile→reshape → {tiled_map[t].name}")

        u_tiled = tiled_map.get(u, u)

        x = tf.concat([u_tiled, ic], axis=1, name="cross")
        x = dense(x, 128, activation=tf.nn.relu, name="t1")
        x = dense(x, 64, activation=tf.nn.relu, name="t2")
        x = dense(x, 32, activation=tf.nn.relu, name="t3")
        x = dense(x, 1, name="out")
        logits = tf.squeeze(x, axis=1, name="logits")
        scores = tf.sigmoid(logits, name="scores")

        # ==== Phase 4: loss + optimizer（训练也在压缩图上跑） ====
        labels = tf.compat.v1.placeholder(tf.float32, [None], name="labels")
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits), name="loss")
        optimizer = tf.compat.v1.train.AdamOptimizer(0.001)
        train_op = optimizer.minimize(loss)

        init = tf.compat.v1.global_variables_initializer()

    # Export
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
    print(f"\n导出: {out}")

    # Train (on compressed graph)
    print("\n训练 (在压缩图上)...")
    with tf.compat.v1.Session(graph=g) as s:
        s.run(init)
        for epoch in range(3):
            total_loss = 0
            for _ in range(50):
                fd = {u1: np.random.randn(32, 3).astype(np.float32),
                      u2: np.random.randn(32, 4).astype(np.float32),
                      i1: np.random.randn(32, 3).astype(np.float32),
                      i2: np.random.randn(32, 2).astype(np.float32),
                      item_n: 1,
                      labels: (np.random.randn(32) > 0).astype(np.float32)}
                _, l = s.run([train_op, loss], fd)
                total_loss += l
            print(f"  epoch {epoch+1}: loss={total_loss/50:.4f}")

    # verify + benchmark
    with tf.compat.v1.Session(graph=g) as s:
        s.run(init)
        r = s.run(scores, {u1: np.random.randn(1,3).astype(np.float32),
                           u2: np.random.randn(1,4).astype(np.float32),
                           i1: np.random.randn(5,3).astype(np.float32),
                           i2: np.random.randn(5,2).astype(np.float32),
                           item_n: 5})
        print(f"verify: shape={r.shape} ✓")

    # Benchmark
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
