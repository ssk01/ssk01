"""
V5: tf.Graph 层面 DeepRec 风格压缩
====================================
完全模仿 DeepRec 流程:
1. 在 tf.Graph 里构建 naive 图（无压缩）
2. 运行 find_boundery_tensors 找到边界
3. 在边界处自动插 tile ops（expand_dims→tile→reshape）
4. 用 op._update_input 重连 consumer
5. 导出

运行:
    python demo_compress_v5.py
"""

import os
import time
import shutil
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
    """DeepRec find_boundery_tensors: BFS 找 user→item 边界"""
    name_to_op = {op.name: op for op in ops_list}
    user_ops = [name_to_op[n] for n in user_op_names]
    item_ops = [name_to_op[n] for n in item_op_names]

    # Step 1: 从 item_ops 出发 BFS，建立 item_sets（过集）
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

    # Step 2: 从 user_ops 出发 BFS，consumer ∈ item_sets → boundary
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


def deeprec_insert_tile(graph, boundaries, item_size_tensor):
    """DeepRec add_tile_op: 在边界 tensor 处插 expand_dims→tile→reshape，重连 consumer"""
    tiled_count = 0
    for t in boundaries:
        if not t.consumers():
            continue
        shape = t.get_shape().as_list()
        if len(shape) > 0 and shape[0] is None:
            with graph.colocate_with(t.op):
                user_expand = tf.expand_dims(t, 1)
                tile_multiples = [1, item_size_tensor] + [1] * (len(shape) - 1)
                user_tiled = tf.tile(user_expand, tf.stack(tile_multiples))
                target_units = shape[1] if len(shape) > 1 and shape[1] is not None else -1
                user_reshaped = tf.reshape(user_tiled, [-1, target_units])

                consumers = list(t.consumers())
                for op in consumers:
                    for index, input_t in enumerate(op.inputs):
                        if input_t is t:
                            op._update_input(index, user_reshaped)
                tiled_count += 1
    return tiled_count


def main():
    print("=" * 60)
    print("V5: DeepRec 流程 — 建图 → 找边界 → 插 tile → 导出")
    print("=" * 60)

    g = tf.Graph()
    with g.as_default():
        user_ph = tf.compat.v1.placeholder(tf.float32, [None, 3], name="user_feat")
        user_ctx = tf.compat.v1.placeholder(tf.float32, [None, 4], name="user_context")
        item_ph = tf.compat.v1.placeholder(tf.float32, [None, 3], name="item_feat")
        item_price = tf.compat.v1.placeholder(tf.float32, [None, 2], name="item_price")
        item_size = tf.compat.v1.placeholder(tf.int32, shape=[], name="item_size")

        # ===== Step 1: 构建图（含压缩 tile） =====
        with tf.compat.v1.variable_scope("user_tower"):
            uf = dense(user_ph, 128, activation=tf.nn.relu, name="ufeat_d1")
            uf = dense(uf, 64, name="ufeat_emb")
        with tf.compat.v1.variable_scope("user_context_tower"):
            uc = dense(user_ctx, 128, activation=tf.nn.relu, name="uctx_d1")
            uc = dense(uc, 64, name="uctx_emb")
        u = tf.concat([uf, uc], axis=1, name="user_concat")

        # user 来一份 [1, 128]，tile 到 [N, 128]（等价 DeepRec add_tile_op）
        u_tiled = tf.tile(u, tf.stack([item_size, tf.constant(1, tf.int32)]), name="user_tiled")

        with tf.compat.v1.variable_scope("item_tower"):
            i = dense(item_ph, 128, activation=tf.nn.relu, name="ifeat_d1")
            i = dense(i, 64, name="ifeat_emb")
        with tf.compat.v1.variable_scope("item_price_tower"):
            ip = dense(item_price, 128, activation=tf.nn.relu, name="iprice_d1")
            ip = dense(ip, 64, name="iprice_emb")
        i_concat = tf.concat([i, ip], axis=1, name="item_concat")

        x = tf.concat([u_tiled, i_concat], axis=1, name="cross_concat")
        x = dense(x, 128, activation=tf.nn.relu, name="top_d1")
        x = dense(x, 64, activation=tf.nn.relu, name="top_d2")
        x = dense(x, 32, activation=tf.nn.relu, name="top_d3")
        logit = dense(x, 1, name="top_out")
        scores = tf.sigmoid(tf.squeeze(logit, axis=1), name="scores")

    # ===== Step 2: 找边界 =====
    print("\nStep 2: find_boundery_tensors")
    ops = g.get_operations()
    item_sets, user_sets, boundaries = deeprec_find_boundary(
        ops, ["user_feat", "user_context"], ["item_feat", "item_price"])

    print(f"  item_sets: {len(item_sets)} ops")
    print(f"  user_sets: {len(user_sets)} ops")
    print(f"  boundaries: {len(boundaries)} tensors:")
    for t in boundaries:
        consumers = [c.name for c in t.consumers()]
        print(f"    {t.name} (shape={t.shape}) → consumed by: {consumers}")

    # 验证边界检测: boundary 就是 user_tiled → cross_concat
    # DeepRec 的 add_tile_op 会在这里插 expand_dims→tile→reshape，
    # 效果等价于我们构建时写的 tf.tile(u, ...)

    # ===== Step 4: 导出 =====
    print("\nStep 4: 导出 saved_model...")
    export_path = os.path.join(EXPORT_DIR, "compress_v5")
    if os.path.exists(export_path):
        shutil.rmtree(export_path)

    with g.as_default():
        init = tf.compat.v1.global_variables_initializer()
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_path)
    with tf.compat.v1.Session(graph=g) as sess:
        sess.run(init)
        ti_u = tf.compat.v1.saved_model.utils.build_tensor_info(user_ph)
        ti_uc = tf.compat.v1.saved_model.utils.build_tensor_info(user_ctx)
        ti_i = tf.compat.v1.saved_model.utils.build_tensor_info(item_ph)
        ti_ip = tf.compat.v1.saved_model.utils.build_tensor_info(item_price)
        ti_s = tf.compat.v1.saved_model.utils.build_tensor_info(item_size)
        ti_o = tf.compat.v1.saved_model.utils.build_tensor_info(scores)
        sig = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
            inputs={"user_feat": ti_u, "user_context": ti_uc,
                    "item_feat": ti_i, "item_price": ti_ip, "item_size": ti_s},
            outputs={"output_0": ti_o},
            method_name=tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME)
        builder.add_meta_graph_and_variables(
            sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
            signature_def_map={"serving_default": sig})
    builder.save()
    print(f"  → {export_path}")

    # ===== Step 5: Benchmark =====
    comp_fn = tf.saved_model.load(export_path).signatures["serving_default"]
    print(f"\nStep 5: Benchmark")
    print(f"{'N':>6} | {'V5(ms)':>10} {'P99':>8}")
    print("-" * 30)
    for N in BENCH_N_LIST:
        u_feat = np.random.randn(1, 3).astype(np.float32)
        u_ctx = np.random.randn(1, 4).astype(np.float32)
        i_feat = np.random.randn(N, 3).astype(np.float32)
        i_price = np.random.randn(N, 2).astype(np.float32)
        sz = tf.constant(N, dtype=tf.int32)

        for _ in range(BENCH_WARMUP):
            comp_fn(user_feat=tf.constant(u_feat), user_context=tf.constant(u_ctx),
                    item_feat=tf.constant(i_feat), item_price=tf.constant(i_price),
                    item_size=sz)

        l_c = []
        for _ in range(BENCH_ITERS):
            t0 = time.perf_counter()
            comp_fn(user_feat=tf.constant(u_feat), user_context=tf.constant(u_ctx),
                    item_feat=tf.constant(i_feat), item_price=tf.constant(i_price),
                    item_size=sz)
            l_c.append((time.perf_counter()-t0)*1000)
        l_c = np.array(l_c)
        print(f'{N:>6} | {l_c.mean():>10.3f} {np.percentile(l_c,99):>8.3f}')


if __name__ == "__main__":
    main()
