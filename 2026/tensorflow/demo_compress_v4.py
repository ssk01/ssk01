"""
V4: 导出时压缩 — Keras 模型图上做分析，构建带 broadcast 的新模型
==================================================================
不碰 pb，在模型对象层面分析 layer 拓扑、定位 user/item 子图，
构建压缩版模型后再导出 saved_model。

运行:
    python demo_compress_v4.py
"""

import os
import time
import numpy as np
import tensorflow as tf

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(WORK_DIR, "saved_models")

BENCH_N_LIST = [10, 50, 100, 500, 1000, 2000, 5000]
BENCH_WARMUP = 50
BENCH_ITERS = 200


def build_and_train_naive():
    user_tower = [
        tf.keras.layers.Dense(256, activation="relu", name="user_d1"),
        tf.keras.layers.Dense(128, activation="relu", name="user_d2"),
        tf.keras.layers.Dense(64, name="user_emb"),
    ]
    item_tower = [
        tf.keras.layers.Dense(256, activation="relu", name="item_d1"),
        tf.keras.layers.Dense(128, activation="relu", name="item_d2"),
        tf.keras.layers.Dense(64, name="item_emb"),
    ]
    top_mlp = [
        tf.keras.layers.Dense(128, activation="relu", name="top_d1"),
        tf.keras.layers.Dense(64, activation="relu", name="top_d2"),
        tf.keras.layers.Dense(32, activation="relu", name="top_d3"),
        tf.keras.layers.Dense(1, name="top_out"),
    ]

    u_in = tf.keras.Input(shape=(3,), dtype=tf.float32, name="user_feat")
    i_in = tf.keras.Input(shape=(3,), dtype=tf.float32, name="item_feat")
    u = u_in
    for layer in user_tower: u = layer(u)
    i = i_in
    for layer in item_tower: i = layer(i)
    x = tf.keras.layers.Concatenate(name="concat")([u, i])
    for layer in top_mlp: x = layer(x)
    scores = tf.keras.layers.Lambda(lambda t: tf.sigmoid(tf.squeeze(t, axis=1)), name="scores")(x)

    model = tf.keras.Model(inputs=[u_in, i_in], outputs=scores, name="naive_two_tower")

    np.random.seed(42)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(
        [np.random.randn(500, 3).astype(np.float32), np.random.randn(500, 3).astype(np.float32)],
        (np.random.randn(500) > 0).astype(np.float32),
        batch_size=32, epochs=3, verbose=0)
    return model


def compress_model(naive):
    """在导出前，从 naive Keras 模型对象构建 compressed 版本"""
    layer_by_name = {l.name: l for l in naive.layers}

    # 建立下游关系 — 通过 tracing intermediate tensor 名字
    downstream = {}
    for layer in naive.layers:
        if not hasattr(layer, 'output'):
            continue
        # 获取这个 layer 产出的 tensor 名字
        output_tensor_names = set()
        if isinstance(layer.output, list):
            for t in layer.output:
                output_tensor_names.add(t.name)
        else:
            output_tensor_names.add(layer.output.name)

        for other in naive.layers:
            if other is layer:
                continue
            # 检查 other 的输入中是否有来自 layer 输出的
            inputs = other.input if isinstance(other.input, list) else [other.input]
            for inp in inputs:
                if hasattr(inp, 'name') and inp.name in output_tensor_names:
                    downstream.setdefault(layer.name, []).append(other.name)
                    break

    # 从 Input 出发 BFS 标记层归属
    attrs = {}
    queue = [(naive.inputs[0].name, "user"), (naive.inputs[1].name, "item")]
    attrs[naive.inputs[0].name] = "user"
    attrs[naive.inputs[1].name] = "item"

    while queue:
        name, owner = queue.pop(0)
        if attrs.get(name) == "boundary":
            continue
        for child in downstream.get(name, []):
            cur = attrs.get(child)
            if cur is None:
                attrs[child] = owner
                queue.append((child, owner))
            elif cur != owner:
                attrs[child] = "boundary"

    assert attrs.get("concat") == "boundary", f"concat 不是 boundary: {attrs}"

    # 收集 user/item 路径上的层 + shared 层
    def collect_path(start_name, stop_name):
        layers = []
        queue = [start_name]
        while queue:
            name = queue.pop(0)
            for child in downstream.get(name, []):
                if attrs.get(child) == attrs.get(start_name) and child != stop_name:
                    layer = layer_by_name.get(child)
                    if layer and layer.name not in [l.name for l in layers]:
                        layers.append(layer)
                    queue.append(child)
        return layers

    user_layers = collect_path(naive.inputs[0].name, "concat")
    item_layers = collect_path(naive.inputs[1].name, "concat")

    shared_layers = []
    queue = ["concat"]
    while queue:
        name = queue.pop(0)
        for child in downstream.get(name, []):
            if attrs.get(child) not in ("user", "item"):
                layer = layer_by_name.get(child)
                if layer and layer.name not in [l.name for l in shared_layers]:
                    shared_layers.append(layer)
                queue.append(child)

    concat_layer = layer_by_name["concat"]

    # 构建 compressed forward
    u_new = tf.keras.Input(shape=(3,), dtype=tf.float32, name="user_feat")
    i_new = tf.keras.Input(shape=(3,), dtype=tf.float32, name="item_feat")

    u = u_new
    for layer in user_layers:
        u = layer(u)
    i = i_new
    for layer in item_layers:
        i = layer(i)

    u_bc = tf.keras.layers.Lambda(
        lambda inputs: tf.broadcast_to(inputs[0], tf.stack([tf.shape(inputs[1])[0], tf.shape(inputs[0])[1]])),
        name="broadcast_user"
    )([u, i])

    x = concat_layer([u_bc, i])
    for layer in shared_layers:
        x = layer(x)

    return tf.keras.Model(inputs=[u_new, i_new], outputs=x, name="compressed_v4")


def main():
    print("=" * 60)
    print("V4: 导出时压缩 — Keras 模型图上分析 + 构建 compressed 版模型")
    print("=" * 60)

    print("训练 naive...")
    naive = build_and_train_naive()

    print("分析 layer 拓扑 + 构建 compressed...")
    compressed = compress_model(naive)

    # 验证
    print("\n数值对拍:")
    for N in [5, 10, 50, 100]:
        u1 = np.random.randn(1, 3).astype(np.float32)
        items = np.random.randn(N, 3).astype(np.float32)
        sn = naive([np.tile(u1, (N, 1)), items], training=False).numpy()
        sc = compressed([u1, items], training=False).numpy()
        diff = np.abs(sn - sc).max()
        print(f"  N={N:>4d}  diff={diff:.2e}  {'✓' if diff < 1e-4 else '✗'}")

    # 导出
    import contextlib, sys, io
    naive_path = os.path.join(EXPORT_DIR, "compress_v4_naive")
    comp_path = os.path.join(EXPORT_DIR, "compress_v4")
    with contextlib.redirect_stdout(io.StringIO()):
        naive.export(naive_path)
        compressed.export(comp_path)

    # Benchmark
    n_fn = tf.saved_model.load(naive_path).signatures["serving_default"]
    c_fn = tf.saved_model.load(comp_path).signatures["serving_default"]

    print(f"\n{'N':>6} | {'Naive(ms)':>10} {'P99':>8} | {'V4(ms)':>10} {'P99':>8} | {'Speedup':>7}")
    print("-" * 65)
    for N in BENCH_N_LIST:
        u1 = np.random.randn(1, 3).astype(np.float32)
        items = np.random.randn(N, 3).astype(np.float32)
        ut_n = tf.constant(np.tile(u1, (N, 1)))
        ut_c = tf.constant(u1)
        it = tf.constant(items)
        for _ in range(BENCH_WARMUP):
            n_fn(user_feat=ut_n, item_feat=it)
            c_fn(user_feat=ut_c, item_feat=it)
        l_n, l_c = [], []
        for _ in range(BENCH_ITERS):
            t0 = time.perf_counter(); n_fn(user_feat=ut_n, item_feat=it); l_n.append((time.perf_counter()-t0)*1000)
            t0 = time.perf_counter(); c_fn(user_feat=ut_c, item_feat=it); l_c.append((time.perf_counter()-t0)*1000)
        l_n, l_c = np.array(l_n), np.array(l_c)
        print(f'{N:>6} | {l_n.mean():>10.3f} {np.percentile(l_n,99):>8.3f} | {l_c.mean():>10.3f} {np.percentile(l_c,99):>8.3f} | {l_n.mean()/l_c.mean():>6.2f}x')


if __name__ == "__main__":
    main()
