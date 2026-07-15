"""
双塔 User 特征压缩 Demo
========================
对比 Naive ([N, feat]) vs Compressed ([1, feat] → broadcast) 的双塔推理性能。

原理: 一次请求 1 user × N item，user 特征在 N 个候选中完全重复。
Compressed 版让 user 塔只在 [1, feat] 上算一次，在 concat 交叉点 broadcast 到 N。
user 塔计算量从 O(N) 降到 O(1)。

运行:
    python demo_compress.py
"""

import os
import shutil
import time
import numpy as np
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints_compress")
EXPORT_BASE_DIR = os.path.join(WORK_DIR, "saved_models")
TFRECORD_PATH = os.path.join(WORK_DIR, "samples.tfrecord")

USER_DIM = 64
ITEM_DIM = 64
N_SAMPLES = 2000
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

BENCH_N_LIST = [10, 50, 100, 500, 1000, 2000, 5000]
BENCH_WARMUP = 50
BENCH_ITERS = 200


def cleanup():
    for d in [CHECKPOINT_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
    for f in [TFRECORD_PATH]:
        if os.path.exists(f):
            os.remove(f)


def build_samples_tfrecord():
    print("\n" + "=" * 60)
    print("阶段 1: 样本构造")
    print("=" * 60)
    np.random.seed(42)
    with tf.io.TFRecordWriter(TFRECORD_PATH) as writer:
        for i in range(N_SAMPLES):
            u = np.random.randn(3).astype(np.float32)
            v = np.random.randn(3).astype(np.float32)
            score = u[0] * v[0]
            prob = 1.0 / (1.0 + np.exp(-score))
            label = 1 if np.random.random() < prob else 0
            example = tf.train.Example(features=tf.train.Features(feature={
                "user_feat": tf.train.Feature(float_list=tf.train.FloatList(value=u)),
                "item_feat": tf.train.Feature(float_list=tf.train.FloatList(value=v)),
                "label":     tf.train.Feature(float_list=tf.train.FloatList(value=[float(label)])),
            }))
            writer.write(example.SerializeToString())
    print(f"写入 {N_SAMPLES} 条样本 → {TFRECORD_PATH}")


def parse_tfrecord(serialized):
    features = {
        "user_feat": tf.io.FixedLenFeature([3], tf.float32),
        "item_feat": tf.io.FixedLenFeature([3], tf.float32),
        "label":     tf.io.FixedLenFeature([1], tf.float32),
    }
    parsed = tf.io.parse_single_example(serialized, features)
    return parsed["user_feat"], parsed["item_feat"], parsed["label"][0]


# ============================================================
# 构建共享层 + 两个模型变体
# ============================================================

def build_shared_layers():
    user_tower = [
        tf.keras.layers.Dense(256, activation="relu", name="user_d1"),
        tf.keras.layers.Dense(128, activation="relu", name="user_d2"),
        tf.keras.layers.Dense(USER_DIM, name="user_emb"),
    ]
    item_tower = [
        tf.keras.layers.Dense(256, activation="relu", name="item_d1"),
        tf.keras.layers.Dense(128, activation="relu", name="item_d2"),
        tf.keras.layers.Dense(ITEM_DIM, name="item_emb"),
    ]
    top_mlp = [
        tf.keras.layers.Dense(128, activation="relu", name="top_d1"),
        tf.keras.layers.Dense(64, activation="relu", name="top_d2"),
        tf.keras.layers.Dense(32, activation="relu", name="top_d3"),
        tf.keras.layers.Dense(1, name="top_out"),
    ]
    return user_tower, item_tower, top_mlp


def build_naive_model(user_tower, item_tower, top_mlp):
    u_in = tf.keras.Input(shape=(3,), dtype=tf.float32, name="user_feat")
    i_in = tf.keras.Input(shape=(3,), dtype=tf.float32, name="item_feat")

    u = u_in
    for layer in user_tower:
        u = layer(u)
    i = i_in
    for layer in item_tower:
        i = layer(i)

    x = tf.keras.layers.Concatenate(name="concat")([u, i])
    for layer in top_mlp:
        x = layer(x)
    scores = tf.keras.layers.Lambda(lambda t: tf.sigmoid(tf.squeeze(t, axis=1)), name="scores")(x)

    return tf.keras.Model(inputs=[u_in, i_in], outputs=scores, name="naive_two_tower")


def build_compressed_model(user_tower, item_tower, top_mlp):
    u_in = tf.keras.Input(shape=(3,), dtype=tf.float32, name="user_feat")
    i_in = tf.keras.Input(shape=(3,), dtype=tf.float32, name="item_feat")

    u = u_in
    for layer in user_tower:
        u = layer(u)

    i = i_in
    for layer in item_tower:
        i = layer(i)

    def broadcast_to_item_batch(inputs):
        user_emb, item_emb = inputs
        n = tf.shape(item_emb)[0]
        d = tf.shape(user_emb)[1]
        return tf.broadcast_to(user_emb, tf.stack([n, d]))

    u = tf.keras.layers.Lambda(broadcast_to_item_batch, name="broadcast_user")([u, i])

    x = tf.keras.layers.Concatenate(name="concat")([u, i])
    for layer in top_mlp:
        x = layer(x)
    scores = tf.keras.layers.Lambda(lambda t: tf.sigmoid(tf.squeeze(t, axis=1)), name="scores")(x)

    return tf.keras.Model(inputs=[u_in, i_in], outputs=scores, name="compressed_two_tower")


# ============================================================
# 阶段 2: 训练 (兼容 Naive 格式样本)
# ============================================================

def train_model(model):
    print("\n" + "=" * 60)
    print(f"阶段 2: 训练 ({model.name})")
    print("=" * 60)

    raw_dataset = tf.data.TFRecordDataset([TFRECORD_PATH])
    dataset = (raw_dataset
               .map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
               .shuffle(512)
               .batch(BATCH_SIZE)
               .prefetch(tf.data.AUTOTUNE))

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    for batch in dataset.take(1):
        _ = model(batch[:2])

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=3)

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for user_feat, item_feat, label in dataset:
            with tf.GradientTape() as tape:
                pred = model([user_feat, item_feat], training=True)
                loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(label, pred))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += loss.numpy()
            n_batches += 1
        avg_loss = epoch_loss / n_batches
        ckpt_path = ckpt_manager.save(checkpoint_number=epoch)
        print(f"Epoch {epoch+1}/{EPOCHS} | loss={avg_loss:.4f}")

    return model


# ============================================================
# 阶段 3: 数值对拍
# ============================================================

def verify_equivalence(naive, compressed):
    print("\n" + "=" * 60)
    print("阶段 3: 数值对拍 (Naive vs Compressed)")
    print("=" * 60)

    np.random.seed(99)
    for N in [5, 10, 50, 100]:
        user_1 = np.random.randn(1, 3).astype(np.float32)
        items_N = np.random.randn(N, 3).astype(np.float32)
        user_tiled = np.tile(user_1, (N, 1))

        s_naive = naive([user_tiled, items_N], training=False).numpy()
        s_comp = compressed([user_1, items_N], training=False).numpy()

        diff = np.abs(s_naive - s_comp).max()
        status = "✓" if diff < 1e-5 else "✗"
        print(f"  N={N:>4d}  max_diff={diff:.2e}  {status}")

    print()


# ============================================================
# 阶段 4: Benchmark
# ============================================================

def benchmark_model(model, N, is_compressed):
    np.random.seed(0)
    user_1 = np.random.randn(1, 3).astype(np.float32)
    items_N = np.random.randn(N, 3).astype(np.float32)

    if is_compressed:
        user_tensor = tf.constant(user_1)
    else:
        user_tensor = tf.constant(np.tile(user_1, (N, 1)))
    item_tensor = tf.constant(items_N)

    infer = tf.function(model)

    for _ in range(BENCH_WARMUP):
        infer([user_tensor, item_tensor])

    latencies = []
    for _ in range(BENCH_ITERS):
        t0 = time.perf_counter()
        infer([user_tensor, item_tensor])
        latencies.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latencies)
    return lat.mean(), np.percentile(lat, 99)


def run_benchmarks(naive, compressed):
    print("=" * 60)
    print("阶段 4: Benchmark (Naive vs Compressed)")
    print("=" * 60)

    print(f"\n{'N':>8} | {'Naive(ms)':>10} {'Naive P99':>10} | {'Comp(ms)':>10} {'Comp P99':>10} | {'Speedup':>7} | {'user_tower_ops':>20}")
    print("-" * 100)

    for N in BENCH_N_LIST:
        n_mean, n_p99 = benchmark_model(naive, N, is_compressed=False)
        c_mean, c_p99 = benchmark_model(compressed, N, is_compressed=True)
        speedup = n_mean / c_mean

        naive_ops = f"matmul {N}× user_tower"
        comp_ops = "matmul 1× user_tower"

        print(f"{N:>8} | {n_mean:>10.3f} {n_p99:>10.3f} | {c_mean:>10.3f} {c_p99:>10.3f} | {speedup:>6.2f}x | {comp_ops:>20}")
    print()


# ============================================================
# 阶段 5: 导出两个 saved_model
# ============================================================

def export_both(naive, compressed):
    print("=" * 60)
    print("阶段 5: 导出 saved_model")
    print("=" * 60)

    naive_path = os.path.join(EXPORT_BASE_DIR, "compress_naive")
    compressed_path = os.path.join(EXPORT_BASE_DIR, "compress_compressed")

    import contextlib, sys, io
    with contextlib.redirect_stdout(io.StringIO()):
        naive.export(naive_path)
        compressed.export(compressed_path)

    for tag, path in [("Naive", naive_path), ("Compressed", compressed_path)]:
        loaded = tf.saved_model.load(path)
        sig = loaded.signatures["serving_default"]
        inputs = [(t.name.split(":")[0], t.dtype.name) for t in sig.inputs if t.dtype != tf.resource]
        outputs = [(t.name.split(":")[0], t.dtype.name) for t in sig.outputs]
        print(f"\n{tag}: {path}")
        for name, dt in inputs:
            print(f"  in  {name}: {dt}")
        for name, dt in outputs:
            print(f"  out {name}: {dt}")

    return naive_path, compressed_path


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("双塔 User 特征压缩 Demo")
    print("Naive [N, feat] vs Compressed [1, feat] → broadcast")
    print("=" * 60)

    cleanup()

    build_samples_tfrecord()

    user_tower, item_tower, top_mlp = build_shared_layers()

    naive = build_naive_model(user_tower, item_tower, top_mlp)
    compressed = build_compressed_model(user_tower, item_tower, top_mlp)

    train_model(naive)
    verify_equivalence(naive, compressed)
    run_benchmarks(naive, compressed)
    export_both(naive, compressed)


if __name__ == "__main__":
    main()
