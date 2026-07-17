"""
XLA Demo — 对比 XLA 开关对双塔推理性能的影响
================================================
XLA (Accelerated Linear Algebra) 是 TF 的 JIT 编译器，
把计算图编译成优化后的机器码，核心收益来自 op 融合：
  MatMul + BiasAdd + Relu → 单个 kernel，消除中间 tensor 分配/拷贝。

对比三种模式:
  eager:      无优化，每次 op 单独执行
  tf.function: TF 图模式，无 XLA
  XLA:         tf.function(jit_compile=True)，启用 XLA 编译

运行:
    python xla_benchmark.py
"""

import os, time, numpy as np, tensorflow as tf

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZES = [1, 8, 32, 128, 512, 2048, 8192]
WARMUP, ITERS = 30, 200
FEAT_DIM = 3
EMB_DIM = 64


def build_two_tower():
    user_in = tf.keras.Input(shape=(FEAT_DIM,), dtype=tf.float32, name="user_feat")
    item_in = tf.keras.Input(shape=(FEAT_DIM,), dtype=tf.float32, name="item_feat")

    u = tf.keras.layers.Dense(256, activation="relu", name="user_d1")(user_in)
    u = tf.keras.layers.Dense(128, activation="relu", name="user_d2")(u)
    u = tf.keras.layers.Dense(EMB_DIM, name="user_emb")(u)

    i = tf.keras.layers.Dense(256, activation="relu", name="item_d1")(item_in)
    i = tf.keras.layers.Dense(128, activation="relu", name="item_d2")(i)
    i = tf.keras.layers.Dense(EMB_DIM, name="item_emb")(i)

    x = tf.keras.layers.Concatenate(name="concat")([u, i])
    x = tf.keras.layers.Dense(128, activation="relu", name="top_d1")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="top_d2")(x)
    x = tf.keras.layers.Dense(32, activation="relu", name="top_d3")(x)
    x = tf.keras.layers.Dense(1, name="top_out")(x)
    scores = tf.keras.layers.Lambda(lambda t: tf.sigmoid(tf.squeeze(t, axis=1)), name="scores")(x)

    return tf.keras.Model(inputs=[user_in, item_in], outputs=scores, name="two_tower")


def bench_eager(model, batch_size):
    u = tf.constant(np.random.randn(batch_size, FEAT_DIM).astype(np.float32))
    v = tf.constant(np.random.randn(batch_size, FEAT_DIM).astype(np.float32))
    for _ in range(WARMUP):
        model([u, v], training=False)

    lats = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        model([u, v], training=False)
        lats.append((time.perf_counter() - t0) * 1000)
    return np.mean(lats), np.percentile(lats, 99)


def bench_traced(model, batch_size, jit_compile):
    @tf.function(jit_compile=jit_compile)
    def infer(u, v):
        return model([u, v], training=False)

    u = tf.constant(np.random.randn(batch_size, FEAT_DIM).astype(np.float32))
    v = tf.constant(np.random.randn(batch_size, FEAT_DIM).astype(np.float32))

    for _ in range(WARMUP):
        infer(u, v)

    lats = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        infer(u, v)
        lats.append((time.perf_counter() - t0) * 1000)
    return np.mean(lats), np.percentile(lats, 99)


def main():
    print("=" * 60)
    print("XLA Benchmark — eager vs tf.function vs XLA")
    print("=" * 60)

    model = build_two_tower()
    _ = model([tf.constant(np.zeros((1, FEAT_DIM), np.float32)),
               tf.constant(np.zeros((1, FEAT_DIM), np.float32))])

    print(f"\n{'BS':>6} | {'eager(ms)':>10} {'P99':>8} | {'graph(ms)':>10} {'P99':>8} | {'XLA(ms)':>10} {'P99':>8} | {'graph/eager':>10} | {'XLA/graph':>10}")
    print("-" * 110)

    for bs in BATCH_SIZES:
        e_mean, e_p99 = bench_eager(model, bs)
        g_mean, g_p99 = bench_traced(model, bs, jit_compile=False)
        x_mean, x_p99 = bench_traced(model, bs, jit_compile=True)

        ge = g_mean / e_mean if e_mean > 0 else 0
        xg = x_mean / g_mean if g_mean > 0 else 0

        print(f'{bs:>6} | {e_mean:>10.3f} {e_p99:>8.3f} | {g_mean:>10.3f} {g_p99:>8.3f} | {x_mean:>10.3f} {x_p99:>8.3f} | {ge:>9.2f}x | {xg:>9.2f}x')


if __name__ == "__main__":
    main()
