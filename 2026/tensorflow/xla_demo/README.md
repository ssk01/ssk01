# XLA 优化 Demo

> 测试环境：MacBook Pro M4 Pro (CPU)。XLA 在 CPU 上的收益来自 op 融合（减少中间 tensor 分配/拷贝），在 GPU/TPU 上还有 kernel launch overhead 节省，加速比会更显著。

XLA (Accelerated Linear Algebra) 是 TensorFlow 的 JIT 编译器。`@tf.function(jit_compile=True)` 会把计算图编译成优化后的机器码，核心收益来自 **op 融合**：

```
MatMul → BiasAdd → Relu     →    单个 fused kernel
（3 个 op，2 次中间 tensor 分配）    （1 个 op，0 次中间分配）
```

## 三种模式对比

| 模式 | 代码 | 特点 |
|------|------|------|
| eager | `model([u, v], training=False)` | 每个 op 单独执行，Python→C 往返 |
| tf.function | `@tf.function` 无 jit_compile | 图模式，消除 Python dispatch 开销 |
| XLA | `@tf.function(jit_compile=True)` | 图编译 + op 融合 + 寄存器分配 |

## Benchmark

模型：user tower (3→256→128→64) + item tower (3→256→128→64) + top_mlp (128→64→32→1)，约 99K 参数。

| batch | eager | tf.function | XLA | XLA vs graph |
|-------|-------|-------------|-----|-------------|
| 1 | 2.32ms | 0.12ms | 0.09ms | 0.73x |
| 32 | 2.42ms | 0.19ms | 0.20ms | 1.08x |
| 128 | 2.87ms | 0.39ms | 0.38ms | 0.98x |
| 512 | 3.40ms | 0.80ms | 0.76ms | 0.95x |
| 2048 | 5.82ms | 1.78ms | 1.57ms | 0.88x |
| 8192 | 12.97ms | 5.53ms | 2.83ms | 0.51x |

## 结论

- **eager → tf.function**：小 batch 也有 5-20x 加速，消除了 Python↔C 往返开销
- **tf.function → XLA**：batch=8192 时 **1.95x** 加速，大 batch 下 op 融合收益明显
- **小 batch（<128）**：XLA 的 JIT 编译 overhead 抵消了融合收益，甚至略慢
- **中大 batch（512+）**：XLA 稳定快于 graph mode，且 batch 越大优势越明显

## 什么时候开 XLA

- batch ≥ 512：**建议开**，收益稳定
- 计算密集型模型（多层 MatMul、Conv）：**强烈建议**
- batch < 128、轻量模型：收益微小，可能负优化
- 动态 shape（变长序列等）：XLA 会频繁重编译，**不建议**

## 运行

```bash
cd xla_demo
python xla_benchmark.py
```
