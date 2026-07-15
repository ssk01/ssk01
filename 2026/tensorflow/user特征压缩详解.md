# 用户特征压缩详解

## 概述

一次推荐打分请求是 **1 个 user × N 个候选 item**。原始做法把 user 特征在 N 个候选中复制 N 份，导致 user 塔重复计算 N 遍完全相同的结果。用户特征压缩就是把 user 输入从 `[N, feat]` 改为 `[1, feat]`，user 塔只算一次，在 cross point（与 item 交汇处）用 broadcast 扩展到 `[N, feat]`。

```
before:  user [N,3]  → user_tower → user_emb [N,16]  ─┐
         item [N,3]  → item_tower → item_emb [N,16]  ─┤ concat → top_mlp → score [N]
                                                        │
after:   user [1,3]  → user_tower → user_emb [1,16]  ─┤
                                         │ broadcast    │
         item [N,3]  → item_tower → item_emb [N,16]  ─┘ concat → top_mlp → score [N]
```

## 基础压缩 vs 深度压缩

本 demo 实现的是**双塔上的基础压缩**：在 concat 交叉点做 broadcast。对纯双塔而言，concat 就是唯一的交叉点，基础压缩已经把 broadcast 推到位了。

更复杂的模型（多个特征在不同深度与 item 交互、target-attention 等）需要**深度压缩**——逐算子分析每个中间张量是 user-dependent 还是 item-dependent，把每个 user 特征的 broadcast 位置「各自推迟到它的第一个交叉点」，而不是一刀切在输入层。

| | 基础压缩 | 深度压缩 |
|---|---|---|
| broadcast 位置 | 用 tf.shape(item)[0] 动态取 N，在 concat 处展开 | 每个 user 特征按自己的交叉点各自 broadcast |
| 适用模型 | 双塔（user/item 独立编码） | 任意图结构 |
| 实现方式 | Keras 构建时定义两条 forward 路径 | 图分析 + saved_model.pb 改写 |
| 额外输入 | 不需要（从 item tensor 取 N） | 可能需要 fc_opt_batch（尤其有 padding 时） |

对本 demo 的双塔，两者效果相同——都推到 concat。

## 实现方式

### Keras 原生（本 demo）

两个模型共享同一组 `Dense` 层引用，区别在于 user 输入的 batch 维和交叉点的 broadcast：

```python
# 共享层（只定义一次）
user_tower = [Dense(256,'relu'), Dense(128,'relu'), Dense(64)]
item_tower = [Dense(256,'relu'), Dense(128,'relu'), Dense(64)]
top_mlp   = [Dense(128,'relu'), Dense(64,'relu'), Dense(32,'relu'), Dense(1)]

# Naive: user [N,3], item [N,3]
# Compressed: user [1,3], item [N,3]
#   交叉点: tf.broadcast_to(user_emb, [tf.shape(item)[0], 64])
```

### broadcast 的具体实现

Compressed 模型在 concat 之前插入一个 `Lambda` 层，负责把 user 输出从 `[1, 64]` 展开到 `[N, 64]`：

```python
def broadcast_to_item_batch(inputs):
    user_emb, item_emb = inputs
    n = tf.shape(item_emb)[0]     # 运行时 N，取 item 的 batch 维
    d = tf.shape(user_emb)[1]     # embedding 维度（64）
    return tf.broadcast_to(user_emb, tf.stack([n, d]))

u_bc = Lambda(broadcast_to_item_batch, name="broadcast_user")([u, i])
```

关键点：

1. **用 `Lambda` 包装 TF ops**：Keras Functional API 中，`Input` 产出的 `KerasTensor` 不能直接传给 `tf.shape` / `tf.broadcast_to`，必须包在 Lambda 层里，等图执行时才调用
2. **N 从 item 动态取**：`tf.shape(item_emb)[0]` 在运行时返回 item 的实际 batch size，无需额外占位符
3. **BroadcastTo 是 copy-free**：TF 的 `BroadcastTo` 不实际复制数据，只改 metadata（stride=0 在 batch 维），不影响内存
4. **embedding 维度不变**：user_emb 是 `[1, 64]`，broadcast 后是 `[N, 64]`，64 不变

导出的 saved_model 中，这部分变成 `Shape → StridedSlice → Pack → BroadcastTo` 四个 op 的串联子图。

## Benchmark

模型：user tower (3→256→128→64), item tower (3→256→128→64), top_mlp (128→64→32→1)，约 99K 参数，user 塔占 35%。

| N (候选数) | Naive (ms) | Compressed (ms) | Speedup |
|-----------|-----------|----------------|---------|
| 10 | 0.15 | 0.14 | 1.12x |
| 50 | 0.23 | 0.19 | 1.17x |
| 100 | 0.32 | 0.25 | 1.24x |
| 500 | 0.70 | 0.52 | 1.36x |
| 1000 | 1.11 | 0.80 | 1.38x |
| 2000 | 1.55 | 1.34 | 1.15x |
| 5000 | 3.93 | 2.89 | 1.36x |

加速比峰值 1.38x，理论上限 = 1/(1-0.35) = 1.54x。N=2000+ 时 top_mlp 的 MatMul 开始主导，user 塔的相对占比下降，加速比有所回落但整体稳定。

## 数值对拍

Compressed 是 Naive 的等价变换，打分应完全一致：

```python
user [1, 3], items [N, 3]
s_naive = naive([np.tile(user, (N,1)), items])
s_comp  = compressed([user, items])
assert np.allclose(s_naive, s_comp, atol=1e-6)  # ✓ max_diff < 2e-7
```

验证通过（误差 < 2e-7），确认无损。

## 收益来源

1. **传输节省**：user 特征从 N 份降到 1 份（尤其长序列/大 embedding 场景明显）
2. **计算节省**：user 塔 MatMul 从 N 遍降到 1 遍
3. **与双塔天然契合**：两塔独立编码，交叉点（concat/点积）靠后，压缩能覆盖整条 user 子图

加速比 = `1 / (1 - user_tower_ratio)`。user 塔越重（层数多、序列长）、N 越大，收益越高。

## 边界与坑

- **target-attention（如 DIN）**：query 是 item，输出是 item-dependent，不能压缩。深度压缩（按属性/交叉点）才能自动区分。
- **动态 N**：candidate 数每次请求可变，broadcast 到动态维度。与 XLA 全图编译可能产生冲突（需分档或 skip）。
- **BN/LayerNorm**：推理时用固定 moving stats，不受 batch 变化影响，安全。但如果是依赖 batch 维统计的算子需单独检查。
- **小 N 场景**：（如精排几十个候选）broadcast 开销可能接近节省量，建议按实测决定是否开启。

## 运行

```bash
python demo_compress.py
# 阶段 1: 样本构造
# 阶段 2: 训练
# 阶段 3: 数值对拍
# 阶段 4: Benchmark
# 阶段 5: 导出 Naive + Compressed 两个 saved_model

python serving_benchmark.py
# 加载两个 saved_model，压测对比
```

## 参考

- [demo_compress.py](demo_compress.py) — 实现代码
- [demo_compress_plan.md](demo_compress_plan.md) — 方案设计
- [demo_two_tower.py](demo_two_tower.py) — 原始双塔 demo
