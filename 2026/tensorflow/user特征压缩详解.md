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

## 四种实现

| | V1 Keras 原生 | V2 pb 改写 | V3 自动边界(pb) | V4 导出时压缩 |
|---|---|---|---|---|
| 文件 | `demo_compress.py` | `demo_compress_v2.py` | `demo_compress_v3.py` | `demo_compress_v4.py` |
| 需要模型源码 | 需要 | 不需要 | 不需要 | 需要 |
| 时机 | 构建时 | 导出后改 pb | 导出后改 pb | **导出前**（分析 → 构建 → 导出） |
| 原理 | 共享 layer | 解析 pb → 找 concat → 插 broadcast | BFS 传播归属 → 自动找 boundary → 插 broadcast | BFS 分析 Keras 层拓扑 → 构建新模型 → 导出 |

> V2/V3 是在导出的 pb 上做后期手术；V4 是在模型对象上分析并构建压缩版，更接近 DeepRec 在 `serving_input_receiver_fn` 中做图变换的思路。

### V1: Keras 原生 (demo_compress.py)

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

Compressed 模型在 concat 之前插入一个 `Lambda` 层，负责把 user 输出从 `[1, 64]` 展开到 `[N, 64]`：

```python
def broadcast_to_item_batch(inputs):
    user_emb, item_emb = inputs
    n = tf.shape(item_emb)[0]     # 运行时 N，取 item 的 batch 维
    d = tf.shape(user_emb)[1]     # embedding 维度
    return tf.broadcast_to(user_emb, tf.stack([n, d]))

u_bc = Lambda(broadcast_to_item_batch, name="broadcast_user")([u, i])
```

**优点**：简单、可读性好。**缺点**：需要知道 user_tower/item_tower/top_mlp 的全部结构，感知一切。

### V2: pb 改写 (demo_compress_v2.py)

不依赖模型源码，直接修改已导出的 `saved_model.pb`。步骤：

1. 解析 `saved_model.pb` → 找到 `__inference___call___` 函数（FunctionDef，含 53 个节点）
2. 通过节点名匹配找到 `user_emb_1/BiasAdd`（user 塔输出）、`item_emb_1/BiasAdd`（item 塔输出）、`concat_1/concat`（交叉点）
3. 在 concat 前插入 broadcast 子图：

```
Shape(item_emb) → StridedSlice[0] → Pack([batch, emb_dim]) → BroadcastTo(user_emb, target_shape)
```

4. 修改 concat 的 input 引用，将 user_emb 换为 broadcast 输出
5. 序列化回新的 pb

**优点**：不需要模型源码。**缺点**：需要知道关键节点的命名约定（如 `user_emb`、`concat`）。

### V3: 自动边界检测 (demo_compress_v3.py)

完全自动化——只需告诉它 user/item 输入名，自动完成剩余步骤：

1. 解析 FunctionDef，构建 producer-consumer 图
2. 从 `user_feat` 和 `item_feat` 出发，BFS 标记每个节点的归属：
   - 纯 user 下游 → 标记 `user`
   - 纯 item 下游 → 标记 `item`
   - 同时接收 user 和 item 输入的节点 → 标记 `boundary`
3. 对每个 boundary 节点，为它的 user 输入自动插 broadcast（从另一个 item 输入获取 N）
4. 序列化回新的 pb

```python
analyzer = GraphAnalyzer(func_def)
analyzer.propagate(user_inputs=["user_feat"], item_inputs=["item_feat"])
boundaries = analyzer.find_boundary_inputs()
# 对每个 boundary 的 user 输入插 broadcast
```

**优点**：不依赖源码、不依赖命名约定，适用于任意 saved_model。

### V4: 导出时压缩 (demo_compress_v4.py)

在导出 saved_model **之前**，分析 Keras 模型的 layer 拓扑图，自动定位 user/item 子图，构建带 broadcast 的压缩版模型后再导出。接近 DeepRec 在 `serving_input_receiver_fn` 里做图变换的思路。

步骤：
1. 训练 naive 模型
2. 分析 layer 之间的 tensor 连接关系（通过 `layer.input.name` / `layer.output.name` 匹配）
3. 从 user/item Input 层出发 BFS，标记每个 layer 的归属
4. 找到 concat（boundary 层），收集 user 路径、item 路径、shared 路径上的 layer
5. 构建新 Keras Model：复用原 layer 引用，在 concat 前插 broadcast
6. 导出 compressed saved_model

**优点**：在模型对象层面操作，不需要解析 pb；自动分析层拓扑。**缺点**：需要模型源码。| N | Naive | V1 | V2 | V3 | V4 |

| N | Naive | V1 | V2 | V3 | V4 |
|---|-------|----|----|----|----|
| 100 | 0.27ms | 0.27ms | 0.31ms | 0.34ms | 0.21ms |
| 500 | 0.68ms | 0.70ms | 0.70ms | 0.71ms | 0.49ms |
| 1000 | 1.11ms | 1.00ms | 1.06ms | 1.05ms | 0.77ms |
| 5000 | 3.38ms | 2.61ms | 2.58ms | 2.61ms | 3.14ms |

V1/V2/V3 三者最终效果一致（N=5000 时 1.3x 加速），V2/V3 在 N 较小时因 broadcast 子图（Shape+StridedSlice+Pack+BroadcastTo）有微小额外开销。

## 数值对拍

Compressed 是 Naive 的等价变换，打分应完全一致：

```python
user [1, 3], items [N, 3]
s_naive = naive([np.tile(user, (N,1)), items])
s_comp  = compressed([user, items])
assert np.allclose(s_naive, s_comp, atol=1e-6)  # ✓ max_diff < 2e-7
```

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
# V1: Keras 原生（需要从头训练）
python demo_compress.py

# V2: pb 改写（已有 compress_naive saved_model 即可）
python demo_compress_v2.py

# V3: 自动边界+pb 改写（已有 compress_naive 即可）
python demo_compress_v3.py

# V4: 导出时压缩（从头训练）
python demo_compress_v4.py
```

## 参考

- [demo_compress.py](demo_compress.py) — V1 Keras 原生实现
- [demo_compress_v2.py](demo_compress_v2.py) — V2 pb 改写实现
- [demo_compress_v3.py](demo_compress_v3.py) — V3 自动边界+pb 改写
- [demo_compress_v4.py](demo_compress_v4.py) — V4 导出时压缩（Keras 层图分析）
- [demo_compress_plan.md](demo_compress_plan.md) — 方案设计
- [demo_two_tower.py](demo_two_tower.py) — 原始双塔 demo
