# demo_compress.py 实现方案

## 目标

在双塔模型上实现 user 特征压缩：user 塔只算一遍 `[1, feat]`，在交叉点 broadcast 到 `[N, feat]`，并对比 naive 版本验证收益。

## 实现路径

### Keras 原生构建

**思路**：在模型构建时定义两条 forward 路径，共享 layer 引用。训练 naive → 权重自动同步到 compressed。

**具体操作**：

1. 构建共享层 + 两个模型变体：
   - `NaiveModel`: user `[N, 3]` + item `[N, 3]` → score `[N]`
   - `CompressedModel`: user `[1, 3]` + item `[N, 3]` → score `[N]`
   - 共用同一组 `user_tower`/`item_tower`/`top_mlp` layer 引用

2. CompressedModel 内部流程：
   ```
   user_input    [1, 3]    ──→ user_tower ──→ user_emb [1, 64]
   item_input    [N, 3]    ──→ item_tower ──→ item_emb [N, 64]
                                                    │
   broadcast ← tf.broadcast_to(user_emb, [tf.shape(item)[0], 64])
                                                    │
                                    concat ←────────┘
                                      │
                                    top_mlp → score [N]
   ```
   - `BroadcastTo` 的 target N 用 `tf.shape(item)[0]` 动态获取
   - 不需要额外的 batch 占位符——N 直接从 item 张量 shape 读取

3. Benchmark 时分别跑 NaiveModel 和 CompressedModel，对比延迟

**优点**：不碰 protobuf，Keras 原生支持，权重共享无需手动同步

### SavedModel GraphDef 改写（备选路径）

**思路**：训练好的 NaiveModel 导出 saved_model → 解析 `saved_model.pb` → 修改 GraphDef → 写出新 pb

**具体操作**：

1. NaiveModel 导出 saved_model
2. 用 TF protobuf API 做图改写：
   - 读取 `saved_model.pb` → 解析为 `SavedModel` protobuf
   - 找到 user_tower 输出的最后一个节点
   - 找到 concat 点（`ConcatV2` 节点）
   - 在 concat 前插入 `Shape → StridedSlice → BroadcastTo` 子图
   - 更新 concat 的输入引用和所有受影响的 `_output_shapes`
   - 序列化回新的 pb
3. 加载改写后的 saved_model → 推理 → 对比

**优点**：不依赖模型源码，适用于任意 saved_model

**缺点**：图改写真复杂——节点引用、FunctionDef 内的 shape 属性、protobuf 序列化容易出错

**关于 fc_opt_batch**：某些图改写方案需要 runtime 知道候选数 N 才能做 `BroadcastTo`。Python 版本可以用 `tf.shape(item)[0]` 从图内直接取，不需要显式占位符。如需 padding（batch 对齐到 8/16 倍数），则需额外输入计算 `ceil(N/padding)*padding`，此时需要类似 `fc_opt_batch` 的机制。本 demo 无 padding，直接用 item shape。

## 推荐方案：Keras 原生

理由：
- demo 的目的是验证压缩收益，不是验证图改写真不真
- Keras 构建两种 forward 路径，权重共享，代码清晰易复现
- 不需要处理 protobuf、FunctionDef、shape 属性等细节
- 可以直接导出 compressed 版的 saved_model

## 实现步骤

### 1. 模型构建

```python
# 共享的 tower/MLP 层，定义一次
user_tower = [Dense(256,'relu'), Dense(128,'relu'), Dense(64)]
item_tower = [Dense(256,'relu'), Dense(128,'relu'), Dense(64)]
top_mlp   = [Dense(128,'relu'), Dense(64,'relu'), Dense(32,'relu'), Dense(1)]

# Naive 图
def build_naive():
    u = Input((3,), name='user_feat')       # [N, 3]
    i = Input((3,), name='item_feat')       # [N, 3]
    u_emb = user_tower(u)                   # [N, 64]
    i_emb = item_tower(i)                   # [N, 64]
    x = Concatenate()([u_emb, i_emb])       # [N, 128]
    return Model([u, i], top_mlp(x))

# Compressed 图
def build_compressed():
    u = Input((3,), name='user_feat')       # [1, 3]
    i = Input((3,), name='item_feat')       # [N, 3]
    u_emb = user_tower(u)                   # [1, 64]
    i_emb = item_tower(i)                   # [N, 64]
    n = tf.shape(i)[0]
    u_emb_bc = tf.broadcast_to(u_emb, [n, 64])  # [N, 64]
    x = Concatenate()([u_emb_bc, i_emb])    # [N, 128]
    return Model([u, i], top_mlp(x))
```

### 2. 权重同步

CompressedModel 直接复用 NaiveModel 的 layer 引用，训练一次自动同步。

### 3. Benchmark

```python
for N in [10, 50, 100, 500, 1000, 2000, 5000]:
    # naive: user [N, 3], item [N, 3]
    # compressed: user [1, 3], item [N, 3]
    # 测 latency, 计算 speedup = naive_time / compressed_time
```

### 4. 数值对拍

```python
user = some_user_feat       # [1, 3]
items = N_items_feat        # [N, 3]
s_naive = naive([np.tile(user, (N, 1)), items])
s_comp  = compressed([user, items])
assert np.allclose(s_naive, s_comp, atol=1e-6)
```

## 预期结果

加速比天花板 = `1 / (1 - user_tower_ratio)`，user 塔占比越高、N 越大，收益越明显。
