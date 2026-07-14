# checkpoint 文件详解

## 概述

checkpoint 是训练过程中周期性保存的变量快照。一套 checkpoint 由三类文件组成：

```
checkpoints/
├── checkpoint                    ← 文本索引
├── ckpt-4.data-00000-of-00001    ← 实际数据
└── ckpt-4.index                  ← 变量名→数据位置 映射表
```

每一类文件承担不同的职责，下面逐项展开。

## 1. `checkpoint` 文本文件

这是整个 checkpoint 目录的「目录索引」，用 protobuf text format 记录。

**内容示例 (本 demo 实际输出)：**

```
model_checkpoint_path: "ckpt-4"           ← 当前最新版本
all_model_checkpoint_paths: "ckpt-2"      ← 历史版本 1
all_model_checkpoint_paths: "ckpt-3"      ← 历史版本 2
all_model_checkpoint_paths: "ckpt-4"      ← 历史版本 3
all_model_checkpoint_timestamps: 1784015298.994252
all_model_checkpoint_timestamps: 1784015299.2607172
all_model_checkpoint_timestamps: 1784015299.510666
last_preserved_timestamp: 1784015297.175224
```

**字段含义：**

| 字段 | 说明 |
|------|------|
| `model_checkpoint_path` | 当前最新 checkpoint 的前缀名 (不含 `.index`/`.data-*`) |
| `all_model_checkpoint_paths` | 所有存留的 checkpoint 列表 (最多 `max_to_keep` 个) |
| `all_model_checkpoint_timestamps` | 每个 checkpoint 的保存时间戳 |
| `last_preserved_timestamp` | 最后一次保留旧 checkpoint 的时间 |

**用途：** 续训 / restore 时，TF 先读这个文件找到最新的 checkpoint 前缀，再去找对应的 `.index` 和 `.data-*`。

**底层 protobuf 类型：** `CheckpointState`

```protobuf
message CheckpointState {
  string model_checkpoint_path = 1;
  repeated string all_model_checkpoint_paths = 2;
  repeated double all_model_checkpoint_timestamps = 3;
  double last_preserved_timestamp = 4;
}
```

## 2. `ckpt-N.index` 索引文件

这是整个 checkpoint 机制的核心——一个**变量名 → 数据偏移量的二进制查找表**。

### 2.1 作用

`index` 文件存在的目的是：**不用读完整个 data 文件就能找到某个变量**。

假设推理恢复时需要 `model/user_dense1/kernel` 这一个变量的值。如果 data 文件里是顺序存放的 29 个变量，没有 index 就得从头扫全部 19KB。有了 index，直接查表得到 `offset=xxx, size=yyy`，一次 seek + read 即可。

### 2.2 内部结构

`index` 文件的本质是一个 **protobuf 编码的有序键值表**。

```
┌────────────────────────────────────┐
│  BundleHeaderProto (文件头)         │
│  - num_shards: 1                   │
│  - endianness: little              │
├────────────────────────────────────┤
│  BundleEntryProto × N (有序排列)     │
│  ┌──────────────────────────────┐  │
│  │ key: "model/user_dense1/..." │  │
│  │ offset: 1234  (在 data 中的位│  │
│  │ size: 384                     │  │
│  │ crc32: 0xABCD1234            │  │
│  │ dtype: DT_FLOAT              │  │
│  │ shape: [3, 32]               │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │ key: "model/user_dense2/..." │  │
│  │ ...                          │  │
│  └──────────────────────────────┘  │
│  ... (共 N 条)                     │
└────────────────────────────────────┘
```

**BundleEntryProto 核心字段 (protobuf 定义)：**

```protobuf
message BundleEntryProto {
  // 对应 TensorFlow 内部的 DataType 枚举
  DataType dtype = 1;

  // 变量的形状 (编码后的 TensorShapeProto)
  TensorShapeProto shape = 2;

  // 在 data 文件里的字节偏移 (正整数)
  int64 offset = 3;

  // 数据在 data 文件里占的字节数
  int64 size = 4;

  // CRC32C 校验和 (用于检测文件损坏)
  fixed32 crc32 = 5;

  // 分片信息: 这个变量存在第几片 data 文件里
  repeated int32 slices = 6;
}
```

**一个具体的 entry 在文件中的样子 (本 demo 的实际数据)：**

```
变量名: model/item_dense1/kernel/.ATTRIBUTES/VARIABLE_VALUE
dtype:  DT_FLOAT (1)
shape:  [3, 32]
offset: 约 6000  (在 data 文件中的字节位置)
size:   384      (3×32×4 bytes = 384)
crc32:  0x???
```

### 2.3 变量名组织规则

TF2 的 checkpoint 变量名遵循以下命名规范：

```
<trackable_name>/<attr_name>/.ATTRIBUTES/VARIABLE_VALUE
       │              │                    │
       │              │                    └── 固定后缀 (TF2 checkpoint 特征)
       │              └── layer 内部属性名 (kernel / bias)
       └── 对象属性链 (model → user_dense1 → kernel)
```

**对照本 demo：**

| 代码中的对象 | checkpoint 中的 key |
|-------------|-------------------|
| `model.user_dense1.kernel` | `model/user_dense1/_kernel/.ATTRIBUTES/VARIABLE_VALUE` |
| `model.user_dense1.bias` | `model/user_dense1/bias/.ATTRIBUTES/VARIABLE_VALUE` |
| `optimizer._iterations` | `optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE` |
| `step` (tf.Variable) | `step/.ATTRIBUTES/VARIABLE_VALUE` |

TF1 的 checkpoint 则不同——格式是 `<scope>/<variable_name>` (如 `user_tower/dense/kernel:0`)，没有 `.ATTRIBUTES/VARIABLE_VALUE` 后缀。两者不兼容。

### 2.4 Index 的大小

本 demo：
- 29 个变量
- index 文件大小：**1716 bytes** (约 1.7 KB)
- 平均每条 entry 约 59 bytes (变量名 + protobuf 字段编码)

变量越多，index 文件越大，但始终远小于 data 文件。

### 2.5 为什么要有 index，不能直接写在 data 开头？

因为 checkpoint 需要支持**增量写入**和**选择性读取**：
- 训练到一半 crash 了，restore 时只需读需要的几个变量
- 分布式训练中不同 shard 写入不同变量，index 知道每个变量在哪片 data 里
- 没有 index，就得顺序解析整个 data 文件，O(n) 查找变成 O(n) 扫描

## 3. `ckpt-N.data-*` 数据文件

### 3.1 内容

`data-*` 文件里存放**变量的实际数值**，按 index 中记录的 offset + size 散布在文件中。文件内的顺序由变量名的**字典序**决定（因为 index 是有序的键值表）。

本 demo 的 data 文件 (~26 KB) 内容分布：

```
┌──────────────────────────────────────────────────┐  offset=0
│                                                  │
│  _CHECKPOINTABLE_OBJECT_GRAPH                    │
│  ───────────────────────────────                 │
│  • 类型: protobuf string                         │
│  • 内容: TrackableObjectGraph — 记录了所有       │
│    被追踪对象的拓扑关系 (谁是谁的子模块、        │
│    依赖关系、slot 变量归属)                      │
│  • 用途: TF2 restore 时靠它重建对象层级图，       │
│    知道 model.item_dense1 是 TwoTowerModel        │
│    的 item_dense1 属性                           │
│  • 大小: ~4 KB (varies)                          │
│                                                  │
├──────────────────────────────────────────────────┤  offset≈4000
│                                                  │
│  模型权重 (14 个) — 按字典序排列                  │
│  ─────────────────────────────                   │
│  model/item_dense1/bias          float32[32]      │
│  model/item_dense1/kernel        float32[3,32]   │
│  model/item_dense2/bias          float32[16]      │
│  model/item_dense2/kernel        float32[32,16]  │
│  model/top_dense1/bias           float32[16]      │
│  model/top_dense1/kernel         float32[32,16]  │
│  ... (其余 8 个)                                  │
│  每个变量以 TensorProto 格式存储                  │
│                                                  │
├──────────────────────────────────────────────────┤
│                                                  │
│  Optimizer slot 变量 (28 个) — Adam 的 m 和 v    │
│  ───────────────────────────────────             │
│  optimizer/_variables/2          float32[3,32]   │
│  optimizer/_variables/3          float32[3,32]   │
│  ... (其余 26 个)                                 │
│  每个 slot 的 shape 与对应的权重完全一致           │
│                                                  │
├──────────────────────────────────────────────────┤
│                                                  │
│  scalar 变量 (4 个)                               │
│  ─────────────────                                │
│  optimizer/_iterations           int64 标量       │
│  optimizer/_learning_rate        float32 标量     │
│  save_counter                    int64 标量       │
│  step                            int64 标量       │
│  标量用 protobuf varint 编码，不一定 4 bytes      │
│                                                  │
└──────────────────────────────────────────────────┘
```

**为什么是这个顺序？** index 文件内部按变量名的字典序组织键值对，data 文件中各变量的 offset 自然也是字典序递增的。所以你会看到 `model/...` (字母 m) 排在 `optimizer/...` (字母 o) 前面。

### 3.2 各变量详细说明

除了模型权重之外，checkpoint 还存储了以下变量，每类都有各自的作用：

| 变量 | 类型 | 谁创建的 | 作用 |
|------|------|---------|------|
| `step` | int64 标量 | `tf.Variable(0, dtype=tf.int64)` 用户代码 | 训练全局步数计数器，每 batch +1。续训时从这个值继续 |
| `save_counter` | int64 标量 | `tf.train.Checkpoint` 内部 | checkpoint 保存次数。每次 `ckpt_manager.save()` 自动 +1，用于生成 ckpt-N 中的 N |
| `optimizer/_iterations` | int64 标量 | `Adam.__init__` 内部 | Adam 的迭代计数。用于计算 bias correction：`lr_t = lr * sqrt(1 - β2^t) / (1 - β1^t)`，t 越大校正越小 |
| `optimizer/_learning_rate` | float32 标量 | `Adam.__init__` 内部 | 优化器的学习率。续训时恢复，确保学习率衰减不被打断 |
| `optimizer/_variables/{2..29}` | float32 张量 | `Adam.apply_gradients` 内部 | Adam 的 slot 变量：每个可训练权重对应 2 个 slot — 一阶矩 m (指数移动平均的梯度) 和二阶矩 v (指数移动平均的梯度平方)。偶数编号是 m，奇数编号是 v |
| `_CHECKPOINTABLE_OBJECT_GRAPH` | string (protobuf) | `tf.train.Checkpoint` 内部 | TF2 特有的 TrackableObjectGraph。restore 时先解析这个图，重建 Python 对象层级 (谁是谁的属性)，再把数值灌回对应的变量。没有它，TF 不知道 `optimizer/_variables/2` 对应 `model.user_dense1.kernel` 的 Adam slot |

**注意区分** `step` 和 `save_counter`：`step` 是训练步数（用户语义），`save_counter` 是保存次数（框架语义）。5 个 epoch × 32 batch/epoch = 160 step，但只有 5 次 save → save_counter=5。

### 3.3 数据格式

每个变量的数据以 TensorProto 格式存储：

```
┌──────────────────────────────────────┐
│  TensorProto (protobuf 消息)          │
│  - dtype: DT_FLOAT                   │
│  - tensor_shape: [3, 32]             │
│  - tensor_content: 384 raw bytes     │  ← 直接 memcpy 出来的 float32 值
└──────────────────────────────────────┘
```

标量 (如 `step`、`save_counter`) 用 protobuf 的 varint 编码，不一定是 4 bytes。

### 3.4 分片机制

`data-00000-of-00001` 的命名含义：
- `00000`：当前文件是第 0 片
- `00001`：总共 1 片

大模型 (数十 GB) 可以分片到多个 data 文件，加速分布式读写。本 demo 只有 19KB，不需要分片。

## 4. 变量清单 (本 demo 的 ckpt-4 实际内容)

```
模型权重 (14 个) - 推理需要:
  User Tower:
    model/user_dense1/kernel       [3, 32]   = 384 bytes
    model/user_dense1/bias         [32]      = 128 bytes
    model/user_dense2/kernel       [32, 16]  = 2048 bytes
    model/user_dense2/bias         [16]      = 64 bytes
  Item Tower:
    model/item_dense1/kernel       [3, 32]   = 384 bytes
    model/item_dense1/bias         [32]      = 128 bytes
    model/item_dense2/kernel       [32, 16]  = 2048 bytes
    model/item_dense2/bias         [16]      = 64 bytes
  顶部 MLP:
    model/top_dense1/kernel        [32, 16]  = 2048 bytes
    model/top_dense1/bias          [16]      = 64 bytes
    model/top_dense2/kernel        [16, 8]   = 512 bytes
    model/top_dense2/bias          [8]       = 32 bytes
    model/top_out/kernel           [8, 1]    = 32 bytes
    model/top_out/bias             [1]       = 4 bytes
  ─────────────────────────────────────
  小计: 7920 bytes (7.7 KB)

Optimizer 状态 (28 个) - 推理不需要:
  每个权重对应 2 个 Adam slot (一阶矩 m + 二阶矩 v)
  optimizer/_variables/{2..29}  总计 15840 bytes (15.5 KB)

其他状态 (4 个):
  optimizer/_iterations    []     ← int64 标量
  optimizer/_learning_rate []     ← float32 标量
  step                     []     ← int64 标量
  save_counter             []     ← int64 标量

对象图 (1 个):
  _CHECKPOINTABLE_OBJECT_GRAPH   ← TF2 对象拓扑 (protobuf string)

总变量数: 47
模型权重: 7.7 KB  ← 推理真正需要的
其余开销: ~15.5 KB ← optimizer slot + 对象图 + 标量
```

**核心洞察：** optimizer 的动量信息占了约 2/3 的存储。这就是为什么 checkpoint 不能直接上线推理——它携带了大量训练专用的膨胀数据。

## 5. `_CHECKPOINTABLE_OBJECT_GRAPH` 详解

这是 TF2 checkpoint 与 TF1 最核心的差异。它是一个序列化的 `TrackableObjectGraph` protobuf，记录的是**对象之间的引用关系**，而非计算图。

### 5.1 先纠正一个常见误解

很多人的第一反应是：「checkpoint 里存了这个对象图 → restore 时通过它重建 Python 对象」。**不对。**

**Python 对象树在你重跑代码时已经建好了。** `restore` 不创建任何新对象——它只往已有的 `tf.Variable` 里灌数值。

```python
# 续训代码
model = TwoTowerModel()                           # ← 对象树在这里创建
opt = Adam()
ckpt = tf.train.Checkpoint(model=model, optimizer=opt, step=step)

ckpt.restore("checkpoints/ckpt-4").expect_partial()   # ← 只灌值, 不建对象
```

那 `_CHECKPOINTABLE_OBJECT_GRAPH` 到底干了什么？**它是一张「查找表」——把 Python 对象树里的每个变量，映射到 checkpoint data 文件中的对应 key。**

### 5.2 restore 的四个步骤

以本 demo 为例，`ckpt.restore()` 内部的实际工作：

**Step 1：两边都有树**

```
Python 对象树 (重跑代码产生)               checkpoint 对象图 (从文件读取)
──────────────────────────────────       ──────────────────────────────
ckpt                                     root (n0)
├── .model → TwoTowerModel               ├── child "model" → n1
│   ├── .user_dense1 → Dense             │   ├── child "user_dense1" → n5
│   │   ├── ._kernel → Variable          │   │   ├── child "_kernel" → n19
│   │   │                                  │   │   │   checkpoint_key = "model/.../VARIABLE_VALUE"
│   │   └── .bias → Variable             │   │   └── child "bias" → n20
│   └── ...                              │   └── ...
├── .optimizer → Adam                    ├── child "optimizer" → n2
└── .step → Variable                     └── child "step" → n3
```

**Step 2：从 Python 侧生成 tracking path**

TF 遍历 Python 对象的属性链，为每个 `tf.Variable` 生成一个「追踪路径」：

```
Python 对象                        tracking path
──────────────────────────────────────────────────
ckpt.model.user_dense1._kernel  →  "model/user_dense1/_kernel"
ckpt.model.user_dense1.bias     →  "model/user_dense1/bias"
ckpt.optimizer._iterations      →  "optimizer/_iterations"
ckpt.step                       →  "step"
...
```

**Step 3：tracking path → checkpoint_key (靠对象图)**

TF 用同样的追踪路径去遍历 checkpoint 的对象图，找到每个变量节点上记录的 `checkpoint_key`：

```
tracking path                          checkpoint_key (从对象图查到)
─────────────────────────────          ─────────────────────────────────────
model/user_dense1/_kernel              model/user_dense1/_kernel/.ATTRIBUTES/VARIABLE_VALUE
model/user_dense1/bias                 model/user_dense1/bias/.ATTRIBUTES/VARIABLE_VALUE
optimizer/_iterations                  optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE
step                                   step/.ATTRIBUTES/VARIABLE_VALUE
```

**Step 4：用 checkpoint_key 读到值，灌进 Python Variable**

```
checkpoint_key  →  index 查 offset+size  →  data 文件读值  →  写入 Python Variable
```

整个过程就是一张大号的「路径 → 键 → 值」查表，没有任何对象被新建。

### 5.3 对象图内部结构

protobuf 定义：

```
TrackableObjectGraph
└── nodes: 数组, 下标即 node_id
    └── 每节点:
        ├── children:      [(local_name, node_id), ...]    ← 子对象引用
        ├── attributes:    [{name, checkpoint_key}, ...]  ← 变量属性
        └── slot_variables: [...]                          ← optimizer slot 注册
```

本 demo 的 61 个节点形成以下树 (只保留关键路径)：

```
n0  (root: tf.train.Checkpoint)
├── "model"     → n1  (TwoTowerModel)
│   ├── "user_dense1" → n5  → "_kernel" → n19 → ckpt_key="model/user_dense1/_kernel/..."
│   │                      → "bias"    → n20 → ckpt_key="model/user_dense1/bias/..."
│   ├── "user_dense2" → n6  → "_kernel" → n21
│   │                      → "bias"    → n22
│   ├── "item_dense1" → n7  → ...
│   ├── "item_dense2" → n8  → ...
│   ├── "top_dense1"  → n9  → ...
│   ├── "top_dense2"  → n10 → ...
│   └── "top_out"     → n11 → "_kernel" → n31
│                           → "bias"    → n32
├── "optimizer" → n2  (Adam)
│   ├── "_iterations"  → n15 → ckpt_key="optimizer/_iterations/..."
│   ├── "_learning_rate" → n16
│   ├── "_trainable_variables" → n13 → [n19, n20, ..., n32] (14 个权重引用)
│   ├── "_momentums" → n17 → [n33, n36, ...] (14 个 m slot)    ← 见下文 "slot 变量"
│   ├── "_velocities" → n18 → [n34, n37, ...] (14 个 v slot)    ← 见下文 "slot 变量"
│   └── "_variables" → n12 → [_iterations(n15), _lr(n16), m0(n33), v0(n34), m1(n36), ...]
├── "step"         → n3  →  ckpt_key="step/..."
└── "save_counter" → n4  →  ckpt_key="save_counter/..."
```

**什么是 slot 变量？** 优化器为每个被训练的权重创建的配套状态变量。Adam 为每个权重创建 2 个 slot：一阶矩 m (动量的指数移动平均) 和二阶矩 v (梯度平方的指数移动平均)。续训时如果丢失了这些 slot，Adam 就得从头开始累积动量——训练效果会退化。

Adam 在对象图中用三组平行列表管理这个映射，按位置一一对应：

```
_trainable_variables[i] → 第 i 个权重
_momentums[i]           → 第 i 个权重的 m slot
_velocities[i]          → 第 i 个权重的 v slot
```

本 demo 的实际映射 (前3个权重)：

```
权重                                         m slot                                    v slot
───────────────────────────                   ───────────────────────────              ───────────────────────────
model/user_dense1/_kernel/...                 optimizer/_variables/2/...               optimizer/_variables/3/...
model/user_dense1/bias/...                    optimizer/_variables/4/...               optimizer/_variables/5/...
model/user_dense2/_kernel/...                 optimizer/_variables/6/...               optimizer/_variables/7/...
... (共 14 个权重, 28 个 slot)
```

不同优化器的 slot 数量不同：SGD=0、SGD+Momentum=1、Adam=2、RMSProp=2。这就是为什么 checkpoint 比 saved_model 大——restore 需要 slot 状态才能正确续训，saved_model 只需要权重。

### 5.4 如果代码改了，会发生什么？

这是理解对象图价值的关键。假设你在模型里加了一层 `top_dense3`：

```python
class TwoTowerModel(tf.keras.Model):
    def __init__(self):
        ...
        self.top_dense3 = tf.keras.layers.Dense(4)  # ← 新增
```

重新跑代码、restore 旧 checkpoint：

```
Python 侧                          checkpoint 对象图
─────────────────────              ──────────────────
model/top_dense1/_kernel     ←→    有
model/top_dense1/bias        ←→    有
model/top_dense3/_kernel     ←→    无  !!
```

`restore().expect_partial()` 会报告：Python 侧多了一个变量，checkpoint 里没有对应值。这个变量保持初始随机值。其他变量正常恢复。**重要的是，TF 不会因为多了一个变量就报错或崩溃**——只要旧的那些 tracking path 匹配上了，就能恢复。

反过来，如果 Python 侧少了一个变量 (删了一层)，checkpoint 里有这个变量的值但 Python 里没有对应对象 → 同样会报告不匹配，但也能继续。

这就是 `_CHECKPOINTABLE_OBJECT_GRAPH` 的核心价值：**它让 restore 有了容错能力**——匹配得上的恢复，匹配不上的报警但不阻塞。

### 5.5 计算图去哪了？为什么 checkpoint 里没有？

**这是 TF2 和 TF1 最根本的设计区别。**

```
TF1:  图是显式的, 静态的
      ├── 先构图 (define by run)
      ├── 图保存在 .meta 文件里 (MetaGraphDef: op 定义、tensor 连接)
      └── restore 时必须重建一模一样的图

TF2:  图是隐式的, 动态的 (eager execution)
      ├── 不需要先构图, 代码即图 (define by run)
      ├── 每次 forward pass 时, TF 动态追踪操作
      └── checkpoint 只存变量值, 不需要存图结构
```

**那图去哪了？在你的 Python 代码里。**

```python
class TwoTowerModel(tf.keras.Model):
    def call(self, user_feat, item_feat, training=False):
        user_vec = self.user_dense1(user_feat)
        user_vec = self.user_dense2(user_vec)
        item_vec = self.item_dense1(item_feat)
        item_vec = self.item_dense2(item_vec)
        combined = tf.concat([user_vec, item_vec], axis=1)
        x = self.top_dense1(combined)
        ...
        return tf.sigmoid(logit)
```

这段代码**就是图**。每次调用 `model(x)`，TF 在 eager 模式下逐行执行这些操作。如果需要序列化图给 serving 用，那是 `saved_model` 的职责（导出时把 `call` 标上 `@tf.function`，trace 出 ConcreteFunction）。

```
checkpoint:   变量值 (供续训)
saved_model:  正向推理图 + 变量值 (供上线 serving)
              ↑ 这里才有序列化的计算图
```

**一句话总结：checkpoint 没用 `_CHECKPOINTABLE_OBJECT_GRAPH` 来「建树」——树是代码建的。它只是一张查找表，桥接 Python 变量路径 → data 文件中的值。而计算图根本不需要存在 checkpoint 里，因为 TF2 用 eager execution，图就是你写的 Python 代码本身。**

## 6. TF1 vs TF2 checkpoint 差异

> **当前 demo 使用的是 TF2 格式。**
> 
> 判断依据：变量名带 `.ATTRIBUTES/VARIABLE_VALUE` 后缀、data 文件内含 `_CHECKPOINTABLE_OBJECT_GRAPH`、index 条目的 shape/dtype 编码方式。

| | TF1 checkpoint | TF2 checkpoint |
|---|---|---|
| 变量命名 | `scope/var_name:0` | `obj/attr/.ATTRIBUTES/VARIABLE_VALUE` |
| 对象图 | `.meta` 文件单独存计算图 (MetaGraphDef) | `_CHECKPOINTABLE_OBJECT_GRAPH` 嵌入 data 文件 (TrackableObjectGraph)，只存对象层级，**不存计算图** |
| 索引格式 | 旧版，shape 不一定记录 | 新版，每条 entry 含 dtype + shape |
| checkpoint 文本 | 同样用 `checkpoint` 文件 | 同样用 `checkpoint` 文件 |
| `.meta` 文件 | 有 (`ckpt-N.meta` 存图结构) | **无** (TF2 用 eager execution，不需要序列化图) |
| Saver vs Checkpoint | `tf.train.Saver` | `tf.train.Checkpoint` |

两者不兼容——TF2 无法 restore TF1 的 checkpoint，反之亦然。

## 7. 可视化 & 检查工具

### 6.1 代码查看

```python
from tensorflow.python.training.py_checkpoint_reader import CheckpointReader

reader = CheckpointReader('checkpoints/ckpt-4')

# 列出所有变量
for name, shape in reader.get_variable_to_shape_map().items():
    print(f'{name}: {shape}')

# 读取某个变量的值
tensor = reader.get_tensor('model/user_dense1/_kernel/.ATTRIBUTES/VARIABLE_VALUE')
print(tensor)  # numpy array of shape [3, 32]
```

### 6.2 CLI 工具

```bash
# 列出所有变量
python -m tensorflow.python.tools.inspect_checkpoint \
  --file_name=checkpoints/ckpt-4 \
  --all_tensors

# 只查看某个张量
python -m tensorflow.python.tools.inspect_checkpoint \
  --file_name=checkpoints/ckpt-4 \
  --tensor_name=model/user_dense1/_kernel/.ATTRIBUTES/VARIABLE_VALUE
```

### 6.3 TensorBoard

```bash
tensorboard --logdir checkpoints/
```

只能可视化训练时写入了 summary 的标量/histogram/embedding，不能直接浏览 checkpoint 内部的变量名列表。

### 6.4 查看 `checkpoint` 文本文件

```bash
cat checkpoints/checkpoint
```

最简单的了解当前 checkpoint 状态的方式——看有多少版本、最新是哪个。

## 7. checkpoint → saved_model 的导出过程

对照 demo 代码，导出本质就是一次「裁剪」：

```
1. 训练代码里已持有 model 对象 (变量已加载在内存中)
2. 构建一个只含前向算子的 tf.Module (ExportRoot)
3. 把 model 的 layer 逐个挂到 ExportRoot 上 (变量引用传递)
4. tf.saved_model.save(ExportRoot) 
   → TF 遍历 ExportRoot 的子模块
   → 找到所有 tf.Variable (即原来的模型权重)
   → 写出 saved_model.pb (正向图 + signature) + variables/ (权重)
   
整个过程中不需要显式读 checkpoint 文件——
变量已经在内存里 (训练完自然在, 或通过 tf.train.Checkpoint.restore 恢复)。
```

导出的结果是 **saved_model**，它只包含正向推理图 + 权重，不含 optimizer 状态、对象图、训练标量。这正是「checkpoint → 训练态；saved_model → 推理态」的技术含义。

## 参考

- [TensorFlow CheckpointReader 源码](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/py_checkpoint_reader.py)
- [BundleHeaderProto 定义 (TF internal)](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/tensor_bundle/tensor_bundle.h)
