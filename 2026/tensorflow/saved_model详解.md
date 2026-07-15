# saved_model 详解

## 概述

saved_model 是 TensorFlow 的推理交付格式——一个自包含的目录，打包了「计算图 + 权重 + 签名」，脱离训练代码即可加载运行。

本 demo 使用 Keras Functional API + `model.export()` 导出：

```
saved_models/20260714/
├── saved_model.pb                     ← 核心: MetaGraphDef (图 + 签名定义)
├── fingerprint.pb                     ← 文件指纹 (校验完整性)
└── variables/
    ├── variables.data-00000-of-00001  ← 权重数值
    └── variables.index                ← 变量名 → 数据位置 映射表
```

## 1. saved_model.pb 内部结构

`saved_model.pb` 是一个 protobuf 序列化的 `SavedModel` 消息：

```
SavedModel
└── meta_graphs: repeated MetaGraphDef
    ├── meta_info_def       ← 元信息 (TF 版本, tags)
    ├── graph_def           ← 计算图 (所有 op 和它们的连接关系)
    ├── signature_def       ← 签名定义 (模型的输入输出接口)
    ├── object_graph_def    ← SavedObjectGraph (变量/函数/资源的对象树)
    ├── saver_def           ← Saver 定义 (变量保存/恢复方式)
    └── collection_def      ← 集合 (TF1 遗留, TF2 基本不用)
```

### 1.1 SignatureDef — 模型接口

这是最重要的部分——定义了 serving 端「喂什么、取什么」：

```

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['item_feat'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 3)
        name: serving_default_item_feat:0
    inputs['user_feat'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 3)
        name: serving_default_user_feat:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['output_0'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: StatefulPartitionedCall_1:0
  Method name is: tensorflow/serving/predict
```

**method_name 含义：** 告诉 TF Serving 输出类型，影响响应格式：

| method_name | 用途 |
|-------------|------|
| `tensorflow/serving/predict` | 通用预测，返回原始 tensor |
| `tensorflow/serving/classify` | 分类，自动加 softmax + top-k |
| `tensorflow/serving/regress` | 回归 |

### 1.2 GraphDef — 计算图

TF2 的 GraphDef 分两层：`graph_def.node` + `graph_def.library.function`。

**外层 graph_def.node (137 个 op)** — 全是初始化/管理 boilerplate：

```
ReadVariableOp:        56 个  从变量容器读 tensor (初始化阶段用)
VarHandleOp:           42 个  创建变量句柄 (声明有哪些 weight)
VarIsInitializedOp:    14 个  守卫: 变量未初始化时抛错
AssignVariableOp:      14 个  restore 时给变量赋值
Placeholder:            5 个  serving 输入占位符
StatefulPartitionedCall: 4 个  调用 library function 的入口 (展开执行内部 op)
NoOp, Const:           各 1 个
```

这些 op 不是推理计算——它们负责变量创建、检查和赋值。真正的正向推理 op 在 library function 里。

**内层 library.function (5 个)** — 实际的图函数：

```
__inference___call___           → 38 ops  ← 正向推理在这里
__inference__traced_save       → 128 ops
__inference__traced_restore    → 63 ops
2 个 signature_wrapper         → 各 3 ops (一层薄封装, 内部调用 __inference___call___)
```

**为什么 137 ≠ 38？** 137 是外层所有初始化 op 的总和——它们是 graph 加载时执行的准备代码。加载完后，真正每次推理只跑 `StatefulPartitionedCall` → 展开执行 `__inference___call___` 里的 38 个 op。这 38 个才是你的模型计算。

**op 之间的依赖关系** — 每个 NodeDef 有 `input` 字段，列出该 op 的上游 tensor（格式 `node_name:output_index`，`^` 前缀表示控制依赖）。通过 input 可追溯到输入直至输出，构成完整 DAG。

以 `user_dense1/MatMul` 为例，它在 protobuf JSON 中的原始格式：

```json
{
  "name": "two_tower_1/user_dense1_1/MatMul",
  "op": "MatMul",
  "input": [
    "user_feat",
    "two_tower_1/user_dense1_1/Cast/ReadVariableOp:value:0"
  ],
  "attr": {
    "T": { "type": "DT_FLOAT" },
    "_output_shapes": {
      "list": { "shape": [{ "dim": [{ "size": "-1" }, { "size": "32" }] }] }
    }
  }
}
```

| 字段 | 含义 |
|------|------|
| `name` | 节点唯一名称 |
| `op` | 算子类型 (`MatMul`, `BiasAdd`, `Relu`…) |
| `input` | 上游 tensor 名。`user_feat` 是函数输入参数，`ReadVariableOp:value:0` = 节点第 0 号输出 `value`。反查 upstream 节点即可追溯整条计算链 |
| `attr.T` | 数据类型 (`DT_FLOAT`) |
| `attr._output_shapes` | 输出 shape，`-1` 表示 batch 维度 (动态) |

它的上游 `ReadVariableOp` 则引用一个 `DT_RESOURCE` 资源变量（14 个 kernel + bias，对应 14 个 `VarHandleOp`），格式：

```json
{
  "name": "two_tower_1/user_dense1_1/Cast/ReadVariableOp",
  "op": "ReadVariableOp",
  "input": [
    "two_tower_1_user_dense1_1_cast_readvariableop_resource"
  ],
  "attr": {
    "dtype": { "type": "DT_FLOAT" },
    "_output_shapes": {
      "list": { "shape": [{ "dim": [{ "size": "3" }, { "size": "32" }] }] }
    }
  }
}
```

这就是 `Dense(3→32)` 的 kernel weight，shape `[3, 32]`。

完整 7 个 MatMul + 38 个 op 的原始 protobuf JSON 见 `saved_model_annotated.json`。

#### MatMul 依赖一览

每个 MatMul 恰好 2 个输入：上游激活 + 本层权重（ReadVariableOp）：

```
item_dense1/MatMul    ← item_feat                            ← ReadVariableOp(item_dense1/kernel)
user_dense1/MatMul    ← user_feat                            ← ReadVariableOp(user_dense1/kernel)
user_embedding/MatMul ← user_dense1/Relu:activations:0       ← ReadVariableOp(user_embedding/kernel)
item_embedding/MatMul ← item_dense1/Relu:activations:0       ← ReadVariableOp(item_embedding/kernel)
top_dense1/MatMul     ← concat/concat:output:0               ← ReadVariableOp(top_dense1/kernel)
top_dense2/MatMul     ← top_dense1/Relu:activations:0        ← ReadVariableOp(top_dense2/kernel)
top_out/MatMul        ← top_dense2/Relu:activations:0        ← ReadVariableOp(top_out/kernel)
```

#### 正向推理 38 个 op 完整展开

```
# === user tower (5 ops) ===
ReadVariableOp  user_dense1/Cast/ReadVariableOp    ← kernel 变量资源
MatMul          user_dense1/MatMul                  ← user_feat, kernel
ReadVariableOp  user_dense1/BiasAdd/ReadVariableOp  ← bias 变量资源
BiasAdd         user_dense1/BiasAdd                 ← MatMul:product:0, bias
Relu            user_dense1/Relu                    ← BiasAdd:output:0

# === item tower (5 ops) ===
ReadVariableOp  item_dense1/Cast/ReadVariableOp     ← kernel
MatMul          item_dense1/MatMul                   ← item_feat, kernel
ReadVariableOp  item_dense1/BiasAdd/ReadVariableOp   ← bias
BiasAdd         item_dense1/BiasAdd                  ← MatMul:product:0, bias
Relu            item_dense1/Relu                     ← BiasAdd:output:0

# === user embedding (3 ops，最后一层无激活) ===
ReadVariableOp  user_embedding/Cast/ReadVariableOp   ← kernel
MatMul          user_embedding/MatMul                 ← user_dense1/Relu:activations:0, kernel
ReadVariableOp  user_embedding/BiasAdd/ReadVariableOp ← bias
BiasAdd         user_embedding/BiasAdd                ← MatMul:product:0, bias   → user_emb (16维)

# === item embedding (3 ops) ===
ReadVariableOp  item_embedding/Cast/ReadVariableOp    ← kernel
MatMul          item_embedding/MatMul                  ← item_dense1/Relu:activations:0, kernel
ReadVariableOp  item_embedding/BiasAdd/ReadVariableOp  ← bias
BiasAdd         item_embedding/BiasAdd                 ← MatMul:product:0, bias   → item_emb (16维)

# === top MLP (14 ops: Concat → 16 → 8 → 1) ===
Const           concat/concat/axis                    ← 无 (常量 axis=1)
ConcatV2        concat/concat                         ← user_emb, item_emb, axis

ReadVariableOp  top_dense1/Cast/ReadVariableOp        ← kernel
MatMul          top_dense1/MatMul                      ← concat:output:0, kernel
ReadVariableOp  top_dense1/BiasAdd/ReadVariableOp      ← bias
BiasAdd         top_dense1/BiasAdd                     ← MatMul:product:0, bias
Relu            top_dense1/Relu                        ← BiasAdd:output:0

ReadVariableOp  top_dense2/Cast/ReadVariableOp        ← kernel
MatMul          top_dense2/MatMul                      ← top_dense1/Relu:activations:0, kernel
ReadVariableOp  top_dense2/BiasAdd/ReadVariableOp      ← bias
BiasAdd         top_dense2/BiasAdd                     ← MatMul:product:0, bias
Relu            top_dense2/Relu                        ← BiasAdd:output:0

ReadVariableOp  top_out/Cast/ReadVariableOp            ← kernel
MatMul          top_out/MatMul                          ← top_dense2/Relu:activations:0, kernel
ReadVariableOp  top_out/Add/ReadVariableOp             ← bias
AddV2           top_out/Add                            ← MatMul:product:0, bias   (Dense(1) 无激活, 用 AddV2)

# === 输出 (3 ops) ===
Squeeze         scores/Squeeze                         ← AddV2:z:0            (去掉维度1 → shape (batch,))
Sigmoid         scores/Sigmoid                         ← Squeeze:output:0
Identity        Identity                               ← Sigmoid:y:0, ^NoOp   (serving 最终输出)

# === 控制依赖 ===
NoOp            NoOp                                   ← ^14个ReadVariableOp  (确保 weight 全部初始化完毕)
```

> **`^` 控制依赖**: NoOp 的 14 个 `^ReadVariableOp` 不传数据，只保证执行顺序——所有变量读取在 NoOp 之前完成。Identity 通过 `^NoOp` 确保输出在所有操作后。

完整 38 个 op 的 JSON 标注见 `saved_model_annotated.json`。

### 1.3 SavedObjectGraph — 对象图

与 checkpoint 的 `_CHECKPOINTABLE_OBJECT_GRAPH` 类似，记录变量/函数/资源的层级关系，用于加载时重建可调用对象。

## 2. variables/ 目录

与 checkpoint 的 `data-*` + `index` 格式相同（见 [checkpoint详解.md](checkpoint详解.md)），但内容不同：

| | checkpoint variables/ | saved_model variables/ |
|---|---|---|
| 包含权重 | 是 | 是 |
| 包含 optimizer slot | 是 (46 个 m/v/iterations/lr) | **否** |
| 包含 step/save_counter | 是 | **否** |
| 大小 (本 demo) | ~40 KB | ~12 KB (只有权重) |

## 3. 导出过程

Functional API 模型的导出只需一行：

```python
model.export("saved_models/20260714")
```

内部自动完成：

```
训练模型 (内存中, Functional API)
         │
         ▼
  1. trace forward pass → ConcreteFunction (38 ops)
  2. 从 Input layer 提取签名 → serving_default 有 user_feat + item_feat 两个输入
  3. 写出 variables/ (14 个权重变量)
  4. 写出 saved_model.pb (图 + signature + 对象图)
         │
         ▼
saved_models/20260714/
```

**为什么 Functional API 不需要任何额外处理？** 建模时 `Input(shape=(3,), name="user_feat")` 显式声明了输入名和 shape——Keras 知道签名有两个输入、各自叫什么、什么类型和 shape，`model.export()` 自动生成正确的 `serving_default`。

对比 Subclassing API 的 `call(user_feat, item_feat)` ——签名是隐式的，Keras 只能假设 `call(inputs)` 单参数，多参数就崩。这就是为什么生产环境用 Functional API。

## 4. 加载 & 推理

```python
loaded = tf.saved_model.load("saved_models/20260714")
serve = loaded.signatures["serving_default"]
output = serve(
    user_feat=tf.constant([[0.5, -0.1, 0.6]]),
    item_feat=tf.constant([[1.5, -0.2, -0.2]]),
)
print(output["output_0"].numpy())  # [0.50...]
```

## 5. saved_model vs checkpoint

| 维度 | checkpoint | saved_model |
|------|-----------|-------------|
| 目的 | 训练存档、续训 | 推理交付、上线 serving |
| 包含图 | **否** (TF2 eager) | **是** (GraphDef + ConcreteFunction) |
| 包含权重 | 是 | 是 |
| 包含 optimizer slot | 是 | 否 |
| 自包含 | 否 (需代码重建对象树) | 是 (独立加载即可推理) |
| signature | 无 | 有 (SignatureDef) |
| tag | 无 | 有 (serve/train/eval 多图) |
| 加载方 | `tf.train.Checkpoint.restore()` | TF Serving / `tf.saved_model.load()` |
| 大小 (本 demo) | ~40 KB | ~100 KB |

## 6. 常见操作

### 查看完整内容

```bash
saved_model_cli show --dir saved_models/20260714 --all
```

### 只查看 serving_default

```bash
saved_model_cli show --dir saved_models/20260714 \
  --tag_set serve --signature_def serving_default
```

### CLI 跑推理

```bash
saved_model_cli run --dir saved_models/20260714 --tag_set serve \
  --signature_def serving_default \
  --input_exprs 'user_feat=[[0.5,-0.1,0.6]];item_feat=[[1.5,-0.2,-0.2]]'
# Result for output key output_0: [0.53342783]
```

### Python 加载推理

```python
loaded = tf.saved_model.load("saved_models/20260714")
output = loaded.signatures["serving_default"](
    user_feat=tf.constant([[0.5, -0.1, 0.6]]),
    item_feat=tf.constant([[1.5, -0.2, -0.2]]),
)
print(output["output_0"].numpy())
```

### 导出为 JSON 查看

```python
from tensorflow.core.protobuf import saved_model_pb2
from google.protobuf import json_format

sm = saved_model_pb2.SavedModel()
with open('saved_models/20260714/saved_model.pb', 'rb') as f:
    sm.ParseFromString(f.read())
with open('saved_model.json', 'w') as f:
    f.write(json_format.MessageToJson(sm, preserving_proto_field_name=True))
```

## 7. 可视化工具

| 工具 | 方式 | 特点 |
|------|------|------|
| **Netron** | `netron.start('saved_model.pb')` | 交互式节点图，拖拽折叠 |
| **TensorBoard** | `tensorboard --logdir tb_logs/` | TF 原生，Graphs 标签 |
| **saved_model_cli** | `saved_model_cli show --dir ... --all` | 纯文本，可 grep |
| **JSON export** | `json_format.MessageToJson()` | 可 jq 检索 |

## 参考

- [checkpoint详解.md](checkpoint详解.md) — checkpoint 的内部结构对照
- [demo_two_tower.py](demo_two_tower.py) — 导出代码实现
