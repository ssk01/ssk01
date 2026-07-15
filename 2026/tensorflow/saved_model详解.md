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

正向推理计算图，包含所有 op 和 tensor 连接。本 demo：

```
137 个 op
├── ReadVariableOp:   56 个
├── VarHandleOp:      42 个
├── VarIsInitializedOp: 14 个
├── AssignVariableOp: 14 个
├── Placeholder:       5 个
├── StatefulPartitionedCall: 4 个
└── NoOp, Const

5 个 library functions:
├── __inference___call___           → 38 ops (正向推理)
├── __inference__traced_save       → 128 ops
├── __inference__traced_restore    → 63 ops
└── 2 个 signature_wrapper         → 各 3 ops
```

**正向推理 op 序列 (38 个 op)：**

```
user_feat → MatMul → BiasAdd → Relu → MatMul → BiasAdd
item_feat → MatMul → BiasAdd → Relu → MatMul → BiasAdd
         → ConcatV2 → MatMul → BiasAdd → Relu → MatMul → BiasAdd → Relu
         → MatMul → BiasAdd → Squeeze → Sigmoid → output
```

这就是导出时「图裁剪」的结果——没有 `GradientTape`、反向传播、optimizer 更新，只有从输入到输出的正向路径。

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
