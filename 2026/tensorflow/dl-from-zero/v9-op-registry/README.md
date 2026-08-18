# v9-op-registry — op 注册系统 + 队列 + 稀疏张量 + 查找表

> 对应 TF commit 1 的 op 注册地基 + 数据管线机制 + 稀疏张量 + 查找表。
> v0-v8 用枚举 switch 实现 op，v9 改为**注册表驱动** (op 即数据)，并加入
> 搜广推/LLM 训练的核心数据结构。

## 核心概念 (meta 问题)

v0-v8 的 op 实现方式：
```cpp
enum NodeType { ADD, MUL, MATMUL, ... };  // 枚举
switch (node->type) {                      // 中心化 switch
    case ADD: ...
    case MUL: ...
}
```

问题：
- **中心化**：每加一个 op 都要改 `kernels.h` 的 switch
- **不可扩展**：用户无法注册自定义 op
- **元信息缺失**：没有 op 的输入/输出类型、属性定义

v9 改为**注册表驱动** (对应 TF 的 REGISTER_OP):
```cpp
OpRegistry::Register("MatMul")
    .Input("a: float")
    .Input("b: float")
    .Output("product: float")
    .Attr("transpose_a: bool = false")
    .SetShapeFn([](InferenceContext* c) { ... });

KernelRegistry::Register("MatMul")
    .Device(CPU)
    .TypeConstraint<float>("T")
    .Kernel([](OpKernelContext* ctx) { ... });
```

优势：
- **去中心化**：op 定义和 kernel 实现分离，可插拔
- **可扩展**：用户可注册自定义 op
- **元信息完整**：类型检查、形状推导、属性验证

---

## 四大功能

### 1. Op 注册系统 (`framework/op_registry.h`)

```cpp
class OpDef {
    string name;
    vector<ArgDef> inputs;   // name + type
    vector<ArgDef> outputs;
    vector<AttrDef> attrs;   // name + type + default
};

class OpRegistry {
    static void Register(const OpDef& def);
    static const OpDef* LookUp(const string& op_name);
};

// 宏简化注册
#define REGISTER_OP(name) \
    OpDefBuilder(name).Register()
```

### 2. 队列 + 数据管线 (`ops/queue_ops.h`)

**核心问题**：GPU 等数据 → CPU 成为瓶颈

**解决方案**：生产者/消费者解耦
- **FIFOQueue**: enqueue 满则阻塞，dequeue 空则阻塞
- **QueueRunner**: 后台线程不停 run enqueue 子图
- **Coordinator**: 管理多个 QueueRunner 生命周期

```cpp
// 队列 op
Node* queue = g.fifo_queue("q", capacity=100, dtype=FLOAT);
Node* enqueue = g.enqueue(queue, data);  // 满则阻塞
Node* dequeue = g.dequeue(queue);        // 空则阻塞

// QueueRunner: 后台线程喂数据
QueueRunner runner(sess, enqueue_ops);
runner.start();  // 启动生产者线程
// ... 训练循环消费 dequeue ...
runner.stop();   // 停止生产者
```

**对应**：
- TF `data_flow_ops.cc` (FIFOQueue/RandomShuffleQueue)
- `python/training/queue_runner.py` + `Coordinator`
- tf.data 前身，LLM dataloader 同一概念

### 3. 稀疏张量 (`framework/sparse_tensor.h`)

**核心问题**：搜广推特征稀疏（百万维只有几十个非零）

**解决方案**：SparseTensor = (indices, values, dense_shape)
```cpp
// 稠密: [0, 0, 3, 0, 0, 5]  → 6 个 float
// 稀疏: indices=[2, 5], values=[3, 5], shape=[6]  → 2 个 int + 2 个 float
```

**ops**：
- `sparse_matmul`: SparseTensor × DenseTensor
- `sparse_reduce_sum`: 沿某维度归约
- `sparse_to_dense`: 转回稠密

**对应**：TF `sparse_ops.cc` + `python/ops/sparse_ops.py`

### 4. 查找表 (`ops/lookup_ops.h`)

**核心问题**：特征 ID 映射（user_id → embedding_index）

**解决方案**：HashTable 资源
```cpp
// 创建查找表
Node* table = g.hash_table("user_table", key_dtype=INT64, value_dtype=INT32);
Node* init = g.table_init(table, keys, values);  // 初始化

// 查找
Node* ids = g.placeholder("user_ids", {batch_size});
Node* indices = g.table_lookup(table, ids);  // 批量查找
```

**对应**：TF `lookup_interface.h` + `data_flow_ops.cc` 的 LookupTable

---

## 实现文件

```
v9-op-registry/
├── framework/
│   ├── op_registry.h      op 注册表 (OpDef/OpDefBuilder)
│   ├── kernel_registry.h  kernel 注册表 (KernelDef/REGISTER_KERNEL)
│   ├── sparse_tensor.h    稀疏张量 (indices/values/shape)
│   └── op_kernel.h        OpKernelContext (统一 kernel 接口)
├── ops/
│   ├── queue_ops.h        队列 op (FIFOQueue/Enqueue/Dequeue)
│   ├── sparse_ops.h       稀疏 op (SparseMatMul/SparseReduceSum)
│   └── lookup_ops.h       查找表 op (HashTable/TableLookup)
├── core/
│   ├── queue_runner.h     QueueRunner (后台线程喂数据)
│   └── coordinator.h      Coordinator (管理多线程生命周期)
└── main.cpp               demo 10-13 验证

demo 10: op 注册表 (动态注册自定义 op)
demo 11: 队列管线 (生产者/消费者解耦, GPU 不等数据)
demo 12: 稀疏张量 (搜广推稀疏特征, 内存节省对比)
demo 13: 查找表 (特征 ID 映射, 批量查找性能)
```

---

## 与 TF 的对应

| v9 这里 | TF commit 1 |
|---------|-------------|
| `framework/op_registry.h` | `framework/op.cc` + `op_def_builder.cc` |
| `framework/kernel_registry.h` | `framework/op_kernel.h` + `kernel_def_builder.cc` |
| `ops/queue_ops.h` | `ops/data_flow_ops.cc` (FIFOQueue) |
| `core/queue_runner.h` | `python/training/queue_runner.py` |
| `framework/sparse_tensor.h` | `framework/tensor.h` (SparseTensor) |
| `ops/sparse_ops.h` | `ops/sparse_ops.cc` |
| `ops/lookup_ops.h` | `ops/data_flow_ops.cc` (LookupTable) |

---

## 今天价值 (2026)

- **op 注册系统**：PyTorch 的 `TORCH_LIBRARY`、JAX 的 primitive 注册、MLIR 的 op 定义，都是同一思想 —— "op 即数据"，框架核心是注册表 + 调度器
- **队列 + QueueRunner**：tf.data、PyTorch DataLoader、HuggingFace datasets 的共同原型 —— 生产/消费解耦，数据预处理与训练并行
- **稀疏张量**：搜广推特征建模（CTR 预估、推荐召回）、图神经网络 (GNN)、NLP 的词袋模型，稀疏是第一性需求
- **查找表**：Embedding 层的本质 —— user_id/item_id → dense vector，批量查找是推理性能关键

---

## 构建与运行

```bash
make && ./build/train
```

预期输出: demo 1-9 同 v8，demo 10-13 验证新功能
