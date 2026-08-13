# TF 初始 commit feature 筛选清单 — dl-from-zero 后续版本选题

> 考证基准: TensorFlow 初始 commit `f41959ccb2d`(2015-11-06) + 论文 arXiv:1605.08695
> 筛选标准(用户定): ① 从 0 到 1 过程中真实需要的 ② 解决了问题的 ③ 对今天(2026, 搜广推 + 大模型)仍具实践价值

---

## 一、初始 commit 全部 feature 盘点(共 22 项未实现)

### A. 图/执行器层(4 项)

| # | Feature | 初始 commit 位置 | 今天价值(搜广推/LLM) | 兴趣 |
|---|---|---|---|---|
| A1 | **控制流原语 + frame 机**(Switch/Merge/Enter/Exit/NextIteration/LoopCond) | control_flow_ops.cc + executor.cc FrameState(2118 行) | 低(模型层框架已封装) | — |
| A2 | **图优化: CSE + 常量折叠** | optimizer_cse.cc(`OptimizeCSE` + consider_fn); 常量折叠 2016-01-25 加入 | ✅✅ LLM 推理优化核心(torch.compile/TensorRT); 搜广推推理延迟 | **有兴趣** |
| A3 | **动态子图执行(feed/fetch 每步指定)** | subgraph.cc(v1 只做静态 prune) | ✅ 训练/推理共用图, 线上按需取子图 | — |
| A4 | 成本模型 costmodel | costmodel.cc/costutil.cc | 低(调度优化, 系统内部) | — |

### B. 设备/分布式层(3 项)

| # | Feature | 位置 | 价值 | 兴趣 |
|---|---|---|---|---|
| B1 | **设备抽象 + 放置**(device/device_mgr/simple_placer) | device.cc/simple_placer.cc | ✅ 多卡训练、异构部署基础概念 | — |
| B2 | **图分区 + Send/Recv + rendezvous**(单机多设备切分) | graph_partition.cc + sendrecv_ops + rendezvous.cc | ✅ PS 训练机制(v2 地基, 闭环 v2) | — |
| B3 | FunctionLibrary(_Arg/_Retval/SymbolicGradient) | function.cc/h | ⚠️ TF2 tf.function 后代, 框架内部 | — |

### C. 状态/数据层(3 项)

| # | Feature | 位置 | 价值 | 兴趣 |
|---|---|---|---|---|
| C1 | **队列 + 数据管线**(FIFOQueue/RandomShuffleQueue + QueueRunner/Coordinator/input.py) | data_flow_ops.cc + python/training/queue_runner.py | ✅✅ 搜广推特征管线/tf.data 前身; LLM dataloader 同一概念 | — |
| C2 | 稀疏张量表示(SparseTensor = indices+values tuple) | sparse_ops.cc + python/ops/sparse_ops.py | ✅ 搜广推稀疏特征建模 | — |
| C3 | 查找表(LookupTable/HashTable 资源) | data_flow_ops.cc + lookup_interface.h | ⚠️ 特征 ID 映射场景 | — |

### D. 注册/元系统层(3 项)—— dl-from-zero 与真实 TF 差距最大的地基

| # | Feature | 位置 | 价值 | 兴趣 |
|---|---|---|---|---|
| D1 | **op 注册系统**(REGISTER_OP/OpDef/KernelDef/属性系统) | op.cc/op_def_builder.cc/kernel_def_builder.cc | ✅ 框架地基: 枚举 switch → 注册表("op 即数据") | — |
| D2 | shape 推断(OpShapeInferenceFn/common_shapes.py) | op.h + python/ops/common_shapes.py | ⚠️ 图构造期检查, 偏内部 | — |
| D3 | **GraphDef 序列化 + importer**(图 ↔ proto) | graph.proto + graph_constructor.cc + python/framework/importer.py | ✅✅ 图的存储/传输/上线载体 | — |

### E. 应用类 ops/kernels(5 项)

| # | Feature | 位置 | 价值 | 兴趣 |
|---|---|---|---|---|
| E1 | 张量操作全集(Gather/Concat/Slice/Reshape + linalg/image/string/parsing/io/random) | array_ops.cc 等 20 个 ops 文件 + 323 kernels | ✅ 通用(embedding_lookup 的 Gather 在 E2) | — |
| E2 | **embedding_lookup**(含 sparse 版本)—— embedding 分片基础原语 | python/ops/embedding_ops.py + array_ops Gather | ✅✅ 搜广推超大 embedding 核心原语 | — |
| E3 | summary/TensorBoard | summary_ops.cc + events_writer | ⚠️ 训练可视化 | — |
| E4 | GPU kernels(cuda) | kernels/*.cu.cc + gpu/ | ⚠️ 需硬件, 概念被 B1 覆盖 | — |
| E5 | **量化/低精度**(⚠️ 不在初始 commit 代码, 论文 v2 2016-05 描述 gemmlowp) | 论文 §5 quantization 段落 | ✅✅✅ LLM 推理绝对核心(GPTQ/AWQ/GGUF/int8); 搜广推 int8 上线 | **有兴趣** |

### F. Python 层(4 项)

| # | Feature | 位置 | 价值 | 兴趣 |
|---|---|---|---|---|
| F1 | variable_scope/变量作用域 | python/ops/variables.py + variable_scope.py | ⚠️ 建模习惯, 概念被 v3 覆盖大半 | — |
| F2 | optimizer 体系(GD/Momentum/Adagrad/RMSProp/Adam/Ftrl 基类) | python/training/optimizer.py 等 | ✅ 优化器实现模式(v2 有简化版) | — |
| F3 | saver/checkpoint | saver.py + saver.proto | 用户已否(学习 demo 无生产需求) | **已排除** |
| F4 | 训练工具套件(learning_rate_decay/moving_averages/input/training_util) | python/training/ | ⚠️ 工程便利件 | — |

---

## 二、筛选结论

### v5 = 图优化(A2: CSE + 常量折叠)

- **用户兴趣** ✅ + **三标准全过**: 论文 §5 明确 master 在优化阶段做 CSE/常量折叠; 初始 commit 实有 `optimizer_cse.cc`
- **解决了问题**: 图能跑 ≠ 图高效——同一子表达式被多个消费者各算一遍(CSE); 常量表达式每次 run 重算(常量折叠)
- **今天价值**: torch.compile / TensorRT / XLA 的前身——所有现代推理引擎的图优化管线都包含这两步; LLM 推理延迟优化核心
- 时间线考证: CSE 在初始 commit(2015-11-06); 常量折叠 2016-01-25 加入(commit `71184628900`, "Creates a local executor and executes a copy of the constant slice")

### v6 = 量化/低精度(E5)

- **用户兴趣** ✅; 论文 v2(2016-05)§5 描述(gemmlowp int8 内核)
- **诚实标注**: 不在初始 commit 代码里, 属开源后 1→N 早期
- **今天价值**: LLM 推理绝对核心(GPTQ/AWQ/GGUF/int8), 搜广推 int8 上线——三标准里的"今天价值"最强

### 排除项

| 排除 | 理由 |
|---|---|
| F3 checkpoint/saver | 用户明确否: "v5 没意义, 我是学习性demo, 不是完整的生产工具" |
| A1 控制流 | 用户评估: 搜广推/Transformer 模型用不到(现代框架已封装 RNN/动态控制流); 实现成本最高(frame 机 2118 行) |
| A4 costmodel / B3 FunctionLibrary / D2 shape 推断 / E3 summary / E4 GPU / F4 训练工具 | 偏系统内部或工程便利件, 概念价值低, 今天实践价值弱 |

---

## 三、路线图

```
v5  图优化(CSE + 常量折叠)   ← 当前版本, 已完成
v6  量化/低精度              ← 下一版(论文 v2 描述, 今天 LLM 推理核心)

后续候选(符合三标准, 未排期):
  C1  队列 + 数据管线        (搜广推特征管线/tf.data 前身)
  B1+B2  设备抽象 + 分区 Send/Recv  (v2 PS 的地基, 闭环分布式概念)
  E2  embedding_lookup/分片  (搜广推超大 embedding 核心原语)
  D1  op 注册系统            (dl-from-zero 与真实 TF 差距最大的地基)
  A3  动态子图执行           (feed/fetch 每步指定)
  D3  GraphDef 序列化        (图的存储/传输载体)
```
