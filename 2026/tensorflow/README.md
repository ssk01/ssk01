# TensorFlow 学习项目

## dl-from-zero/

从 0 到 1 复刻 TensorFlow 概念的总目录 — v0～v10 共 11 个演进版本, 每版解决一个 TF 核心设计问题:

| 版本 | 解决什么 |
|---|---|
| [v0-linear](dl-from-zero/v0-linear/) | 最小骨架: Graph/Node/手写 autodiff/Session, 跑通线性回归 |
| [v1-prune](dl-from-zero/v1-prune/) | 图裁剪(训练图→推理图) + Python binding + 信用卡欺诈逻辑回归 |
| [v2-ps-worker](dl-from-zero/v2-ps-worker/) | PS/Worker 分布式训练(单机多进程, 同步/异步) |
| [v3-concurrent](dl-from-zero/v3-concurrent/) | 图/运行时状态分离, 同一张图可并发推理 |
| [v4-grad](dl-from-zero/v4-grad/) | 独立梯度子图(对应 TF gradients.py) |
| [v5-opt](dl-from-zero/v5-opt/) | 图优化：CSE、常量折叠 |
| [v6-quant](dl-from-zero/v6-quant/) | int8 量化推理 |
| [v7-parallel](dl-from-zero/v7-parallel/) | 执行队列、设备放置、SME2/AMX、Metal |
| [v8-graph-partition](dl-from-zero/v8-graph-partition/) | 图分区、Send/Recv、跨设备传输 |
| [v9-op-registry](dl-from-zero/v9-op-registry/) | op/kernel 注册与分发 |
| [v10-primitive-cache](dl-from-zero/v10-primitive-cache/) | 二维 MatMul、JIT primitive 生命周期、LRU 与命中统计 |

总文档：[dl-from-zero/README.md](dl-from-zero/README.md)。

本次新增功能从[阅读导览](dl-from-zero/READING-GUIDE.md)开始，了解改动文件和调用链；[实验索引](dl-from-zero/experiments/README.md)区分主线证据与探索材料；[规划文档](dl-from-zero/primitive-cache-plan.md)记录四阶段设计和官方依据。主线复用 LIBXSMM 代码生成器，验证 FP32 MatMul 与代码缓存，再接入教学框架；当前不包含 BF16 或并发验证。

## two_tower_demo/

双塔模型 TF 文件格式详解：checkpoint 内部结构（V2 索引/数据/对象图）、saved_model.pb（MetaGraphDef/SignatureDef/FunctionDef）、TFRecord 格式。附可运行的训练→导出→验证 Demo。

## compress_demo/

用户特征压缩 5 种实现对比：Keras 原生 / pb 改写 / DeepRec 风格 graph 改写 / BFS 自动边界检测 / serving benchmark。

## xla_demo/

XLA JIT 编译 benchmark — eager / tf.function / jit_compile 三种推理模式延迟对比、op 融合原理。
