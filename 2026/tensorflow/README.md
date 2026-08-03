# TensorFlow 学习项目

## dl-from-zero/

从 0 到 1 复刻 TensorFlow 概念的总目录 — 5 个演进版本, 每版解决一个 TF 核心设计问题:

| 版本 | 解决什么 |
|---|---|
| [v0-linear](dl-from-zero/v0-linear/) | 最小骨架: Graph/Node/手写 autodiff/Session, 跑通线性回归 |
| [v1-prune](dl-from-zero/v1-prune/) | 图裁剪(训练图→推理图) + Python binding + 信用卡欺诈逻辑回归 |
| [v2-ps-worker](dl-from-zero/v2-ps-worker/) | PS/Worker 分布式训练(单机多进程, 同步/异步) |
| [v3-concurrent](dl-from-zero/v3-concurrent/) | 图/运行时状态分离, 同一张图可并发推理 |
| [v4-grad](dl-from-zero/v4-grad/) | 独立梯度子图(对应 TF gradients.py) |

总文档:[dl-from-zero/README.md](dl-from-zero/README.md)

## two_tower_demo/

双塔模型 TF 文件格式详解：checkpoint 内部结构（V2 索引/数据/对象图）、saved_model.pb（MetaGraphDef/SignatureDef/FunctionDef）、TFRecord 格式。附可运行的训练→导出→验证 Demo。

## compress_demo/

用户特征压缩 5 种实现对比：Keras 原生 / pb 改写 / DeepRec 风格 graph 改写 / BFS 自动边界检测 / serving benchmark。

## xla_demo/

XLA JIT 编译 benchmark — eager / tf.function / jit_compile 三种推理模式延迟对比、op 融合原理。
