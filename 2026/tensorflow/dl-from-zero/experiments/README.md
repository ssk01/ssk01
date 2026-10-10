# JIT / primitive cache 实验索引

这些实验为 [v10 primitive cache demo](../v10-primitive-cache/README.md) 提供依据。第一次阅读先看[项目与代码导览](../READING-GUIDE.md)，再读主线实验；设计背景见[四阶段规划](../primitive-cache-plan.md)。

关于 JIT 收益归因、Apple 内核面板对照和 oneDNN 成本模型，见[讨论总结](../JIT-MATMUL-COST-MODEL.md)。

| 目录 | 内容 | 在当前主线中的位置 |
| --- | --- | --- |
| [jit-matmul](jit-matmul/README.md) | LIBXSMM SME MatMul 对照 Accelerate，独立测量创建、代码生成、缓存容量 | 主线证据：矩阵各维度大于 100，FP32，M4 Pro |
| [onednn-primitive-cache](onednn-primitive-cache/README.md) | oneDNN MatMul 创建与执行路径探测 | 本机 FP32 落到参考 GEMM，BF16 无法创建，未作为主线加速证明 |
| [onednn-primitive-cache/ARM-JIT.md](onednn-primitive-cache/ARM-JIT.md) | Arm Shuffle/Reorder 的真实 JIT 与缓存诊断 | 辅助理解 oneDNN，不能替代 MatMul 对照 |
| [jit-matvec](jit-matvec/README.md) | 自写 NEON 生成器的小矩阵探索 | 不满足后续尺寸要求，保留过程，不作主线结论 |
| [primitive-cache-case/benchmark.swift](primitive-cache-case/benchmark.swift) | Metal pipeline 创建与复用实验 | 辅助材料，其创建成本不代表 CPU JIT 成本 |

主线阅读顺序：`jit-matmul/README.md` 的结果和测量口径 → `main.cpp` 的实验驱动 → [v10/primitive.h](../v10-primitive-cache/primitive.h) 的对象与 LRU → [v10/tf_matmul.h](../v10-primitive-cache/tf_matmul.h) 的框架适配。

实测 JSON 是重复测量的原始数据，第一遍无需逐项阅读。不同实验的布局、工作负载、计时范围不同，比较时以各自 README 为准。
