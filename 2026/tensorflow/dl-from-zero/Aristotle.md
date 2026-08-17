# Aristotle.md - 用户发言记录

- dl-from-zero 是一个大目录, 下面四个不同版本; 写一个总的文档介绍这四个项目 (2026-08-03) [decision]
- 下一个 feature 选题的文档: 核心概念要写详细, feature 应该是从 0 到 1 过程中解决了很多 meta 问题的东西——先讲卡在哪, 为什么, 怎么样 (2026-08-12) [decision]
- 选题三标准: 从0到1过程中需要的, 解决了问题, 且对今天(2026, 搜广推+大模型)仍具实践价值 (2026-08-12) [decision]
- 继续下面一个: v6 = 量化/低精度推理, 已实现并提交到 v6-quant (2026-08-13) [decision]
- 下一个 feature: v7 = 执行队列 (op 并发执行) + op 可以跑在 Mac 的 GPU 上 (Metal) + 可以利用 AMX, 参照 TF commit 1 (executor/threadpool/simple_placer/GPU) 实现 (2026-08-14) [decision]
- AMX 路线: SME2 intrinsics 手写 GEMM (arm_sme.h, mopa 外积; fallback cblas_sgemm) (2026-08-14) [decision]
- GPU 路线: 手写 Metal compute shader (ObjC++ .mm 用 Metal.h —— Metal.hpp 未随 Xcode 提供) (2026-08-14) [decision]
- Plan agent 太慢卡死, kill 掉, 直接在当前会话里做 plan (2026-08-14) [decision]

<!-- 以下继续记录 -->
