# Aristotle.md - 用户发言记录

- 下一个 feature: v7 = 执行队列 (op 并发执行) + op 可以跑在 Mac 的 GPU 上 (Metal) + 可以利用 AMX, 参照 TF commit 1 (executor/threadpool/simple_placer/GPU) 实现 (2026-08-14) [decision]
- AMX 路线: SME2 intrinsics 手写 GEMM (arm_sme.h, mopa 外积; fallback cblas_sgemm) (2026-08-14) [decision]
- GPU 路线: 手写 Metal compute shader (ObjC++ .mm 用 Metal.h —— Metal.hpp 未随 Xcode 提供) (2026-08-14) [decision]
- Plan agent 太慢卡死, kill 掉, 直接在当前会话里做 plan (2026-08-14) [decision]
- 编译失败 —— 我的删除脚本把 `sme2_mv_za` 的函数结尾 `}` 也删掉了。立即检查并修复 (2026-08-17)
- 接下去: 集成 kernels.h CPU matmul + Makefile `-mcpu=apple-m4`, main.cpp 加 demo 7: SME2/AMX 实测 (2026-08-17)

<!-- 以下继续记录 -->
