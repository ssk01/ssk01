# Socrates.md - 问答记录

### Q: 7c 基准计时为什么恒 0? 是 apple clang 17 的优化 bug 吗?
不是编译器 bug, 是手写 asm 缺 clobber 声明。真凶链: `sme2_mv_za` 的 asm 块只声明了 z0-z2/p0/x9/w11-w15 clobber, 但 Apple M4 上每次 `smstart sm`/`smstop sm` (即使不含任何 ZA 操作) 都会清零 callee-saved 的 d8-d15 (V8-V15, 实测非首次、每次都清)。编译器以为 d8-d15 跨调用安全, 把 best (d8) 和 1e6 除数 (d9) 缓存在里面 → sme2_gemm 调用后 d9=0 → `fdiv` 除零得 inf, best 恒 0。修复: asm 块 clobber 加 v8-v15。曾误判为 "lambda 内联展开破坏 best 初始化" (noinline 化、反汇编比对都不解决问题, 因为汇编逻辑本身正确), 最后用 asm 读寄存器跨调用隔离实验才定位。
(2026-08-17)

### Q: 为什么反汇编里同一函数有 .specialized.1 版本?
clang -O2 对 noinline 函数做常量传播特化 (function specialization): 4 个调用点中传函数指针 `gemm_scalar` 的两个被克隆成 specialized.1 版 (kern 直接内联), 传 `sme2_gemm` 的仍走原始函数指针版。所以 4 个 bench 调用交错使用两个不同版本, 同一函数体在两份汇编里优化布局不同。
(2026-08-17)

### Q: 在 streaming mode 里调用普通函数会怎样?
SIGILL (非法指令): SM=1 时只能执行 streaming-compatible 代码, 调非 SC 的外部函数 (printf 等) 直接崩溃; 同理 SM=0 时执行 `smstop` 也是 SIGILL (非法指令)。asm 里的 smstart/smstop 必须配对且退出后再调外部函数。
(2026-08-17)

### Q: M4 上 smstart/smstop 为什么清 d8-d15? 是所有 Apple Silicon SME 的通用行为吗?
实测: 每次 SM 状态切换后 d8-d15 全为 0, 与是否首次、是否含 ZA 操作无关。机制疑似 Apple SME 状态保存/恢复路径的副作用 (V 寄存器跨 SM 边界的语义由 Apple 实现决定), 未必在别的 SME 硬件上复现。对用户代码的含义: 任何包含 smstart/smstop 的内联 asm 都必须把 v8-v15 列入 clobber, 让编译器负责保存/恢复调用者的寄存器。
(2026-08-17)

### Q: 为什么固定 worker 池的 executor 同时只能跑一个 infer 请求? (convoy)
worker 池模型每次 Run 提交 N 个 WorkerLoop, loop 在共享就绪队列空时 park 等任务。一个 run 卡在 GPU 异步等待 (就绪队列已空) 时, 其全部 worker park, **独占池线程**; 其他并发 run 的任务在池队列里饿死 —— 并发 run 退化为串行, 且是霸占式。TF 正版 SimpleExecutorState 无此问题: 它不 park, 没有就绪节点时计算链终止、线程归还池。修复 = closure 模型 (对齐 SimpleExecutorState): 就绪节点直接作为闭包提交线程池, 便宜节点内联到当前线程的 inline 队列 (tail-call 语义), 空闲 run 占用线程数为 0。实测 (混设备图 × 8 worker × 8 线程并发): 旧版 933 ms (串行) → 新版 293 ms (3.2x, GPU 等待与其他 run 的 CPU 计算重叠)。
(2026-08-17)

### Q: closure 模型重构后, 纯 CPU 小图并发推理为什么反而略慢? 修复白做了吗?
不是白做: 调度开销换并发性。demo 3 推理图是 5 节点链 (3 根节点), 每次 run 新版提交 3 个闭包 (3 次锁+cv 原语), 旧版提交 1 个 WorkerLoop; 40000 次 run 多 8 万次原语 ≈ +15% 耗时 (103 → 120 ms)。旧版在这个场景「快」是因为它本来就串行 (1 worker 下无并发可被牺牲); convoy 收益只出现在 GPU 异步等待场景 (demo 8 的 3.2x)。两种模型各有适用面, 数字如实记录在 README。
(2026-08-17)

### Q: 为什么 closure 版 executor 首次并发推理会 SIGSEGV? (提交循环竞态)
use-after-free 链: Run 的根节点提交循环**无锁**遍历 plan.order 边提交闭包边读 es->pending; 闭包入池后池线程立即执行并递减 pending, 提交循环把「已被并发递减到 0 的非根节点」误判为根节点**重复提交** → op 重复执行 → outstanding 提前归零 → Finish (delete es) 发生在提交循环运行期间 → 提交循环读已删除的 es。修复: 两阶段提交 —— 先持锁把根节点收集进局部 vector, 再提交闭包 (收集发生在任何闭包入池之前, pending 无并发修改)。调试过程: TSan 与 Metal 不兼容、ASan 太慢、-O0 不复现 (时序相关, 仅 -O2) → instrumented 日志证明复现链。
(2026-08-17)

### Q: demo 5 执行队列的加速比为什么只有 ~1.0x? 并行白做了吗?
不是调度问题, 是 matmul kernel 被 ZA 硬件串行化了: demo 5 的 8 个 matmul 走 SME2 (kernels.h 默认), 而 M4 的 ZA/AMX 在并发 streaming 线程间不隔离, `sme2_gemm` 必须用全局 `za_mutex` 互斥 —— 8 个 matmul 在锁上排队串行 (实测 ~27ms ≈ 8 × 单次 3.3ms, 1 worker 和 14 worker 同速, 加速比 ~1.01x)。修复: CPU matmul 加 `matmul_use_sme2` 开关, demo 5 关掉 SME2 走无锁标量路径, 并发真正展开 → 加速比 1.01x → 5.5x (1w 14.0ms vs 14w 2.5ms, 逐位一致)。SME2 速度由 demo 7 单独演示 (2.7x)。取舍: 一个 demo 只展示一件事 —— demo 5 的舞台是调度, kernel 速度是 demo 7 的舞台。
(2026-08-17)

<!-- 以下继续记录 -->
