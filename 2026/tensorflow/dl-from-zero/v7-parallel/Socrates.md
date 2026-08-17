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

<!-- 以下继续记录 -->
