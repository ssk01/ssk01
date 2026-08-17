# v7-parallel — 并发执行 + 多设备 (dl-from-zero 系列第 7 版)

> 对应 TF commit 1 的 `executor.cc`(2118 行)+ `simple_placer.cc` + threadpool + GPU 设备。
> v5/v6 解决"图怎么优化/量化", v7 解决"图怎么**跑得快**": 一次 run 内 op 级并发、
> op 按设备放置、CPU 用上 AMX、GPU 用 Metal —— 四件事对应 TF 从单线程解释器走向
> 生产执行引擎的四块地基。

## 核心概念(meta 问题)

前几版的"并发"是**图级别的**: v3/v4 把状态挪出图, 让同一张图被多线程同时 run。
v7 把并行粒度下到 **op 级别**: 一次 run 内部, 互不依赖的 op 同时执行。这是 TF 从
"demo 解释器"到"生产执行引擎"的关键一跃, 需要四块拼图:

1. **执行队列** (`core/executor.h`): 就绪队列 + pending 计数。每个 op 完成后给
   消费者递减计数, 归零的 op 进就绪队列被 worker 池消费。对应 commit 1 的
   `ExecutorState` + `ScheduleReady`。
2. **线程池** (`core/threadpool.h`): 手写 worker 池 (mutex + deque + CV), 不用
   `std::async` —— 池线程持久, 任务分派零线程创建开销。对应 commit 1 的
   `lib/core/threadpool.{h,cc}`。
3. **设备放置** (`core/place.h`): 每个 Node 带 device 字符串; 显式指定优先,
   未指定 → 按 kernel 注册表 (`HasMetalKernel`) 筛候选设备再按优先级选。
   对应 commit 1 的 `simple_placer.cc` + `FilterSupportedDevices`。
4. **设备 kernel**: CPU 设备补手写 SME2 asm GEMM (M4 AMX, 对应 commit 1 的
   Eigen matmul); GPU 设备用手写 Metal compute shader (对应 commit 1 的
   `gpu_device.cc`, async dispatch + done 回调)。

## 执行队列 (`core/executor.h`)

```
GraphPlan (编译一次, Session 缓存)     ExecState (每次 run)
  order         拓扑序                     pending[id]    剩余未满足入边
  consumers     id → 消费者列表            ready          就绪队列 (共享)
  in_degree     pending 初值               outstanding    未完成 op 数 (原子)
  devices       placer 产物                inline_q       每 worker 内联队列
```

- **就绪判定**: `--pending[consumer] == 0` → 就绪。op 完成时 `outstanding` 原子减一,
  归零即整次 run 完成 (与 commit 1 的 `num_outstanding_ops_` 同款)。
- **ScheduleReady 按开销分流** (commit 1 用 Eigen CostModel, 这里退化为按类型):
  逐元素/标量 op 进当前 worker 的**内联队列** (tail-call 语义, 零锁);
  matmul 族 (O(N·F)) 进共享就绪队列由 worker 池竞争。
- **GPU 异步 op**: kernel 立即返回, done 回调从 Metal 线程传播输出 —— 与 CPU op
  重叠执行。回调路径与同步 op 共用 `PropagateOutputs`。
- **主线程完成等待在独立 `done_cv` 上**: 若与 worker 共用 cv, GPU 完成回调的
  `notify_one` 可能唤醒主线程 (谓词不满足又睡回) 而漏掉该醒的 worker → 饿死死锁
  (接真 Metal 时实际复现过, 头注释有记录)。

## 设备放置 (`core/place.h`)

```
SimplePlace: 每个节点
  显式 device (g.on(n, GPU/CPU)) → 直接采用
  未指定 (AUTO) → HasMetalKernel(type) ? GPU : CPU
                  (kernel 注册表筛选 + 优先级; 无 GPU kernel 的 op 永远只进
                   CPU 候选 —— soft placement 在 demo 里的体现)
```

## SME2/AMX CPU matmul (`kernels/sme2_gemm.h`)

手写 asm: `fmopa za0.s, p0/m, p0/m, z1.s, z0.s` 外积累加进 ZA tile (M4 AMX)。
形状分派: N==1 (框架的 [N,F]×[F]→[N] matvec) 用 4×1 tile, N%4==0 用 4×4 tile
(每指令 16 MAC), 其余回退标量。**踩坑记录全在头注释**, 三个必须知道的:

1. **st1w 存整个 ZA tile**: M4 上 `st1w {z2.s}, p0, [c]` 实际存 64B 完整 4×4 tile
   (列主序), 列 1..3 是幽灵位移点积 → 每次窗口越界写 48B。改 `str q2` 只写单列。
2. **M4 每次 smstart/smstop 清零 callee-saved d8-d15**: 实测每次 SM 切换
   (即使不含 ZA 操作) 后 d8-d15 全 0。asm 必须声明 `v8`-`v15` clobber, 否则调用者
   跨调用缓存的常量被静默清零 → 除零 inf、计时恒 0 (7c 基准曾因此坏掉)。
3. **ZA 多线程并发互相覆盖**: 两个 streaming 线程同时 `smstart za` + `mopa` 时
   结果互相污染。串行化 ZA 段 (mutex), 正确性优先, 并发 matmul 变顺序。

## Metal GPU (`kernels/metal_matmul.metal` + `core/metal_matmul.mm`)

- ObjC++ (Metal.hpp 未随 Xcode 提供 → 用 Metal.h)
- 共享存储 buffer (Apple 统一内存, host/GPU 同一物理内存 → 免拷贝)
- `MatmulAsync`: dispatch 后立即返回, done 从 Metal 内部线程调用 (executor 的
  GPU op 回调路径)
- 无 Metal 环境 → `Available()` 返回 false → placer 全部 CPU 放置

## demo 与结果

```bash
make && ./build/train
```

| demo | 内容 | 结果 |
|------|------|------|
| 1-4 | 继承 v5: CSE / 常量折叠 / prune 串联 | 同 v5 |
| 5 | 执行队列: 8 独立 matmul 分支 + 树状合并, 1 worker vs 硬件并发数 | 输出逐位一致; **加速比 0.997x** (诚实: matmul 走 SME2 且 ZA 段 mutex 串行化, 并行被串行抵消) |
| 6 | 设备放置: matmul 显式 GPU (Metal 异步) + 逐元素自动 CPU | 混设备 0.786 ms vs 全 CPU 0.991 ms (小形状 dispatch 开销占比大, 诚实数字) |
| 7 | SME2/AMX 实测 | 7a/7b 逐位一致; 7c matvec 0.49x (单列 ZA 1/4 tile + 未 unroll), **gemm 2.76x / ~9 GFLOPS** (4×4 tile 16 MAC/指令) |

demo 7 的诚实数字: matvec 用 ZA 一列 (1/4 tile), asm 首版未 unroll (每 k 8 条
insr 串行), SME2 反而慢; GEMM 4×4 每指令 16 MAC, 才是 AMX 的正确打开方式。

## 与 TF 的对应

| v7 这里 | TF commit 1 |
|---------|-------------|
| `core/executor.h` (GraphPlan/ExecState) | `core/common_runtime/executor.cc` (ExecutorState, 2118 行) |
| `ScheduleReady` 按开销分流 | 同款: cheap inline / expensive ready queue |
| `core/threadpool.h` | `lib/core/threadpool.{h,cc}` |
| `core/place.h` + `HasMetalKernel` | `common_runtime/simple_placer.cc` + `FilterSupportedDevices` |
| CPU matmul → 手写 SME2 asm | CPU matmul → Eigen |
| GPU matmul → Metal shader + done 回调 | `common_runtime/gpu/gpu_device.cc` (ComputeAsync) |
| `g.on(n, GPU)` + AUTO 自动放置 | NodeDef.device + 未指定自动放置 |
| ZA mutex 串行化 (并发安全) | commit 1 无 ZA 问题, 但有设备间 Send/Recv 同步 |

## 今天价值 (2026)

- 执行队列的 pending 计数 + 就绪队列是**所有并发图引擎的骨架** (TF 至今、
  PyTorch 的 eager 并发、DAG 调度器), 20 行看懂生产引擎的核心机制
- simple_placer 的 "显式优先 + kernel 注册表自动放置" 仍是现代框架放置语义的
  原型 (K8s 调度也是同一思想: 显式 nodeSelector 优先, 否则按资源/亲和性选)
- M4 AMX 通过 SME2 `fmopa` 一条指令打满外积 —— 手写 SIMD 之上, 还有一层
  "一条指令一个 tile" 的硬件矩阵单元; 踩坑 (ZA 越界、SM 切换清寄存器) 是
  Apple SME 实机行为的稀缺一手记录
- GPU 统一内存 + Metal async dispatch 的重叠执行, 是今天 M 系列 Mac 上
  CPU/GPU 协作的典型形态
