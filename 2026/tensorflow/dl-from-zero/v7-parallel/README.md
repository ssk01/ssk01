# v7-parallel — 并发执行 + 多设备 (dl-from-zero 系列第 7 版)

> 对应 TF commit 1 的 `executor.cc`(2118 行)+ `simple_placer.cc` + threadpool + GPU 设备。
> v5/v6 解决"图怎么优化/量化", v7 解决"图怎么**跑得快**": 一次 run 内 op 级并发、
> op 按设备放置、CPU 用上 AMX、GPU 用 Metal —— 四件事对应 TF 从单线程解释器走向
> 生产执行引擎的四块地基。
>
> **v8 更新 (2026-08)**: executor 从固定 worker 池重写为 TF 正版同款 **closure 模型**
> (对齐 SimpleExecutorState, `executor.cc:1567-1982`), 修复并发 GPU 推理的 **convoy**
> 问题 —— demo 8 实测 8 线程并发混设备推理 933 ms → 293 ms (3.2x); 过程中复现并
> 修复一个提交循环竞态 (SIGSEGV)。详见「convoy 修复 (v8)」。

## 核心概念(meta 问题)

前几版的"并发"是**图级别的**: v3/v4 把状态挪出图, 让同一张图被多线程同时 run。
v7 把并行粒度下到 **op 级别**: 一次 run 内部, 互不依赖的 op 同时执行。这是 TF 从
"demo 解释器"到"生产执行引擎"的关键一跃, 需要四块拼图:

1. **执行队列** (`core/executor.h`): pending 计数 + 就绪判定。每个 op 完成后给
   消费者递减计数, 归零的 op 进就绪集被调度。对应 commit 1 的
   `ExecutorState` + `ScheduleReady`。v8 起为 closure 模型 (见下节)。
2. **线程池** (`core/threadpool.h`): 手写 worker 池 (mutex + deque + CV), 不用
   `std::async` —— 池线程持久, 任务分派零线程创建开销。对应 commit 1 的
   `lib/core/threadpool.{h,cc}`。
3. **设备放置** (`core/place.h`): 每个 Node 带 device 字符串; 显式指定优先,
   未指定 → 按 kernel 注册表 (`HasMetalKernel`) 筛候选设备再按优先级选。
   对应 commit 1 的 `simple_placer.cc` + `FilterSupportedDevices`。
4. **设备 kernel**: CPU 设备补手写 SME2 asm GEMM (M4 AMX, 对应 commit 1 的
   Eigen matmul); GPU 设备用手写 Metal compute shader (对应 commit 1 的
   `gpu_device.cc`, async dispatch + done 回调)。

## 执行队列 (`core/executor.h`)—— v8 closure 模型

```
GraphPlan (编译一次, Session 缓存)     ExecState (每次 run, 堆分配)
  order         拓扑序                     pending[id]    剩余未满足入边
  consumers     id → 消费者列表            outstanding    未完成 op 数 (原子)
  in_degree     pending 初值               done_cb        完成通知 (捕获 Run 的 Sync)
  devices       placer 产物
```

- **就绪判定**: `--pending[consumer] == 0` → 就绪。op 完成时 `outstanding` 原子减一,
  归零的调用者回收 ExecState + 通知主线程 (对齐 TF `Finish()` 的 `delete this`,
  `executor.cc:1973-1982` —— es 由最后一个 op 完成者 delete, 主线程的等待点在
  Run 栈上的独立 Sync 对象, 两者互不悬垂)。
- **调度 = 闭包提交**: 就绪节点作为闭包提交线程池。便宜节点 (逐元素/标量) 进当前
  线程的**内联队列** (tail-call 语义, 零锁), 昂贵节点 (matmul 族, O(N·F)) 走线程池;
  一批内最后一个昂贵节点在内联队列空时直接内联 (Tail recursion optimization,
  `executor.cc:1732-1742`)。没有 cv-wait: 没有就绪节点时这条计算链就终止、线程
  归还池 —— 空闲 run 占用线程数为 0。
- **GPU 异步 op**: kernel 立即返回, done 回调从 Metal 线程传播输出 (回调路径
  `inline_q=nullptr`, 就绪节点全走线程池)。CPU 兜底路径 (非 matmul / 形状不符 /
  Metal 不可用) 会**同步**调 done → 可能直接在提交线程上触发 Finish (delete es),
  GPU 分支因此必须在提交后立即 return。

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

## convoy 修复 (v8) —— 诚实结果

**问题** (v7 初版固定 worker 池): 每次 Run 提交 N 个 WorkerLoop, loop 在共享
就绪队列空时 park 等任务。一个 run 卡在 GPU 异步等待 (共享队列已空) 时, 其全部
worker park, **独占池线程**; 其他并发 run 的任务在池队列里饿死 —— 行为上趋近
「一次只能跑一个 infer 请求」, 且是霸占式 (TF 正版无此问题: 它不 park)。

**修复**: closure 模型 —— 没有 cv-wait, 就绪节点直接作为闭包提交池, 空闲 run
线程占用为 0 (对齐 SimpleExecutorState)。**demo 8 实测**: 混设备图 (GPU matmul
异步 + CPU 逐元素) × 8 worker × 8 线程并发:

```
8 threads x 200 runs:  旧版 ≈ 933 ms (串行: 一个 run 卡 GPU 等待, 其余饿死)
                       新版 ≈ 293 ms (GPU 等待与其他 run 的 CPU 计算重叠)  ≈ 3.2x
```

**过程踩坑** (都值得记):

1. **提交循环竞态 → SIGSEGV**: 新版首跑在并发推理 demo 间歇崩溃。根因: Run 的
   根节点提交循环**无锁**边提交闭包边扫 `es->pending`; 闭包入池后池线程立即执行
   并递减 pending, 提交循环把「已被并发递减到 0 的非根节点」误判为根节点**重复
   提交** → op 重复执行 → outstanding 提前归零 → Finish (delete es) 发生在提交
   循环运行期间 → use-after-free。修复: 先持锁把根节点收集进局部 vector, **之后**
   再提交闭包 (两阶段提交)。调试过程: TSan 与 Metal 不兼容、ASan 太慢 (40000 次
   Metal dispatch 跑不完)、-O0 不复现 (时序相关, 仅 -O2) → 最终用 instrumented
   日志证明复现链 (日志显示 `Run schedule id=3` 出现在 `Process id=4 -> Finish`
   之后)。
2. **gpu_compute 兜底同步 done**: CPU 兜底路径 (kernels.h) 同步调 done() ——
   done 回调可能在**提交线程**上同步触发 Finish (delete es), 外层 Process 栈帧
   还持有 es。因此 GPU 分支提交异步后必须立即 return; 安全无损: GPU 节点
   IsCheap=false 从不入内联队列, 只可能是循环首个节点。
3. **诚实代价**: closure 模型在「根节点多的小图」上调度开销更高 —— demo 3 纯 CPU
   推理图 (3 根节点, 1 worker) 每次 run 提交 3 个闭包 vs 旧版 1 个 WorkerLoop,
   并发推理总耗时 120 ms vs 旧版 103 ms (约 +15%, 40000 run × 2 次额外
   锁+cv 原语)。convoy 收益只出现在 GPU 异步等待场景 (demo 8 的 3.2x), 两种
   模型各有适用面 —— 这数字是诚实的, 不为效果好看而只贴加速的一面。

## demo 与结果

```bash
make && ./build/train
```

| demo | 内容 | 结果 |
|------|------|------|
| 1-4 | 继承 v5: CSE / 常量折叠 / prune 串联 | 同 v5 |
| 5 | 执行队列: 8 独立 matmul 分支 + 树状合并, 1 worker vs 硬件并发数 | 输出逐位一致; **加速比 ~1.01x** (诚实: matmul 走 SME2 且 ZA 段 mutex 串行化, 并行被串行抵消) |
| 6 | 设备放置: matmul 显式 GPU (Metal 异步) + 逐元素自动 CPU | 混设备 ~0.89 ms vs 全 CPU ~1.01 ms (小形状 dispatch 开销占比大, 诚实数字) |
| 7 | SME2/AMX 实测 | 7a/7b 逐位一致; 7c matvec ~0.48x (单列 ZA 1/4 tile + 未 unroll), **gemm ~2.64x / ~9 GFLOPS** (4×4 tile 16 MAC/指令) |
| 8 | 并发 GPU 推理 (v8 convoy 验收): 混设备图 × 8 worker × 8 线程并发 | 新版 ~293 ms vs 旧版基线 ~933 ms (3.2x), mean_y 一致 |

demo 7 的诚实数字: matvec 用 ZA 一列 (1/4 tile), asm 首版未 unroll (每 k 8 条
insr 串行), SME2 反而慢; GEMM 4×4 每指令 16 MAC, 才是 AMX 的正确打开方式。

## 与 TF 的对应

| v7/v8 这里 | TF commit 1 |
|---------|-------------|
| `core/executor.h` (GraphPlan/ExecState) | `core/common_runtime/executor.cc` (ExecutorState, 2118 行) |
| closure 模型: 闭包提交 + 内联队列 (无 cv-wait) | SimpleExecutorState (`executor.cc:1567-1982`): 跑完就绪链即归还线程 |
| `ScheduleReady` 批量 + tail recursion | 同款 (`executor.cc:1701-1743`, cheap inline / expensive ready queue) |
| ExecState 由最后一个完成者 delete + done_cb | `Finish()` 的 delete this (`executor.cc:1973-1982`) |
| `core/threadpool.h` | `lib/core/threadpool.{h,cc}` |
| `core/place.h` + `HasMetalKernel` | `common_runtime/simple_placer.cc` + `FilterSupportedDevices` |
| CPU matmul → 手写 SME2 asm | CPU matmul → Eigen |
| GPU matmul → Metal shader + done 回调 | `common_runtime/gpu/gpu_device.cc` (ComputeAsync) |
| `g.on(n, GPU)` + AUTO 自动放置 | NodeDef.device + 未指定自动放置 |
| ZA mutex 串行化 (并发安全) | commit 1 无 ZA 问题, 但有设备间 Send/Recv 同步 |

## 今天价值 (2026)

- 执行队列的 pending 计数 + 就绪判定是**所有并发图引擎的骨架** (TF 至今、
  PyTorch 的 eager 并发、DAG 调度器), 20 行看懂生产引擎的核心机制; closure 模型
  的「无 cv-wait」设计是"空闲不占线程"的教科书做法
- simple_placer 的 "显式优先 + kernel 注册表自动放置" 仍是现代框架放置语义的
  原型 (K8s 调度也是同一思想: 显式 nodeSelector 优先, 否则按资源/亲和性选)
- M4 AMX 通过 SME2 `fmopa` 一条指令打满外积 —— 手写 SIMD 之上, 还有一层
  "一条指令一个 tile" 的硬件矩阵单元; 踩坑 (ZA 越界、SM 切换清寄存器) 是
  Apple SME 实机行为的稀缺一手记录
- GPU 统一内存 + Metal async dispatch 的重叠执行, 是今天 M 系列 Mac 上
  CPU/GPU 协作的典型形态; 并发 GPU 推理的 convoy 问题 (v8) 是真实生产场景
  (多请求推理服务) 会遇到的问题

## main demo 输出 (2026-08-17, M4 实测)

```
== demo 1: CSE 合并重复子表达式 ==
  before 1st run: 4 nodes
  after  1st run (CSE merged sigmoid): 3 nodes
  values == direct 2*sigmoid(x): yes

== demo 2: 常量折叠 ==
  2a before 1st run: 5 nodes
  2a after  1st run (2*3 folded): 3 nodes
  values == x+6: yes
  2b before 1st run: 5 nodes
  2b after  1st run (chain folded + CSE dedup): 3 nodes
  values == x+4: yes

== demo 4: 负例 —— 有状态节点不合并 ==
  graph before 1st run: 4 nodes
  after 1st run (sgd kept as 2 nodes): 4 nodes
  v = 0.6  (expect 0.6: 两个 sgd 各应用一次, 未合并)

== demo 3: prune + CSE + 常量折叠 串联 ==
  full graph (forward + dup monitor + grad + sgd): 25 nodes
  after 1st run (auto optimize: dup monitor merged): 22 nodes
  epoch 0: loss=20.9918
  epoch 40: loss=2.1121
  epoch 80: loss=0.600487
  epoch 120: loss=0.283229
  epoch 160: loss=0.21664
  epoch 199: loss=0.202812
  trained: a=1.97173  b=2.99197
  after prune{y_pred} (inference graph): 5 nodes

  concurrent inference: 8 threads x 5000 runs on the same graph: 120.436 ms total
    thread 0  x=-4  mean_abs_err=0.105062
    thread 1  x=-3  mean_abs_err=0.0767891
    thread 2  x=-2  mean_abs_err=0.0485153
    thread 3  x=-1  mean_abs_err=0.0202416
    thread 4  x=0  mean_abs_err=0.00803208
    thread 5  x=1  mean_abs_err=0.0363059
    thread 6  x=2  mean_abs_err=0.0645795
    thread 7  x=3  mean_abs_err=0.0928535

== demo 5: 执行队列 (op 并发执行) ==
  图: 8 个独立 matmul 分支 + 树状合并 (33 nodes, [8192,512]×[512])
  顺序 (1 worker):  27.5788 ms   y=4.004
  并行 (14 workers): 27.27 ms   y=4.004
  输出逐位一致: yes
  加速比: 1.01133x

== demo 6: 设备放置 (simple_placer) ==
  混设备图 placement (name: device):
    x: CPU
    w: CPU
    matmul: GPU  (显式)
    b: CPU
    add: CPU
    sigmoid: CPU
  混设备结果 == 全 CPU 参考: yes
  平均单次耗时: 混设备 0.894292 ms  vs  全 CPU 1.00845 ms

== demo 7: SME2/AMX CPU matmul ==
  CPU: M4 FEAT_SME2 (ZA 4x4 f32, VL=128bit streaming)
  7a matvec [8192,512]x[512] == scalar: yes (逐位)
  7b gemm  [1024,256]x[256,128] == scalar: yes (逐位)
  7c matvec: scalar 1.59425 ms  vs  SME2 3.31546 ms  (0.480853x, 2.53015 GFLOPS)
  7c gemm : scalar 19.8415 ms  vs  SME2 7.51187 ms  (2.64135x, 8.9337 GFLOPS)

== demo 8: 并发 GPU 推理 (convoy 修复验收) ==
  8 threads x 200 runs: 293.139 ms total (mean_y=0.358773, 旧版固定 worker 池基线 ≈ 933 ms)
```
