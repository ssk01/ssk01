# dl-from-zero — 从 0 到 1 复刻 TensorFlow

> 目标不是实现一个"能用"的深度学习框架, 而是**复盘 TF 的架构概念**:
> 每一版解决一个 TF 里的核心设计问题, 用最小代码讲清概念, 抛弃 323 个 kernel 那样的细节。

## 目录

```
dl-from-zero/
├── v0-linear/       最小可跑框架: Graph / Node / 手写 autodiff / Session
├── v1-prune/        图裁剪(训练图→推理图) + Python binding + 逻辑回归
├── v2-ps-worker/    PS/Worker 分布式训练(单机多进程)
├── v3-concurrent/   TF-like 图/运行时状态分离, 同一张图可并发推理
├── v4-grad/         独立梯度子图(对应 TF gradients.py)
├── v5-opt/          图优化: CSE + 常量折叠(对应 optimizer_cse.cc + constant_folding.cc)
├── v6-quant/        量化/低精度推理: int8 (对应首个量化 commit ca4e053aa52 + 论文 §5)
└── v7-parallel/     执行队列(op 并发执行) + 设备放置(simple_placer) + SME2/AMX + Metal (对应 commit 1 的 executor/threadpool/simple_placer/GPU)
```

每版 = 上一版 + 一个新概念。建议按顺序读。

---

## v0-linear — 最小可跑框架

- **概念**: 计算图怎么表示(Graph + Node + inputs)、op 怎么执行(前向/反向 switch)、Session 怎么跑(拓扑排序)、反向怎么算(每个 op 手写链式法则, 逆拓扑遍历)
- **能跑**: `y = a*x + b` 线性回归, 200 轮收敛到 a≈1.97, b≈2.99
- **局限(留给后续版本)**: 值存在 Node 里(图有状态)、backward 是手写单独一趟、没有 Python 层
- 文件: `graph/graph.h` `kernels/kernels.h` `session.h` `framework/tensor.h` `main.cpp`

## v1-prune — 图裁剪

- **概念**: 训练图 → 推理图。从 fetch/target 反向 BFS, 只保留可达节点(复刻 TF `PruneForReverseReachability`), loss/梯度子图整段被剪
- **加分项**: pybind11 Python binding(对应 TF 三层 API 的 Python 层)、逻辑回归跑通信用卡欺诈(AUC 0.936)
- 文件: `graph/prune.h` `bindings/lfpy.cpp` `credit_card_fraud.py`

## v2-ps-worker — PS/Worker 分布式

- **概念**: 参数放不下/算不过来怎么办。ClusterSpec(job/task)、transport(TCP+pickle)、ps_server(存参+收梯度+优化器)、worker(前向+反向)、master 协调;同步/异步两种更新模式;mini-batch SGD
- **考证**: TF 初始提交**没有** PS 概念(无 distributed_runtime/ClusterSpec/grpc), 只有单进程多设备的地基(graph_partition 的 Send/Recv、rendezvous、device_mgr);分布式是 2016 年才加的
- 文件: `ps_lib/{cluster,transport,ps_server,worker,master,optimizers,model,graph_placer}.py` `train_ps.py` `README.md`(模块设计文档)

## v3-concurrent — 图/运行时状态分离

- **概念**: 为什么变量值不能放 Node 里? TF 分三层——图节点纯静态 / 每次 run 的临时值(Executor Entry)/ 变量持久状态(VariableOp 的 Var)。改成 RunState(每 run 局部)+ Session 持有变量 → **同一张图可以并发推理**
- **加分项**: sgd_step 优化器节点变真能用(不再手动更新)、执行计划缓存(图只编译一次)
- 文件: `runtime.h` `session.h` `graph/graph.h` `main.cpp`(8 线程并发 demo)

## v4-grad — 独立梯度子图

- **概念**: 反向传播不是"手写 backward 遍历", 而是**把梯度建成图里的节点**(对应 TF `gradients.py`)。梯度子图与正向同图 → 训练 fetch 优化器节点即触发、推理 prune 自动剪掉、与正向统一走 forward switch
- **加分项**: RunState 改稠密数组按 `node->id` 索引(对应 TF executor 编译期定槽 + `vector<Entry>`), 弃用哈希表; 标量变量梯度用 `reduce_sum` 归约
- 文件: `graph/gradients.h` `graph/graph.h` `runtime.h` `session.h`

## v5-opt — 图优化: CSE + 常量折叠

- **概念**: 图能跑 ≠ 图高效。执行前加编译期 pass(对应论文 §5 master 优化阶段): **CSE** 哈希 (type, 输入, 常量值) 合并重复子表达式(`optimizer_cse.cc`, 初始 commit 实有); **常量折叠** 输入全为常量的子图用同一个 forward 求值后原地转成 CONST(`constant_folding.cc`, 2016-01-25 加入)。与 v1 的 prune 构成图优化三件套 DCE / CSE / 常量折叠
- **加分项**: 过滤规则逐条对应 TF `Equivalent()`(is_stateful / attrs / 输入全等 / 交换律规范化); Session run 前自动跑 pass pipeline(CSE → 折叠 → CSE), Graph 加结构代数 `generation` 让编译缓存自动失效
- 文件: `graph/optimize.h` `graph/graph.h` `session.h` `main.cpp`(4 个 demo: CSE / 折叠 / 负例 / 三 pass 串联)

## v6-quant — 量化/低精度推理

- **概念**: fp32 不是免费的——推理瓶颈在带宽和算力。训练好的模型把权重/激活压到 int8(对应 TF 首个量化 commit `ca4e053aa52`, 2016-04-22 + 论文 v2 §5): `QuantizeV2`(min/max → int8, MIN_COMBINED) → `QuantizedMatMul`(int8×int8, int32 累加, offset 即 zero-point 雏形) → `Dequantize`; 图层面把 matmul 替换成三件套, 权重静态烘焙成 int8 常量(freeze), 激活范围由校准数据统计
- **加分项**: 公式逐条对应 2016 源码(含 epsilon nudge、QuantizationRangeForMultiplication 的精确输出范围); 诚实标注 gemmlowp 快路径当时被 `if(false && ...)` 禁用、实际跑 ReferenceGemm; CSE 哈希加入 qmin/qmax 防误合并; demo 验证 int8 推理 AUC 几乎不掉点(0.8807 → 0.8805)
- 文件: `kernels/quantize.h` `graph/quantize.h` `graph/graph.h` `kernels/kernels.h` `main.cpp`(demo 5: 校准 → 量化 → fp32 vs int8)

---

## v7-parallel — 并发执行 + 多设备

- **概念**: 执行不再单线程拓扑遍历 —— op 就绪即可并发跑(TF `executor.cc` 的 ready queue + threadpool): 图按依赖拆成执行队列, worker 线程池消费就绪 op; 设备放置(simple_placer): 每个 Node 带 device 字符串, 无显式指定时按 kernel 支持度自动放置; CPU 设备补上**手写 SME2 asm GEMM**(M4 AMX): `fmopa` 外积累加进 ZA tile, 对应 commit 1 的 Eigen CPU matmul; GPU 设备用手写 Metal compute shader(`metal_matmul.metal`)
- **加分项**: 手写 SME2 asm 的踩坑记录全在 `sme2_gemm.h` 头注释(st1w 在 M4 上存整个 64B ZA tile → 越界 48B; **M4 每次 smstart/smstop 清零 callee-saved d8-d15** → asm 必须声明 v8-v15 clobber); demo 7 诚实数字: matvec 只用 ZA 单列(1/4 tile)且未 unroll, SME2 反而慢 0.49x; GEMM 4×4 每指令 16 MAC, 2.7x / ~9 GFLOPS; ZA 多线程并发互相覆盖 → mutex 串行化 ZA 段, demo 5 打印诚实加速比(0.997x)
- 文件: `core/executor.h` `core/place.h` `core/threadpool.h` `kernels/sme2_gemm.h` `kernels/metal_matmul.metal` `core/metal_matmul.mm` `main.cpp`(demo 5: 执行队列; demo 6: 设备放置; demo 7: SME2/AMX 实测), 详见本目录 README.md

---

## 演进脉络(为什么这么排)

```
v0  手写 backward + 值在 Node 里    → 图有状态, 不能并发
v1  图裁剪                        → 训练图/推理图分离
v2  分布式                        → 变量放远一点(PS)
v3  状态从图里挪出去              → 同一张图可并发跑
v4  反向也变成图                  → 梯度子图统一进图, prune 自然剪掉
v5  执行前先优化图                → CSE/常量折叠, 编译期消除浪费 (三件套: DCE/CSE/折叠)
v6  推理用低精度                  → int8 量化, 带宽/算力降 4x, 精度几乎无损
v7  并发执行 + 多设备            → executor/threadpool + simple_placer + AMX/Metal
```

v4 反过来验证了 TF 最核心的设计直觉:**训练和推理的差别, 从"运行时标记"变成"图结构上的可达性"**——这就是 `gradients.py` 把梯度建成子图的意义。v5/v6 则覆盖了 TF 推理落地的两条主线: 编译期消除浪费(图优化) + 低精度计算(量化)。

## 运行

每版独立 `make`(C++)或 `make pybind && python3 demo*.py`(Python), 详见各版本 README。
