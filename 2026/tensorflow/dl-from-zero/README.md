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
└── v5-opt/          图优化: CSE + 常量折叠(对应 optimizer_cse.cc + constant_folding.cc)
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

---

## 演进脉络(为什么这么排)

```
v0  手写 backward + 值在 Node 里    → 图有状态, 不能并发
v1  图裁剪                        → 训练图/推理图分离
v2  分布式                        → 变量放远一点(PS)
v3  状态从图里挪出去              → 同一张图可并发跑
v4  反向也变成图                  → 梯度子图统一进图, prune 自然剪掉
v5  执行前先优化图                → CSE/常量折叠, 编译期消除浪费 (三件套: DCE/CSE/折叠)
```

最后一版(v4)反过来验证了 TF 最核心的设计直觉:**训练和推理的差别, 从"运行时标记"变成"图结构上的可达性"**——这就是 `gradients.py` 把梯度建成子图的意义。

## 运行

每版独立 `make`(C++)或 `make pybind && python3 demo*.py`(Python), 详见各版本 README。
