# TensorFlow 从 0 到 1 — 以论文为纲的演化史

> 素材: [TensorFlow: A system for large-scale machine learning](https://arxiv.org/abs/1605.08695)(2016-05-31, 本文) + 代码库 git 历史考证
> 论文下载: `paper/tf_paper_1605.08695.pdf`, 文本: `paper/tf_paper.txt`
> 视角: 不看"我们缺什么", 看 **TF 当年是怎么一步步长出来的**——每个设计决策背后的 meta 问题。

---

## 第 0 章: 0 号源头 — DistBelief 的成功与失败

TF 不是凭空设计的, 它是 **DistBelief 的继任者**。论文开头一句话点破:

> "We have based TensorFlow on years of experience with our first-generation system, DistBelief, **both simplifying and generalizing it**."

**DistBelief 成功在哪**: 参数服务器架构, Google 内部 60+ 团队在用, 大规模训练验证过。

**DistBelief 失败在哪**(论文 §2.2 原话):

> "we found this architecture to be **insufficiently extensible**, because adding a new optimization algorithm, or experimenting with an unconventional model architecture would require our users to **modify the parameter server implementation**, which uses C++ for performance."

拆开看这个教训:

- 状态管理(参数存取、更新)是参数服务器的**内置特权代码**——`+=` 是写操作的全部语义
- 想实现 Momentum(需要累加速度)这种"不是单次写操作"的算法 → 必须改 C++ 参数服务器
- 大多数用户只习惯写 Python/Lua 高层模型代码, 高性能 C++ 实现是**入门障碍**

**meta 问题**: 状态管理是"系统的特权"还是"用户可以改的"?——DistBelief 的答案是前者, 所以死了。

---

## 第 1 章: 核心设计决策 — 统一数据流图

### 灵感来源(论文原话)

> "We draw inspiration from the **high-level programming models of dataflow systems**, and the **low-level efficiency of parameter servers**."

### 核心洞察: 数据流 + 可变状态 = 参数服务器的超集

> "The key observation in the parameter server architecture is that **mutable state is crucial**... Dataflow with mutable state enables TensorFlow to **mimic the functionality of a parameter server, but with additional flexibility**, because it becomes possible to execute arbitrary dataflow subgraphs on the machines that host the shared model parameters."

论文结论直接下断言:

> "TensorFlow's dataflow representation **subsumes** existing work on parameter server systems."

即: 参数服务器是 TF 图的一个**特例**(可变状态 + 特化写操作), 而不是架构本身。状态管理从"系统内置"变成"图的一部分" → 第 0 章的 meta 问题被翻转: 状态不再是特权, 而是数据流的一个节点。

### 与三类前人的对比(为什么都不是答案)

| 系统 | 特性 | 为什么不够 |
|---|---|---|
| 批量数据流 (MapReduce / Spark / DryadLINQ) | 输入不可变 + 计算确定性 → 机器挂了可重执行 | 更新模型是重操作: SparkNet 训练要 20 秒广播权重+收集 5 个 worker 的更新; 只能加大 batch, 收敛变慢 |
| 参数服务器 (DistBelief / Project Adam / Li et al. PS) | 状态内置, 写操作特化为 `+=` | 特权代码, 不可扩展(第 0 章) |
| Naiad (timely dataflow) | 循环数据流图 + 可变状态, 毫秒级协调 | 面向稀疏离散数据, 无 GPU/加速器支持; 但 TF 借了它的迭代思想 |

### 与批量数据流的两点根本区别(论文 §3)

1. **支持重叠子图上的多次并发执行**(一个 step 只跑图的一部分, 多个 step 并行跑)
2. **顶点可以有可变状态, 跨不同次执行共享**

这两点是"图"从批处理系统里"活"过来的关键——后面每一章都在展开它们。

---

## 第 2 章: 系统组件 — 每个组件解决什么 meta 问题

按"从 0 到 1 的搭建顺序", 每个组件 = 一个必须解决的问题:

### 2.1 op + tensor + attribute — 计算的原子单元

- 所有数据是 **dense tensor**(稠密多维数组)——故意选稠密: 让底层内存分配/序列化简单
- op 是 named type(Const, MatMul, Assign...)+ **compile-time attributes**
- attribute 让 op 泛化: `Const` 有 T(类型)/Value(值); `AddN` 是 variadic, N 属性决定输入个数
- **meta 问题**: 如何用少量原语覆盖所有计算? 答案: op 类型 + attribute 的组合空间, 而不是每个形态一个 op

### 2.2 Variable + reference handle — 状态进入图

- `Variable` op 拥有一个**可变缓冲区**, 产出 **reference handle**(类型化能力凭证: 能读写什么类型/形状, 由句柄类型决定)
- 读: `Read(r)` 输出张量的值; 改: `AssignAdd(r, x)` 执行 `State[r] ← State[r] + x`
- **meta 问题**: 状态放在哪? 答案: 状态是图里的 op, 读写状态也是图里的 op——第 1 章"可变状态"的具体落地

### 2.3 部分执行 (step + prune) — 图是"所有可能", step 是"一次具体"

- 客户端指定 **feed**(喂哪些边)+ **fetch**(取哪些边)→ runtime 裁剪出必要的子图
- 每次调用 = 一个 **step**; 同一张图可以**多个 step 并发**, 状态 op 负责协调
- 典型训练应用 = 多个并发子图(读数据 / 预处理 / 训练 / checkpoint)通过变量和队列交互
- **meta 问题**: 训练/推理/数据管线是一张图还是三张图? 答案: 一张图 + 部分执行——图是静态的"所有可能计算", step 是动态的"一次具体执行"

### 2.4 队列 — 状态 + 协调

- `FIFOQueue` 也是 stateful op: 拥有内部张量队列, 支持并发访问
- `Enqueue` 满则阻塞, `Dequeue` 空则阻塞 → **天然背压**(backpressure)
- 队列不只是数据管线: §4.4 里**同步训练就是用队列做 barrier**
- **meta 问题**: 状态之间怎么协调节奏? 答案: 状态不止 Variable 一种——队列是"带阻塞语义的共享状态"

### 2.5 分布式执行 — placement + partition + Send/Recv

- 每个 op 放在某个 device(某 task 的 CPU/GPU)
- **placement**(simple_placer): 隐式/显式设备约束 + colocation 组(状态 op 和它的状态必须同设备); 用户可写"任意 GPU"这类偏好
- **partition**(graph_partition): 把图切成 per-device 子图, 跨设备边用 **Send/Recv** 替换
  - `Send`: 张量一可用就发到指定设备, 用 **rendezvous key** 命名
  - `Recv`: 阻塞直到本地出现该 key 的值
- 同一程序可部署到: GPU 集群训练 / TPU 集群服务 / 手机推理
- **meta 问题**: 通信显式化。数据流让通信变成图结构的一部分 → 切分是编译期转换, 而不是手写
- Send/Recv 按设备对特化: CPU↔GPU 用 `cudaMemcpyAsync`(计算与传输重叠), GPU↔GPU 用 DMA, 跨任务用 gRPC/RDMA

### 2.6 控制流 — 从严格求值到非严格求值

- 默认求值是**严格**的(所有输入先算完 op 才执行); 但 RNN 这种算法需要**动态控制流**, 效率上必须非严格
- 原语: `Switch`(数据 + 控制输入 → 选一条边输出, 未选边收到 **dead value**, 递归传播到 `Merge` 为止)、`Merge`(最多转发一个非 dead 输入)
- 循环 = Switch/Merge + 基于 timely dataflow 的结构约束(Naiad 的遗产): 支持多并发迭代和嵌套循环, 但限制每个 op 每迭代每输出只产一个值, 简化内存管理
- **meta 问题**: 执行次数动态化(第 0 到 4 版 dl-from-zero 的拓扑排序解决不了)——答案是把"流"本身变成图

### 2.7 分层架构(论文 §5 图 5)

```
Python client  C++ client ...
───────────────────────────
C API(薄层, 隔离用户语言与核心库)
───────────────────────────
Distributed master    → prune / partition / CSE / 常量折叠; 子图缓存, 大图 step 只需每个 task 一条小消息
Dataflow executor     → 并行调度 kernel(约 200 万 null ops/秒); 多核 CPU / GPU 多 stream
Kernel implementations → 200+ 标准 op; Eigen::Tensor 模板生成 CPU/GPU 并行代码; cuDNN 特化
RPC / RDMA ...        → 跨任务通信
CPU / GPU ...         → 设备层
```

**meta 问题**: 如何让用户语言和核心库都演进? 答案: C API 是稳定契约; 上面用户语言随便加(Python 优先), 下面 C++ 核心随便改, 互不牵制。

---

## 第 3 章: 可扩展性证明 — 四个"DistBelief 里要改 C++"的东西, TF 里全是用户级代码

论文 §4 的四个 case study, 每个都在回答同一个 meta 问题: **"前一代系统内置的功能, 用图原语能不能重新做出来?"** 四个全是的:

### 4.1 自动微分(对应 dl-from-zero v4)

- 用户级库: BFS 找从目标 op(如 loss)到参数的所有反向路径, 把每条路径的偏导求和
- 用户可**特化单个 op 的梯度**(batch norm、gradient clipping 都是这么做的)
- 扩展到条件/迭代子计算, 长序列迭代时管理 GPU 内存
- Momentum/Adagrad/Adadelta/RMSProp/Adam/L-BFGS 全是用户代码——在 DistBelief 里每个都要改 C++ 参数服务器

### 4.2 超大模型(embedding 分片)

- 大语言模型词汇表 80 万词, embedding 几个 GB 到几 TB——**每步拷给 worker 不现实**
- 图原语组合: `Gather`(稀疏抽行,colocate 到变量所在设备)+ `Part`(动态按分片拆 indices)+ `Stitch`(重组结果)
- 每个原语都有梯度 → 自动微分后得到**稀疏更新**(只更新被抽到的行)
- softmax 分片: 权重矩阵按列分到多个 PS task, 乘法/梯度计算与分片 colocate(Project Adam 的做法); 或 sampled softmax(采样 512 个负类, 传输和计算降 78 倍)

### 4.3 容错(checkpoint)

- `Save`/`Restore` 就是图里的 op: 每个 task 一个 Save(最大化分布式文件系统 I/O 带宽), 周期性 run 一次
- 恢复 = 建图时把 `Restore` + `Assign` 连进变量
- checkpoint 保留策略、迁移学习(微调/预训练)全是用户代码——"Having checkpoint and parameter management as programmable operations in the graph gives users the flexibility to implement schemes like these and others that we have not anticipated"
- 一致性取舍: 训练和 checkpoint 并发跑, 快照可能不一致——对异步 SGD 没影响(弱一致性)

### 4.4 同步副本协调

- 三方案全是**用户级代码**: 异步(读最新值, 应用时值已旧)/ 同步(两个队列: barrier 保证所有 worker 读同一版参数, 另一个队列累积梯度原子应用)/ 同步 + backup workers(Proactive: 取 m 个里的前 m 个更新; 论文实验 50 worker + 4 backup 最短 step 时间, 3 个 backup normalized speedup 最高 9.5%)
- 论文点明: "Though we designed TensorFlow for asynchronous training, we have begun experimenting with synchronous methods"——**设计时是异步, 同步是后来在图上做出来的**。这就是"状态是图的一部分"的红利: 一致性模型也能实验

---

## 第 4 章: 时间线 — 论文描述的架构 vs 代码里的实况

| 时间 | 事件 | 证据 |
|---|---|---|
| 2012 | DistBelief 论文 (NIPS) | 第一代系统, 参数服务器 |
| 2015-11-06 | **开源, 初始 commit** `f41959ccb2d` | 单机多设备已经全在: executor frame 机(2118 行)、控制流原语、FIFOQueue、graph_partition + Send/Recv、simple_placer、323 kernels、Python 层(gradients/optimizer/saver/queue_runner) |
| 2016-02-25 | **分布式 runtime 加入** | `00986d48bb6` "Initial version of the open-source distributed TensorFlow runtime." |
| 2016-03-14 | 论文 v1 `1603.04467` | 分布式版论文 |
| 2016-05-31 | 论文 v2 `1605.08695`(本文) | + TPU、RDMA、backup workers |

**关键结论**: 论文(2016-05)描述的架构比初始 commit 正好领先一个"分布式"。
初始 commit 里 §3.3 的全部单机机制(placement/partition/Send-Recv/executor)已经在了, 但 Distributed master 和 gRPC 是 2016-02-25 才进代码——论文 v1 就是那个 commit 的说明文档。
所以 TF 的"从 0 到 1"拆开看是: **先单机多设备把图执行模型做对, 再往上套分布式**——Send/Recv 这个单机就有的原语, 就是为分布式预留的接口。

---

## 第 5 章: 三个 meta 洞察(整篇论文的骨架)

1. **可扩展性 = 把"特权"变成"图结构"**。DistBelief 把状态管理做成内置代码, TF 把它变成图中的 Variable/队列/op——于是优化器、一致性模型、checkpoint 策略全变成用户级实验对象。论文核心一句话: "subsumes existing work on parameter server systems"。
2. **训练/推理/分布式是同一张图的不同执行方式**。图是"所有可能计算", step 是"一次具体执行", 部署只是 placement 的事。所以 TF 能从数据中心一路铺到手机。
3. **用户级代码优先, 系统只提供原语**。200+ 个 op + C API, 其余全是用户代码; 甚至自动微分、checkpoint、同步训练这些"框架功能"都是库。系统的职责是原语够小、执行够快(200 万 null ops/s)。

---

## 附录: 与 dl-from-zero 的对照(附注, 不是主线)

| dl-from-zero | TF 论文/代码的对应 |
|---|---|
| v0 图+Session+手写 backward | 图执行模型 + §4.1 自动微分 |
| v1 图裁剪 | §3.2 部分执行的 prune |
| v2 PS/Worker | 2016-02-25 的分布式 runtime(**不是**初始 commit) |
| v3 图/运行时状态分离 | §3.1 Variable + executor Entry |
| v4 梯度子图 | §4.1 用户级自动微分 |
| 未做: 控制流 / Send-Recv 分区 / 队列 pipeline / Function | §3.4 / §3.3 / §3.1 / function.cc |

对照的启示: dl-from-zero 的顺序是"我们自己排的学习顺序", TF 的演化顺序是"单机执行模型 → 分布式"——两条线在 Send/Recv 和分布式处交汇。
