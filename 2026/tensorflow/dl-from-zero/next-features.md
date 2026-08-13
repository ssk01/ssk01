# 下一个 feature 选题 — 候选清单

> 叙事方式: 每个 feature 都是"**前一版的某个假设, 在什么场景下不成立**"的故事。
> 先讲卡在哪个 meta 问题, 再讲为什么必须引入新机制, 最后讲怎么实现 + 初始 commit 的考证。

## 一图流: 每个版本/候选打破的假设

| 阶段 | 打破的假设 | 引入的机制 |
|---|---|---|
| v0-linear | 数学表达式是手算的 | 计算图 + Session |
| v1-prune | 训练图能直接用于推理 | 反向可达裁剪 |
| v2-ps-worker | 参数必须和计算在一起 | 分布式参数服务器 |
| v3-concurrent | 图可以持有运行时状态 | 图/运行时状态分离 |
| v4-grad | 反向是"手写遍历" | 梯度建成子图 |
| **候选 1 控制流** | **图是 DAG, 每个节点恰好执行一次** | **frame 执行机** |
| **候选 2 多设备切分** | **一张图只装在一个执行器里** | **simple_placer + Send/Recv** |
| **候选 3 队列 pipeline** | **输入是"值", 一次 run 全同步** | **跨 run 队列 + 阻塞 op** |
| **候选 4 Function 子图** | **图是扁平的, 不可复用** | **_Arg/_Retval 函数化** |
| **候选 5 Saver** | **变量只活在内存里** | **Save/Restore 图内 op** |

---

## 候选 1: 控制流 — 打破"图是 DAG"的假设

### 卡在哪(meta 问题)

v0 到 v4 的 executor 有一个共同的静态假设:**图是 DAG, 拓扑排序后每个节点恰好执行一次**。
这个假设的本质是——**执行次数在编译期就定死了**。

训练里天然需要动态次数:

- **循环**: epoch 数、直到收敛才停 —— 执行次数是运行期才知道的
- **分支**: 学习率衰减到阈值切换策略、loss 突变时的 fallback —— 同一批节点按条件决定跑不跑

### 为什么展开不行(不用新机制会怎样)

没有控制流的"循环"只能**图展开**: `while` 展开成 N 份相同的子图。
展开的根本缺陷:**图的大小 = 执行次数的静态上界**。

- N 是运行期才决定的(收敛才停), 展开时根本不知道 N
- 展开后图爆炸, 编译期就输了
- 分支"两边都建、运行时选一边"也不行: 未选分支的开销和**副作用**(比如变量赋值)无法避免

结论: 执行次数动态化 → **"流"本身必须成为图的一部分**。

### 怎么实现(核心机制)

控制流的原语 op(数据只在被选的边上流动):

| op | 角色 |
|---|---|
| `Switch(data, pred)` | 分支: pred 决定数据走 true 边还是 false 边 |
| `Merge(inputs)` | 汇合: 任一输入到达即产生输出, 附 `value_index` 指示哪条来了 |
| `Enter(data, frame_name)` | 进入循环 frame |
| `Exit(data)` | 离开循环 frame |
| `NextIteration(data)` | 本迭代结束, 数据回到 frame 头, 开启下一迭代 |
| `LoopCond(data)` | 循环条件: 决定下一迭代是否执行 |

**executor 从"拓扑排序"变"frame 机"**——初始 commit 的 `executor.cc`(2118 行)已经有完整实现:

1. **编译期**: 扫每个节点的 `frame_name` 属性, 统计每个 frame 的输入数(`frame_input_count_`)
2. **运行期**: `FrameState` 结构, 每个循环创建一个 frame, "Execution starts at iteration 0"; `outstanding_frame_count` 跟踪活跃迭代
3. **关键**: 节点"就绪"不再由静态入度决定, 而是 frame 内的动态就绪 —— 同一批节点被多次执行, 每次迭代一套新值

### 和 v4 的碰撞(为什么这是最深的一课)

梯度子图(v4)是"梯度沿反向边流回"——图是 DAG 时这很干净。
一旦有控制流:**梯度也要跨 frame 反着流**。初始 commit 里 `control_flow_grad.py` 存在, 它要回答的问题: 循环体的梯度怎么累计回迭代边界? 分支的梯度怎么知道走哪条边?

### 考证(初始 commit 实况)

- ops 全套: `Switch / RefSwitch / RefSelect / Merge / Enter / RefEnter / Exit / NextIteration / LoopCond / ControlTrigger`(control_flow_ops.cc)
- Python 层**只有原语**(`control_flow_ops.py` 的 `switch/merge/_Enter/exit`), **没有高层 `tf.cond` / `tf.while_loop`** —— 高层 API 是后来才加的
- executor 的 frame 支持在初始 commit 就已完整(见上)

---

## 候选 2: 多设备切分 — 打破"一个执行器装下整张图"

### 卡在哪(meta 问题)

v0-v4 的 Session 持有整张图、一个执行器跑完。隐含假设:**图小到单设备装得下**。
这个假设会碎: embedding 表 100GB, 单设备内存放不下; GPU 算得快但显存小。

"放不下"你已经在 v2 尝过滋味——但注意:**v2 用的是 2016 年的 PS 架构, 而初始 commit 的地基是单进程多设备切分**。v2 的考证里点名了这块地基(`graph_partition` 的 Send/Recv、rendezvous、device_mgr), 但没实现。

### 为什么需要新机制

设备是异构的(CPU 内存大算得慢 / GPU 显存小算得快), 切分的选择:

1. 整图放一个设备 → 单设备能力 = 整图瓶颈
2. 手动拆节点 + 手写数据拷贝 → 图是静态的, 拆法应该由**系统编译期**决定

TF 的答案:**设备是节点的属性, 切分是一次图转换**。

### 怎么实现(核心机制)

```
原始图: A --edge--> B          (A 放 CPU, B 放 GPU)
切分后: A -> Send --> | 图边界 | --> Recv -> B
                        rendezvous key 配对
```

1. **`simple_placer`**: 为每个节点选 `assigned_device_name`(策略: colocate 约束、name scope 提示)
2. **`graph_partition`**: 对每个跨设备的 edge, 在源侧插 `Send` 节点、目标侧插 `Recv` 节点, 用 rendezvous key 配对; 跨设备还要插 `Cast`(CPU float32 → GPU float32)
3. 之后每个设备上是**一张独立子图**, 各跑各的 executor, 数据经 Send/Recv 流动

### 核心洞察: Send/Recv 是进程内版"网络消息"

Send/Recv 在进程内是**两个线程间的通道**(rendezvous), 但概念上和网络消息完全同构: 有 key、有 source/destination、有阻塞等待。
**把 rendezvous 换成 TCP socket, 单进程多设备 → 跨进程分布式**——这就是 v2 的 ps-worker 做的事。

这一版做出来, 你的 v2 就闭环了: transport 是 Send/Recv 的一种实现, 不是另起炉灶。

### 考证(初始 commit 实况)

- `graph_partition.cc`: `AddSend` / `AddRecv`, attrs 有 `send_device` / `send_device_incarnation` / `recv_device`; 同设备 GPU 间也有特例(`NeedSameDeviceSendRecv`)
- 配套: `device_mgr`(设备注册表)、`rendezvous`(framework 层)、`simple_placer.cc`

---

## 候选 3: 队列数据 pipeline — 打破"输入是静态值"

### 卡在哪(meta 问题)

现在的数据流是:**batch 张量一次性进图**, 或 Python 侧每轮喂。
隐含假设:**run() 的输入输出都是"值", 数据源是一次性生成的**。

现实: 读盘/预处理慢、计算快。同步喂 → GPU 等数据, 利用率掉一半。流式数据(实时样本)更是根本没法预生成——它本来就是"流", 不是"值"。

### 为什么需要新机制

"值"是一次性的,"流"是持续的 —— 同步接口的表达力不够。
TF 的答案:**队列 = 跨 run() 的持久状态 + 阻塞语义的 op**:

- `FIFOQueue`: `enqueue` 队列满则阻塞, `dequeue` 队列空则阻塞 —— **生产和消费被解耦成两个独立执行流**, 节奏互不拖累
- `QueueRunner`: 后台线程不停 run enqueue 子图(生产者)
- `Coordinator`: 管理多个 QueueRunner 的生命周期(启动 / 停止 / 异常传播)

### 和 v3 的关系

v3 引入了第一个"跨 run 持久状态"(变量 Var)。
队列是**第二个跨 run 状态**——但变量是"所有人共享、无阻塞", 队列是"生产者/消费者协调、有阻塞"。状态 + 协调, 这一版把 v3 的状态概念推到完整形态。

### 考证(初始 commit 实况)

- ops: `FIFOQueue / RandomShuffleQueue / QueueEnqueue / QueueEnqueueMany / QueueDequeue / QueueDequeueMany / QueueClose / QueueSize`(data_flow_ops.cc)
- Python: `queue_runner.py` + `coordinator.py` + `input.py`(数据进队列的标准管线)

---

## 候选 4: Function 子图函数化 — 打破"图是扁平的"

### 卡在哪(meta 问题)

v4 的梯度子图是"原地嵌入": 每个 op 的梯度节点直接连进大图。
隐含假设:**图是扁平的、一个名字空间、一段逻辑只出现一次**。

现实: 同样的 MLP 块复用 N 次 → 图按 N 倍膨胀; 图没有封装边界, 无法"引用"一段已有逻辑。

### 为什么需要新机制

函数 = **带 `_Arg`/`_Retval` 边界的子图**。调用 = 实例化。

```
FunctionDef(名字)  --注册-->  FunctionLibraryDefinition
                                     |
                             NewFunctionLibraryRuntime(device, runner)
                                     |
                     GetFunctionBody -> {arg_nodes, ret_nodes,
                                          arg_types, ret_types} 的 Graph
```

函数体实例化后就是一张普通子图, `_Arg` 节点做数据进入、`_Retval` 做数据返回。

### 和 v4 的呼应(一体两面)

`SymbolicGradient` 是 function.cc 里的一个关键常量:**"函数本身是图, 对函数求梯度 = 对图求梯度"**——直接复用 gradients 机制。
v4 的"梯度子图"和 Function 是同一件事的两个视角: 梯度生成的是子图, 子图通过 Function 机制组织成可复用单元。

### 考证(初始 commit 实况)

- `function.cc` 常量: `kArgOp = "_Arg"`、`kRetOp = "_Retval"`、`kGradientOp = "SymbolicGradient"`、`kNodeLabel = "Func"`
- `FunctionBody` 实例化后还有手写优化("A few hand-crafted optimization on the instantiated function body")
- `function.h`: `NewFunctionLibraryRuntime(Device*, Runner, FunctionLibraryDefinition*)`——**函数实例化绑定 device**(呼应候选 2)

---

## 候选 5: Saver/checkpoint — 打破"变量只活在内存里"

### 卡在哪(meta 问题)

v3 引入持久变量后, 训练是长时间过程 —— **进程一崩, 全部白练**。
隐含假设:**状态只存在于内存, 进程生命周期 = 训练生命周期**。

### 为什么需要新机制(核心设计)

保存/恢复不是"Session 导出数据", 而是**图里的 op**:

- `Save` op: 把变量列表写盘 —— 与训练同图, 跑起来就是一次 run
- `Restore` op: 从盘读回变量 —— 恢复模型 = 建一个 Restore 图跑一遍
- **filename 本身是图中的 tensor**(`filename_tensor_name`)—— 存到哪由**图**决定, 不是 Session 内部状态。这是最体现 TF 哲学的一点: 连"保存"都是数据流的一部分

### 考证(初始 commit 实况)

`saver.proto` 的 `SaverDef`:

| 字段 | 含义 |
|---|---|
| `filename_tensor_name` | 文件名作为一个 tensor 的名字 |
| `save_tensor_name` / `restore_op_name` | 保存/恢复 op 在图中叫什么 |
| `max_to_keep` | 只留最近 N 个 checkpoint(写新的删旧的) |
| `sharded` | 按设备分片存 —— 呼应候选 2 |
| `keep_checkpoint_every_n_hours` | 时间维度额外保留 |

你在 `two_tower_demo` 已写过 checkpoint 详解文档, 概念是熟的——这一版的价值是把它落进 dl-from-zero 的演进线。**难度最低, 适合热身**。

---

## 推荐顺序

| 顺序 | 候选 | 理由 |
|---|---|---|
| 1 | 控制流 | 最深: 直接打破 v0-v4 的拓扑排序假设, 补上执行器最后一课 |
| 2 | 多设备切分 | 呼应 v2: Send/Recv 是进程内版网络消息, 让 v2 闭环 |
| 3 | Function | 呼应 v4: 梯度子图的函数化组织 |
| 4 | 队列 pipeline | 工程实用, 训练系统的实际形态 |
| 5 | Saver | 热身: 概念已熟, 机制独立 |
