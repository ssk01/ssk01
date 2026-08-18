# Socrates

### Q: TF 初始 commit 中哪些 feature 值得学习和实现？
基于对 TensorFlow 初始 commit 代码的分析，对比 dl-from-zero v0-v7 已实现的部分，推荐优先实现的 feature：

**已实现**:
- v0-v7: 计算图、图裁剪、分布式(PS)、状态分离、梯度子图、CSE/常量折叠、量化、执行队列+设备放置

**值得实现但还没做的**（按优先级）:

1. **图分区 (Graph Partition) + Send/Recv** ⭐⭐⭐⭐⭐
   - `graph_partition.cc` (1050 行): 跨设备边插入 Send/Recv 节点
   - 连接 v7 (设备放置) 和 v2 (分布式): 单进程多 GPU → 多进程分布式是同一套机制
   - Send/Recv 是进程内和跨进程的统一抽象

2. **Saver / Checkpoint** ⭐⭐⭐⭐
   - Save/Restore op 存变量到磁盘，filename 本身是图中的 tensor
   - v3 引入了变量但只活在内存里，这是训练系统的最后一块拼图

3. **队列 (Queue) + QueueRunner** ⭐⭐⭐
   - FIFOQueue: 生产/消费解耦，解决 GPU 等数据问题
   - 打破"输入是静态值"的假设

4. **Function / FunctionLibrary** ⭐⭐⭐
   - 打破"图是扁平的"假设，子图可复用
   - v4 的梯度子图和 Function 是一体两面

5. **BFC Allocator (GPU 内存管理)** ⭐⭐
   - Best-Fit with Coalescing，减少 GPU 显存碎片

**不推荐**: 控制流 (Switch/Merge/Enter/Exit) —— 用户明确表示不喜欢，没必要

**推荐路线**: v8 做图分区，因为它和 v7 衔接最自然，且补完 v2 的考证。

(2026-08-18 15:35)

### Q: v8-graph-partition 实现完成
v8-graph-partition 已成功实现并验证：

**核心机制**:
1. **图分区算法** (`graph/partition.h`): 遍历所有边，跨设备边插入 Send/Recv
2. **Rendezvous** (`core/rendezvous.h`): mutex + map + cv，Send/Recv 通过 key 配对
3. **Send/Recv kernel** (`kernels/kernels.h`): Send 写 rendezvous，Recv 阻塞读

**测试结果** (test_partition.cpp):
- 原始图 5 节点 → 分区后 9 节点
- 跨设备边正确识别: `x(CPU)->matmul(GPU)` 和 `matmul(GPU)->add(CPU)`
- 自动插入 4 个 Send/Recv 节点，rendezvous_key 格式正确

**与 v2/v7 的关系**:
- v7 的 simple_placer 只分配设备，v8 补上跨设备数据传输
- v2 的 PS transport 本质是 Send/Recv 的跨进程实现
- 统一抽象: 进程内 rendezvous = 共享内存 map，跨进程 = TCP/RDMA

**文件清单**:
- `graph/partition.h`: 图分区主逻辑
- `core/rendezvous.h`: 配对机制
- `graph/graph.h`: 新增 SEND/RECV 节点类型 + send/recv 方法
- `kernels/kernels.h`: Send/Recv 的 forward 实现
- `runtime.h`: RunState 加 rendezvous 指针
- `session.h`: partition_once 幂等调用
- `README.md`: 完整文档

(2026-08-18 15:52)

<!-- 以下继续记录 -->
