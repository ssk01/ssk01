# v8-graph-partition — 图分区 + Send/Recv (dl-from-zero 系列第 8 版)

> 对应 TF commit 1 的 `graph_partition.cc` (1050 行) + rendezvous 机制。
> v7 已经做了 simple_placer 给节点分配设备，v8 补上**跨设备数据传输**：
> 在切边处插入 Send/Recv 节点，通过 rendezvous 配对完成设备间通信。

## 核心概念 (meta 问题)

v7 的 simple_placer 把节点放到不同设备，但**跨设备的边还没处理** —— 如果 matmul 在 GPU、add 在 CPU，matmul 的输出怎么传给 add？

朴素做法：让用户手写拷贝节点。但设备放置是编译期决定的（甚至是自动的），用户根本不知道哪条边跨设备。

TF 的答案：**图分区 (Graph Partition) = 按设备切分图 + 在切边处自动插入 Send/Recv**

## 核心机制

### 1. 图分区算法 (`graph/partition.h`)

```
输入: 完整图 + 每个节点的 device 分配 (v7 SimplePlace 的产物)
输出: 每个设备一张子图

伪代码:
  for each edge (src -> dst):
    if src.device != dst.device:
      在 src 子图插入 Send 节点
      在 dst 子图插入 Recv 节点
      两者通过 rendezvous_key 配对 (key = "src_name:src_output:dst_name")
```

### 2. Send/Recv 节点

**Send**: 
- inputs: [data]
- 属性: dst_device, rendezvous_key
- 执行: 把 data 放进 rendezvous，key 配对

**Recv**:
- inputs: [] (无输入，从 rendezvous 读)
- 属性: src_device, rendezvous_key
- 输出: 从 rendezvous 取出的 data
- 执行: 阻塞等待 Send 写入，然后取出

### 3. Rendezvous 机制 (`core/rendezvous.h`)

```cpp
class Rendezvous {
  // Send 侧：写入 key -> tensor
  void Send(const string& key, const Tensor& tensor);
  
  // Recv 侧：读取 key -> tensor (阻塞等待)
  Tensor Recv(const string& key);
  
private:
  mutex mu_;
  map<string, Tensor> pending_sends_;      // 已 Send 但未 Recv 的
  map<string, cv*> pending_recvs_;         // 已 Recv 但未 Send 的 (wait on cv)
};
```

**关键**: Send 和 Recv 可能以任意顺序到达
- Send 先到: 把 tensor 存进 pending_sends_，Recv 后到时直接取
- Recv 先到: 在 pending_recvs_ 里放一个 cv，阻塞等 Send 唤醒

## 与 v7 的关系

v7 已有的:
- SimplePlace: 每个节点分配到 CPU/GPU
- Executor: 按设备执行节点（但跨设备边还是直接传指针，错误的）

v8 补上的:
- Partition: 把图切成多个子图，每个子图一个设备
- Send/Recv: 跨设备边变成 rendezvous 通信
- 多设备并发执行: 每个子图一个 ExecState，同时跑

## 与 v2 (PS-Worker) 的关系

v2 的 transport (TCP + pickle) 本质是 **Send/Recv 的一种实现**:
- 进程内 Send/Recv: rendezvous = 共享内存 map
- 跨进程 Send/Recv: rendezvous = TCP socket

**这就是 TF 从单机多设备扩展到分布式的统一抽象** —— v8 补上 v2 考证里说的"初始 commit 的地基"。

## 实现文件

- `graph/partition.h`: 图分区算法 + Send/Recv 节点插入
- `core/rendezvous.h`: 进程内 rendezvous (mutex + map + cv)
- `graph/graph.h`: 新增 SEND / RECV 节点类型
- `kernels/kernels.h`: Send/Recv kernel 实现
- `core/executor.h`: 多设备并发执行 (每设备一个 ExecState)

## demo 验证

**demo 9**: 跨设备图
```
x (CPU) -> matmul (GPU) -> add (CPU) -> sigmoid (CPU)
                ↑
           w (GPU, 显式)
```

预期分区结果:
```
CPU 子图:  x -> Send("x:0->matmul")
GPU 子图:  Recv("x:0->matmul") -> matmul <- w
           matmul -> Send("matmul:0->add")
CPU 子图:  Recv("matmul:0->add") -> add -> sigmoid
```

验证:
- 输出 == 单设备参考 (全 CPU)
- rendezvous 日志显示 Send/Recv 配对成功

---

## 与 TF 的对应

| v8 这里 | TF commit 1 |
|---------|-------------|
| `graph/partition.h` | `core/graph/graph_partition.cc` (1050 行) |
| Partition 算法 | `AddSend` / `AddRecv` + `DupRecvTable` |
| rendezvous_key = "src:out:dst" | `tensor_name` attr (格式略有不同) |
| `core/rendezvous.h` | `core/framework/rendezvous.h` + `IntraProcessRendezvous` |
| Send/Recv 阻塞语义 | 同款 (Send 非阻塞, Recv 阻塞) |

## 今天价值 (2026)

- **Send/Recv 是分布式训练的统一抽象**: 无论进程内多卡还是跨机分布式，都是同一套 API —— 只是 rendezvous 实现不同（共享内存 vs TCP vs RDMA）
- **图分区 = 编译期自动化**: 用户写单设备逻辑，框架自动切图 + 插通信节点 —— 这是 TF/PyTorch DDP 的共同设计模式
- **阻塞语义 + 异步执行**: Recv 阻塞不会卡死整个 executor（executor 是多线程 ready queue），其他无依赖的 op 继续跑 —— 这是 v7 executor 并发执行的价值体现

## 构建与运行

```bash
make && ./build/train
```

预期输出: demo 1-8 同 v7，demo 9 显示跨设备图分区 + rendezvous 配对日志
