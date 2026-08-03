# v2-ps-worker — PS/Worker 分布式训练 (dl-from-zero 系列第 2 版)

> dl-from-zero 系列第 2 迭代: 从单进程多设备 → 多进程/多机分布式训练
> 概念优先: 先把"一个分布式训练系统有哪些模块、怎么协作"讲清楚, 再动手

---

## 0. 结论先行: TF 第一个版本有 PS 吗?

**没有。**

对 `tensorflow/tensorflow/core` (initial commit `f41959ccb2d`, 2015-11) 的检索确认:

- ❌ 无 `distributed_runtime/` 目录、无 `GrpcServer`、无 `ClusterSpec` / job / task / ps / worker
- ❌ 无 "parameter server" 相关代码

TF 初始提交是 **单进程、多设备** 的执行模型。但它的核心层已经为分布式铺好了底子:

| 单进程原语 (初始提交已有) | 说明 | 分布式时要复用/升级成什么 |
|---|---|---|
| `graph/graph_partition.cc` | 按设备切图, 跨设备插 `_Send`/`_Recv` | 按 job 切图, 插跨进程 Send/Recv |
| `common_runtime/rendezvous` (IntraProcessRendezvous) | 进程内张量传递通道 | 跨进程 Rendezvous (走网络) |
| `common_runtime/device_mgr` / `device_set` / `simple_placer` | 设备管理 + 放置 | DeviceSet 扩展到远程设备 |
| `graph/algorithm.cc` Prune | 图裁剪 | 同上, 推理/训练图都适用 |

**分布式 TF 是 2016 年加入的**(`tensorflow/core/distributed_runtime/`: master、worker、grpc_session、cluster 等)。
也就是说: PS/worker 不是 TF 一开始就有, 而是"单进程多设备的原语 + 网络层"演化出来的。

---

## 1. 为什么要 PS / Worker?

单进程单机训练遇到三个墙:

1. **参数墙 (容量)**: embedding / 大模型参数放不进一块显存/一台机器 → 参数分散到多台 (PS 分片)
2. **算力墙 (速度)**: 单机 GPU 算不过来 → 多个 worker 并行算梯度
3. **通信墙 (带宽)**: 每轮把全部梯度/参数在 worker 间全交换太贵 → 用集中式 PS 做参数聚合, 只发梯度、收参数

对应三种拓扑:

```
PS/Worker (同步):           PS/Worker (异步):          AllReduce (PS 的替代):
worker1 ---\  /--- ps0     worker1 -梯度-> ps0       worker1 <-> worker2 <-> worker3
worker2 ----/   \--- ps1    worker2 -梯度-> ps1       参数在 worker 间环形交换
worker3     worker2 <-参数-- ps0/ps1                 (不用 PS, 数据并行标配)
```

PS 架构胜在 **embedding 等大稀疏参数天然分片** 与 **异步容错**, 这就是推荐系统 (DeepRec / XDL, 你已有的 `sparse_embedding_demo`) 大量使用 PS 的原因。

---

## 2. 核心概念

| 概念 | 含义 |
|---|---|
| **Cluster / ClusterSpec** | 集群描述: `{job: {task_index: host:port}}`, 例 `{'ps': ['ps0:2222', 'ps1:2222'], 'worker': ['w0:2222', ...]}` |
| **job** | 一组同角色的进程集合 (ps / worker) |
| **task** | job 里的一个具体进程, 用 `job_name/task_index` 标识 |
| **master** | 协调者: 接收 Session 请求, 负责切分/分发图, 编排执行; 通常挂在一个 worker 上 |
| **ps (parameter server)** | 存变量: 接收 worker 推来的梯度 → 更新参数 → 供 worker 拉取 |
| **worker** | 跑训练子图 (前向+反向), 产生梯度, 发给对应 ps |
| **Send/Recv 节点** | 图切分后跨进程/跨设备的边上插入的通信节点 |
| **Rendezvous** | 张量传输通道: 进程内 (共享内存) 或跨进程 (网络) |
| **同步/异步更新** | 每轮是否等所有 worker 梯度齐了才更新参数 |

---

## 3. 需要开发的模块清单

> 与 TF `distributed_runtime/` 对照; 每行给出"概念问题 + 最小实现", 按开发顺序编号。

### 3.1 集群描述与身份 `cluster.h`
- **概念**: 集群里有多少 ps / worker, 每个进程自己是谁 (`job/task_index`)
- **最小实现**: 一个 `ClusterSpec` 结构体 (job → task → addr 列表) + `Task` 结构体, 解析命令行参数
- **参考**: TF `ClusterSpec` / `ServerDef`

### 3.2 参数分片策略 `partition.h` (PS 侧, 可选先做)
- **概念**: 大参数表 (embedding) 按什么规则散到多个 ps 上 (取模 / 一致性哈希)
- **最小实现**: 变量名 → ps_index 的 `hash(name) % n_ps`
- **参考**: TF `ReplicaDeviceAssigner`、DeepRec embedding 分片

### 3.3 通信层 `transport.h` / `rendezvous.h`
- **概念**: 跨进程张量怎么传; 在**单机多进程起步**用 `socket` + `pickle` 即可, 不必上 gRPC
- **最小实现**: `Transport` 接口 {`send(key, tensor)`, `recv(key) -> tensor`}; 进程内用 shared dict, 跨进程用 TCP
- **参考**: TF `RendezvousInterface` / `RpcRecvTensor`

### 3.4 图切分 (训练图按 job 分) `graph_placer.h`
- **概念**: 把一张训练图切成 "worker 子图" + "ps 子图", 在切边上插 `_Send`/`_Recv`
  - 关键切边: **变量节点在 ps 上**, 消费变量的算子 (matmul 等) 在 worker 上 → 变量和它的梯度都跨进程
- **最小实现**: 复用 dl-from-zero 的图结构, 新增 `Send`/`Recv` 两种节点类型 + 一个切分 pass
- **参考**: TF `graph_partition.cc` (初始提交已有, 单进程版直接推广)

### 3.5 Master (协调者) `master.h`
- **概念**: 谁决定整张图怎么分、派给哪些 task、何时结束一轮
- **最小实现**: 简单 `RunStep`: 切图 → 把子图分发给 worker/ps 执行 → 聚合
- **参考**: TF `Master` / `MasterSession`; 初始提交的 `LocalSession`

### 3.6 PS 服务端 (参数服务) `ps_server.h`
- **概念**: 存变量表; 处理两个请求: `Update(梯度) -> 更新参数`, `Pull(name) -> 返回参数`
- **最小实现**: `unordered_map<string, Tensor> params` + 两个 RPC handler + 优化器 (SGD/Adam) 应用梯度
- **参考**: TF `parameter_server` 语义、Distributed TF 的 ps kernel

### 3.7 Worker 服务端 `worker_server.h`
- **概念**: 执行被分配的子图 (前向+反向), 计算梯度后发给 ps; 从 ps 拉参数
- **最小实现**: 复用 `Session` 跑子图 + `Transport` 发梯度/收参数
- **参考**: TF `WorkerService`

### 3.8 训练驱动与更新模式 `trainer.h`
- **概念**: 同步 vs 异步
  - **同步**: worker 梯度齐了 ps 才更新 (收敛稳, 慢 worker 拖后腿)
  - **异步**: 各 worker 独立推送梯度 (快, 但梯度陈旧 stale)
- **最小实现**: trainer 里一个 `sync_mode` 开关; 同步用 barrier 等齐
- **参考**: TF `SyncReplicasOptimizer`、`AsyncReplicasOptimizer`

### 3.9 容错与调度 (进阶, 先不做)
- worker 挂了怎么恢复 / 梯度累计 (backup worker) / 动态弹性

---

## 4. 推荐实现路径 (概念优先, 单机多进程起步)

```
第 0 步   cluster.h + transport.h (socket/pickle)         —— 两个进程能互传张量
第 1 步   ps_server.h 独立进程: 存参数 + 收梯度更新        —— 用 curl 式文本协议也能先跑通
第 2 步   worker 端: 训练图切分成 worker/ps 两份 (graph_placer)
         + 变量用 _Send/_Recv 接到 ps 上                  —— 最核心的一步
第 3 步   master 编排 + 同步/异步训练循环
第 4 步   多 worker + 多 ps 分片 (hash) + 真实大 embedding
第 5 步   (可选) 换成 gRPC / 复用真实网络栈
```

> 关键洞察: **dl-from-zero 已有的 `graph_partition` 切边 + Send/Recv 思想, 与单进程版完全同构**。
> 分布式 = 单进程多设备的 `rendezvous` 从"共享内存"换成"网络", 再把变量放到另一台机器上。
> 所以最难的、也最值得复盘的, 就是第 2 步"图怎么切、切边怎么插 Send/Recv"。

## 5. 配套材料

- 你已有的 `sparse_embedding_demo` (MovieLens 1M + 多进程 PS): 完整走通了 PS 训练, 可作为第 2 步的行为参照
- `DeepRec` / `x-deeplearning (XDL)` 目录: 工业级 PS 实现参考
- TF 官方 `distributed_runtime/` (2016 后版本): master/worker/grpc 的模块划分范本

---

## 6. 实现状态 (2026-08-03)

**已实现**: 单机多进程 PS 训练全链路 (`ps_lib/` + `train_ps.py`)。

| 模块 | 文件 | 状态 |
|---|---|---|
| 3.1 cluster | `ps_lib/cluster.py` | ✅ |
| 3.3 transport (TCP+pickle) | `ps_lib/transport.py` | ✅ |
| 3.4 图切分 (概念展示) | `ps_lib/graph_placer.py` | ✅ |
| 3.5 master | `ps_lib/master.py` | ✅ |
| 3.6 ps_server + 优化器 | `ps_lib/ps_server.py` / `optimizers.py` | ✅ SGD+Adam |
| 3.7 worker | `ps_lib/worker.py` | ✅ |
| 3.8 同步/异步 | ps_server `mode` 参数 | ✅ |
| 3.9 容错 | — | ❌ 后续 |

### 运行

```bash
python3 train_ps.py --n_workers 2 --steps 300 --lr 0.5 --mode async
python3 train_ps.py --n_workers 2 --steps 300 --lr 0.5 --mode sync
python3 train_ps.py --n_workers 4 --steps 300 --lr 0.05 --opt adam --mode async --batch_size 128
```

> worker 每步在其分片上**采样一个 mini-batch**(默认 256, `--batch_size` 可调), 每个 epoch 重新 shuffle —— 真实的 mini-batch SGD, 而不是整片全量梯度。

### 结果 (合成二分类数据, sklearn 基线 holdout acc=0.916)

| 配置 | holdout acc |
|---|---|
| async 2 worker sgd (mini-batch 256) | 0.917 |
| sync 2 worker sgd (mini-batch 256) | 0.916 |

### 关键设计

- **同步/异步在 PS 侧**: sync 模式 PS 按全局 step 缓冲各 worker 梯度, 到齐取平均后一次性应用 (barrier); async 模式谁先到谁先更新
- **图切边的落点**: worker 每步 `pull_all`(Recv 参数) → 本分片前向/反向 → `push_all`(Send 梯度), 这正是图切分里变量节点上的 Send/Recv
- **传输**: TCP + pickle 长度分帧; 换真多机只改地址, 协议不变
- **数据分片**: Master 负责数据加载与分片 (对应 TF input pipeline 概念), 分片通过进程参数下发

### 遗留 TODO

- [x] 通信层用 socket+pickle (已定, 概念优先)
- [x] 同步/异步 (两个都做了)
- [ ] 多 PS 分片 (变量 hash → 多 ps 进程)
- [ ] 换 gRPC / 真实网络栈
- [ ] 容错 (worker 挂了 / backup worker / stale 统计)

