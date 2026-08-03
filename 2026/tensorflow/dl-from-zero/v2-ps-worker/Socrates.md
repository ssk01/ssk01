# Socrates.md - 问答记录

### Q: 第一个版本的 TF 有 ps 概念吗？
**没有。** 对 initial commit `f41959ccb2d` (2015-11) 检索确认：无 `distributed_runtime/`、无 GrpcServer、无 ClusterSpec/job/task/ps/worker。

但初始提交的**单进程多设备**层已经埋好了分布式的地基：
- `graph_partition.cc`：按设备切图 + 跨设备插 `_Send`/`_Recv`
- `rendezvous` (IntraProcessRendezvous)：进程内张量传输通道
- `device_mgr` / `simple_placer`：设备管理与放置

分布式 TF (master/worker/grpc/ClusterSpec) 是 2016 年才加的。**关键洞察：分布式 = 单进程多设备原语 + 网络层**，rendezvous 从共享内存换成网络，变量放到另一台机器上。 (2026-08-03)

### Q: PS/worker 实现时同步/异步怎么落地？
- 同步/异步是 **PS 侧** 的行为, 不是 worker 侧:
  - **sync**: PS 按全局 step 缓冲各 worker 梯度, 到齐(n_workers)取平均后一次性应用; worker push 后阻塞等 ack → 天然 barrier
  - **async**: 谁先到谁先应用 (直接 `param - lr*grad`)
- 图切边的落点: worker 每步 `pull_all`(Recv) → 前向/反向 → `push_all`(Send); 这就是图切分里变量节点的 Send/Recv
- 结果: async/sync 在 2 worker 下 holdout 都 ≈ sklearn 基线 (0.914~0.917 vs 0.916) (2026-08-03)

<!-- 以下继续记录 -->
