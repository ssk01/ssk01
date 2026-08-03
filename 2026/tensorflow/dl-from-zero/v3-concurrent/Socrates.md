# Socrates.md - 问答记录

### Q: TF 里变量/标量/矩阵的值存在哪? 为什么不能像我们那样放 Node::output?
TF 分三层, 图节点不存值:
1. **图节点** (Graph Node): 纯静态(op 定义/inputs/attrs), arena 分配, 无数据
2. **执行期临时值**: `executor.cc:343` 的 `Entry { Tensor val; bool has_value; }`, 每次 run 分配 `input_tensors` 槽位, run 结束释放
3. **变量持久状态**: `variable_ops.h` 的 `VariableOp` 通过 resource_mgr 持有 `Var*`(独立 Tensor), Assign/ApplyGradientDescent 就地改它

我们之前把"节点持有状态"和"op 输出"混在 `Node::output` 里 → 图有状态、不能并发跑。
v3 改成 RunState(每次 run 的值槽)+ Session::vars_(变量持久表), 图变纯静态。 (2026-08-03)

### Q: Abseil Performance Hints 里 empty_device_info() 是干什么的?
`memory_manager.cc` 的 `LiveTensor` 构造函数里, 当传入的 `dinfo` 为空时, 用**静态分配共享的** `empty_device_info()` 兜底, 而不是每次 new 一个 `shared_ptr<DeviceInfo>`。作用: 把"必然出现、内容永远为空"的默认值做成单例, 省掉大量重复的小分配。是 "Avoid unnecessary allocations" 节的示例之一。注意该代码在开源 TF 初始提交里**不存在**(是 Google 内部抽象, hints 文档自己声明了), 我们框架里对应的是 `kEmptyTensor()`。 (2026-08-03)

### Q: v3 的 topo 排序怎么写得直白?
之前是 反向BFS + children 表 + 入度表(Kahn), 三张结构。v3 改成两阶段:
1. 反向 BFS 收集 targets 的可达集 needed
2. 在 needed 上 DFS, 沿 inputs(上游)递归、**退出时 push** —— 后序天然就是"上游在前"的执行序, 不需要 reverse
踩的坑: 一开始多 reverse 了一次, 把执行序倒过来导致 loss=0。 (2026-08-03)

<!-- 以下继续记录 -->
