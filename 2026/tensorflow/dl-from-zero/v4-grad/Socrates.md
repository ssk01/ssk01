# Socrates.md - 问答记录

### Q: TF 的运行时张量槽位怎么存, 为什么不用 unordered_map<const Node*, Tensor>?
TF 不用哈希表。executor **编译期(构造时)给每个节点分配稠密槽位**(`item.input_start`/`output_slots`), 运行时是 `input_tensors[item.input_start + i]` 这种纯数组下标 —— `executor.cc:666` 一个 `std::vector<Entry>`(Entry 含 `Tensor val` + `bool has_value`), 按节点 id/槽位直接索引, 无哈希、无指针追寻、cache 友好。

我们的改进: Node 创建时拿单调递增 `id`, RunState 改成 `std::vector<Tensor> outputs` 按 id 索引 + `has` 标记。 (2026-08-03)

### Q: 实现独立梯度子图踩了什么坑?
- 标量变量被广播到整批, 梯度子图产出 `[N]` 向量梯度, 而标量变量需要**求和成标量** —— 否则只应用了单个样本的梯度, 收敛方向错、loss 上升
  - 旧实现 `tensor_add_to` 对 scalar 目标是隐式求和; 子图改为显式 `REDUCE_SUM` 节点(对应 TF 梯度 shape-match 的 reduce)
- `MEAN`/`MATMUL` 梯度依赖运行时 shape, 无法用原始 op 组合, 用**专用梯度 kernel**(`mean_grad`/`matmul_grad_a/b`, 对应 TF 的 MeanGrad/MatMulGrad)
- pybind `build_gradients` 返回 dict, `keep_alive` 不适用, 去掉即可 (2026-08-03)

<!-- 以下继续记录 -->
