# 下一步规划 (v9 完成后)

基于 TF 初始 commit 分析，推荐的实现顺序：

## 已完成 ✓
- v0-v7: 计算图、图裁剪、分布式(PS)、状态分离、梯度子图、CSE/常量折叠、量化、执行队列+设备放置+并发执行
- **v8: 图分区 + Send/Recv + rendezvous**
- **v9: Op 注册系统 + 队列数据管线 + 稀疏张量 + 查找表** ← 刚完成 (推荐系统 CTR demo 验证通过)

## 下一步选项

### 选项 1: Saver / Checkpoint ⭐⭐⭐⭐ (推荐)
**为什么**: 
- v3 引入了变量，但只活在内存里 → 训练系统的最后一块拼图
- 实现简单（~200-300 行），独立功能不改核心
- Save/Restore 本身是图中的 op（设计优雅）

**对应 TF**: `tensorflow/python/training/saver.py` + `saver.proto`

**实现难度**: 简单

---

### 选项 2: Function / FunctionLibrary ⭐⭐⭐
**为什么**:
- 打破"图是扁平的"假设 → 子图可复用
- v4 的梯度子图和 Function 是一体两面

**对应 TF**: `tensorflow/core/framework/function.cc`

**实现难度**: 中等偏高（子图实例化机制）

---

## 我的推荐

**v10 做 Saver/Checkpoint**，理由：
1. 简单且独立，快速完成
2. 补完"训练系统"闭环（v3 变量 + v4 梯度 + v10 持久化）
3. 之后可以做 Function（子图复用）

要开始做 v10-saver 吗？
