# Socrates.md - 问答记录

### Q: 对照 TF 初始提交，给 linear_framework 加新功能有什么推荐？
对照 TF 初始提交分析（323+ kernels、自动微分、6 优化器、Saver/checkpoint、数据 I/O、图分区、BFC、多设备），结合框架现状（已有 Graph/Node、7 op、前向+反向、TopoSort/Session、SGD）给出的路线：

- **功能向**：MatMul+二维 Tensor 打通 MLP/MNIST；注册表取代 switch-case（复刻 REGISTER_OP）；Adam/Momentum/RMSProp 对比收敛；Saver/checkpoint 序列化；数值梯度检查；图优化 pass
- **后续修正（用户方向）**：这些偏"细节"。用户真正要的是**概念层次**的从 0 到 1 —— 按 TF 的架构骨架逐层复刻：计算图表示 → 执行引擎（拓扑排序/调度）→ 自动微分 → 优化器/训练循环 → 序列化/checkpoint → 多设备/图分区。每一层用最小实现讲清概念，算子能跑通即可，不追求 323+ 的规模

因此推荐路径是**广度优先**：先让每一层概念都有最小可运行骨架，而不是在某一个 op 或优化器上深挖 (2026-08-03)

<!-- 以下继续记录 -->
