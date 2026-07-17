# TensorFlow 学习项目

## two_tower_demo/

双塔模型生产流程：样本构造 → 训练 → 导出 → 验证，完整走通上线链路。

[详细说明](two_tower_demo/README.md)

## compress_demo/

用户特征压缩 5 种实现对比：Keras 原生 / pb 改写 / DeepRec 风格 graph 改写 / BFS 自动边界检测 / serving benchmark。

[详细说明](compress_demo/README.md)
