# TensorFlow 学习项目

## two_tower_demo/

双塔模型 TF 文件格式详解：checkpoint 内部结构（V2 索引/数据/对象图）、saved_model.pb（MetaGraphDef/SignatureDef/FunctionDef）、TFRecord 格式。附可运行的训练→导出→验证 Demo。

## compress_demo/

用户特征压缩 5 种实现对比：Keras 原生 / pb 改写 / DeepRec 风格 graph 改写 / BFS 自动边界检测 / serving benchmark。

## xla_demo/

XLA JIT 编译 benchmark — eager / tf.function / jit_compile 三种推理模式延迟对比、op 融合原理。
