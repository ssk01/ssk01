# Plato.md - 项目约定

### 迭代式开发
- 每个新功能从上一版 dl-from-zero 拷贝代码到新目录，在新目录开发，形成版本演进（dl-from-zero → dl-from-zero-1-prune → ...）(2026-08-03)

### 推理优化实现顺序
- 先从概念最简单、价值最高的**图裁剪 (prune)** 做起，后续再考虑 CSE / executor 编译缓存 / data-driven 并行调度 (2026-08-03)

### Python 层是"接真实数据"的入口
- 框架保持 C++ 内核，Python binding (pybind11) 镜像 TF API；有了 Python 层才能用 sklearn 等真实数据集验证经典问题 (2026-08-03)

<!-- 以下继续记录 -->
