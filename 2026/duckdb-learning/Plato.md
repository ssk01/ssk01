# Plato — DuckDB 学习规范与原则

### 文档规范
- 所有 DuckDB 学习文档放在 `playgroud/2026/duckdb-learning/` 下，通过软链接指向源文件
- 讨论记录用 Aristotle.md（观点/决策）、Socrates.md（Q&A 知识）、Plato.md（规范/原则）三文件体系
- 代码分析文档需附真实文件路径和行号

### 编码理解规范
- DuckDB 禁止裸指针，用 `optional_ptr<T>`（可空）/ `reference<T>`（非空）/ `unique_ptr`/`shared_ptr` 覆盖所有场景
- `reference_wrapper` 是可以放进容器的引用替代品，在 DuckDB 中用于需要容器存引用且禁裸指针的场景
- DuckDB 没有虚函数（除少数接口类），类继承仅用 CRTP（`SegmentBase<T>`）
- C++ 特性分析要区分「设计意图」和「实际有用」，如 CRTP 的 T 参数当前未使用属于预留设计

### 存储层理解
- 只有一种表引擎（DataTable），无多引擎系统
- INSERT 分非并行（LocalStorage 内存缓冲）和并行（OptimisticDataWriter 各线程独立写盘）两路
- 压缩在 Checkpoint 时做，不在 INSERT 时做
- Checkpoint = 压缩 + 写 DataBlock + 更新 Metadata + 截断 WAL

### 架构对比
- DuckDB vs ClickHouse: DuckDB 嵌入式（无 TCP 开销），自建 BufferPool（非依赖 OS Page Cache）
- DuckDB 无预聚合/物化视图——计算快到不需要

<!-- 以下继续记录 -->
