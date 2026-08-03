# playgroud for Kaizen

Learning by doing. Each sub-directory is an independent experiment.

> 建这个 repo 的初衷是自己动手写每一行代码来加深理解，但说实话，里面相当一部分项目已经是 vibe coding 了——和 AI 结对完成的。时代变了，Kaizen 的方式也在变。

<details open>
<summary><b>2026</b></summary>

| Project | Language | Description |
|---------|----------|-------------|
| [tensorflow/dl-from-zero](2026/tensorflow/dl-from-zero/) | C++ / Python | 从 0 到 1 复刻 TensorFlow 概念 — 5 个演进版本（v0-linear 手写 autodiff → v1-prune 图裁剪+binding → v2-ps-worker 分布式 → v3-concurrent 图/状态分离 → v4-grad 独立梯度子图），总文档见 umbrella README |
| [rtti](2026/rtti/) | C++ / Java | RTTI 运行时类型信息详解 — Preorder 区间编号（最小每对象 4B + O(1) instanceof）、JVM 压缩 klass ptr 模拟、C++ vptr/vtable/dynamic_cast 模拟、Euler Tour 三题 |
| [tensorflow/two_tower_demo](2026/tensorflow/two_tower_demo/) | Python | TF 文件格式详解 — checkpoint V2 索引/数据/对象图结构、saved_model.pb（MetaGraphDef/SignatureDef/FunctionDef）、TFRecord 格式 |
| [tensorflow/compress_demo](2026/tensorflow/compress_demo/) | Python | 双塔 User 特征压缩 — 5 种实现对比（Keras 原生/pb 改写/DeepRec 风格 graph 改写）、BFS 自动边界检测、serving benchmark + tf.profiler |
| [tensorflow/xla_demo](2026/tensorflow/xla_demo/) | Python | XLA JIT 编译优化 — eager vs tf.function vs jit_compile 三种推理模式 benchmark、op 融合原理、batch 规模建议 |
| [cpp](2026/cpp/) | C++ | C++ 手动实现 StringView/PinnableView (RocksDB 风格零拷贝) + Cleanable 清理链源码分析 |
| [gemm](2026/gemm/) | C++ | GEMM 矩阵乘优化实验 — 从 naive 到 M4 Pro 上性能媲美 OpenMP，含 packA strides / tiling / vectorization / prefetch 多版本对比 |
| [duckdb-learning](2026/duckdb-learning/) | C++ / Python | DuckDB 源码学习笔记 — 聚合性能对比 ClickHouse（查即算 vs 预聚合）、C++ 特性（optional_ptr/CRTP）、无锁队列设计、压缩算法（Dict/FSST）详解 |
| [hospital](2026/hospital/) | Python | 医院排队叫号系统设计 — 多优先级队列 + 多科室分流 + 等候时间预估 |
| [linear-scan-regalloc](2026/linear-scan-regalloc/) | Python (Jupyter) | 线性扫描寄存器分配算法学习笔记 — 从 Tiger 编译器出发，实现 live intervals 计算 + 线性扫描 + farthest-end spill heuristic + 指令改写，含 5 组测试 |
| [sparse-embedding-demo](https://github.com/ssk01/sparse_embedding_demo) | Python | 推荐系统 Sparse Embedding × Parameter Server 教学 Demo — MovieLens 1M + 多进程 PS 架构，讲清 embedding 大表分片/worker pull-push |
| [CacheLib](https://github.com/facebook/CacheLib) | Markdown | Meta CacheLib (in-process C++ 缓存引擎) 源码解读 — 存储引擎/Navy I/O/并发模型/设计取舍；笔记：[CacheLib深度解析.md](2026/cachelib/CacheLib深度解析.md) |
| [chaoba](https://github.com/ssk01/chaoba) | Python | 最小中文分词器 demo — 词图 DAG + Viterbi 最短路径，~100 行复现 jieba/HanLP 核心机制 |
| [mini-lucene](https://github.com/ssk01/mini-lucene) | C++ | C++17 重写 Apache Lucene 1.0.1 — 完整检索路径 + 多 agent 协作研究（deepseek 实现 / claude review） |
| [trie](https://github.com/ssk01/trie) | C++ | 26-路 Trie + Aho-Corasick C++17 实现 — 在 `/usr/share/dict/words` 和 Moby Dick 上 5 类场景对照 `unordered_set/map` |
| [FM](https://github.com/ssk01/FM) | Python | MF / FM / 双塔 — 向量化召回/粗排/排序 Demo |
| [kv-cache](2026/kv-cache/) | Python | KV cache 完整 demo — 手写单层 attention 验证机制 + GPT-2 真实延迟对比（5x 加速）+ 多模型显存账本（含 GQA） |
| [autograd](2026/autograd/) | Python | 动态图 vs 静态图 autograd 玩具实现 — 手写计算图、反向传播、if 分支对比，演示 PyTorch vs TF 1.x 的核心差异 |
| [bili-views](2026/bili-views/) | TypeScript | B站实时在线观看人数统计系统 — Express + Redis + WebSocket |
| [cache-friendly-layout](2026/cache-friendly-layout/) | C++ | 验证 [Bun Install 博客](https://bun.com/blog/behind-the-scenes-of-bun-install) 中 cache-friendly 数据布局的优化效果：AoS vs SoA、HashMap vs flat array、JSON 对象树 vs SoA+字符串池、真实依赖树遍历 |
| [go](2026/go/) | Python | AlphaGo Zero 风格的围棋 AI — MCTS + 神经网络自我对弈训练 |
| [microgpt.py](2026/microgpt.py) | Python | 从零实现的微型 GPT — 纯 Python 无依赖，手写 autograd + 字符级语言模型 |

</details>

<details>
<summary><b>2025</b></summary>

| Project | Language | Description |
|---------|----------|-------------|
| [branchless_sort.cpp](2025/branchless_sort.cpp) | C++ | Branchless quicksort — 用 conditional move 消除分支预测失败，对比 std::sort 实现 2.5x 加速 |
| [leet_168](2025/leet_168_excel-sheel-column-title.ipynb) | Python | LeetCode 168: Excel 列号转换 (整数 → "ZY") |
| [leet_365](2025/leet_365_water-and-jug-problem.ipynb) | Python | LeetCode 365: 水壶问题 (BFS 搜索) |
| [sort.ipynb](2025/sort.ipynb) | Python | 分析 branchless sorting 为什么快：分支预测代价量化，验证 Jeff Dean 的性能估算模型 |

</details>

