# Paper Reading

统一记录这次讨论涉及的论文与文章，后续逐篇跟进。创建于 2026-10-26。

状态含义：**已讨论 / 有实验**表示已有阅读或实验材料，仍可继续追问；**待读**表示已收录，尚未系统阅读；**TODO**表示只建立话题占位。

| 论文 / 文章 | 年份 / 类型 | 阅读关注点 | 当前状态 | 本地笔记 / 后续入口 |
|---|---|---|---|---|
| [Scalability! But at what COST?](https://www.usenix.org/system/files/conference/hotos15/hotos15-paper-mcsherry.pdf) | 2015 · HotOS 论文 | 有效单线程基线、并行开销、PageRank 布局与连通分量算法 | 已讨论 / 有实验 | [项目](../cost-pagerank-prototype/README.md)、[中文译文](../cost-pagerank-prototype/cost-paper-zh.md)、[作者实现核对](../cost-pagerank-prototype/author_sources.md) |
| [DuckDB: an Embeddable Analytical Database](https://ir.cwi.nl/pub/28800/28800.pdf) | 2019 · SIGMOD 演示论文 | 嵌入式分析数据库的设计与执行方式 | 已讨论，待整理笔记 | TODO |
| [DuckDB and the changing physics of analytics](https://www.allthingsdistributed.com/2026/08/duckdb-and-the-changing-physics-of-analytics.html) | 2026 · 博客文章 | 单机分析能力与分布式架构取舍 | 待读 | TODO；作为文章收录 |
| [The Anatomy of a Large-Scale Hypertextual Web Search Engine](https://research.google.com/pubs/archive/334.pdf) | 1998 · WWW 论文 | 早期 Google 架构、PageRank 与搜索的关系 | 待读；已介绍入口 | TODO |
| [Pregel: a system for large-scale graph processing](https://research.google/pubs/pregel-a-system-for-large-scale-graph-processing/) | 2010 · SIGMOD 论文 | Google 图计算系统、PageRank 示例、通信与容错 | 待读；已介绍入口 | TODO，重点看第 5.1 节及图 4 |
| [Scaling PageRank to 100 Billion Pages](https://storage.googleapis.com/gweb-research2023-media/pubtools/5465.pdf) | 2020 · WWW 论文 | 固定通信模式、隐式目标编号、减少重复通信 | 待读；已讨论核心想法 | TODO；工作完成于 Yahoo Research，作者后来加入 Google |
| [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) | 2016 · arXiv 初稿，后续修订 | HNSW 搜索路径与邻居向量访问局部性 | 待读 / 有探索想法 | TODO |
| [MapReduce: Simplified Data Processing on Large Clusters](https://www.usenix.org/conference/osdi-04/mapreduce-simplified-data-processing-large-clusters) | 2004 · OSDI 论文 | Google 如何实现 MapReduce | TODO | [谷歌怎么实现 MapReduce](google-mapreduce.md) |

## 后续记录方式

在本目录为各话题新增 Markdown 笔记，并更新表格中的状态和入口。已存在的实验项目继续通过相对链接关联；本目录作为共享阅读索引和后续笔记目录。

当前只建立阅读列表和 MapReduce TODO，不启动新的阅读或实现任务。
