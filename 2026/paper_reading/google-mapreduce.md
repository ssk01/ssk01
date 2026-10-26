# 谷歌怎么实现 MapReduce

状态：**TODO — 后面处理**

## 背景

从 COST、PageRank 和 Google 的 Pregel 实现讨论延伸而来，希望后续理解 Google 的 MapReduce 是怎样实际运行的。本次仅记录题目与待办。

## 主要阅读材料

- Jeffrey Dean、Sanjay Ghemawat，*MapReduce: Simplified Data Processing on Large Clusters*，OSDI 2004。
- [论文页面](https://www.usenix.org/conference/osdi-04/mapreduce-simplified-data-processing-large-clusters)
- [Google 提供的 PDF](https://research.google.com/archive/mapreduce-osdi04.pdf)

## TODO

- [ ] 阅读论文，整理一次作业从输入到输出的完整过程。
- [ ] 解释 Master / Worker、任务划分与调度。
- [ ] 解释中间结果、分区、Shuffle、排序及 Reduce 的衔接。
- [ ] 解释故障恢复、慢任务处理和数据局部性。
- [ ] 联系 COST：哪些开销是框架引入的，换来了哪些能力？
- [ ] 阅读后再决定是否需要本地 demo。

尚未展开分析或实现。
