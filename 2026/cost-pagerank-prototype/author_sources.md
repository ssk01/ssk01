# 作者实现核对与本地实验范围

## 已核对的源码

Rust 仓库：[frankmcsherry/COST](https://github.com/frankmcsherry/COST)。固定提交：`d5bef75f976ee5880811305621a2883d499f3e65`。本地副本在 `references/author-cost/`，随附原 MIT 许可证。

- [pagerank.rs](https://github.com/frankmcsherry/COST/blob/d5bef75f976ee5880811305621a2883d499f3e65/src/bin/pagerank.rs)：同一个 PageRank 内核通过 EdgeMapper 接受不同存储/遍历顺序；零初始化，float32，20 轮，每个节点每轮先计算贡献，沿边累加。当前公开源码用 unchecked 索引。
- [graph_iterator.rs](https://github.com/frankmcsherry/COST/blob/d5bef75f976ee5880811305621a2883d499f3e65/src/graph_iterator.rs)：`NodesEdgesMemMapper` 读取节点/目标列表；`UpperLowerMemMapper` 从块级高位和逐边低位还原端点。两者均使用 mmap 支持的 TypedMemoryMap。
- [hilbert_curve.rs](https://github.com/frankmcsherry/COST/blob/d5bef75f976ee5880811305621a2883d499f3e65/src/hilbert_curve.rs)：`convert_to_hilbert` 编码、分组排序，再输出端点高低位；`BytewiseHilbert` 使用字节查表与旋转状态，`bit_entangle` 是逐位参考实现。
- [to_hilbert.rs](https://github.com/frankmcsherry/COST/blob/d5bef75f976ee5880811305621a2883d499f3e65/src/bin/to_hilbert.rs)：生成 `.upper/.lower` 文件。Hilbert 重排是预处理，有真实成本。
- [label_prop.rs](https://github.com/frankmcsherry/COST/blob/d5bef75f976ee5880811305621a2883d499f3e65/src/bin/label_prop.rs)：原地标签传播，用标签总和不再下降判断停止。
- [union_find.rs](https://github.com/frankmcsherry/COST/blob/d5bef75f976ee5880811305621a2883d499f3e65/src/bin/union_find.rs)：按 rank 合并、不含路径压缩；当前源码还预先取父指针并使用 unchecked 索引。

C# 仓库：[MicrosoftResearch/NaiadSamples](https://github.com/MicrosoftResearch/NaiadSamples)。固定提交：`05b4c4b1cc0321e0a6a38d667d2d70a9fd406a18`。

- [SingleThreaded.cs](https://github.com/MicrosoftResearch/NaiadSamples/blob/05b4c4b1cc0321e0a6a38d667d2d70a9fd406a18/COST/COST/SingleThreaded.cs)：已阅读图扫描、HilbertPagerank、HilbertPagerankFromDisk 等路径；存在 `LargePages` 分配与 `UnbufferedIO` 路径。不同历史入口细节不同，不把它们当作逐行等价版本。
- 本地副本 `references/author-csharp/SingleThreaded.cs` 随附该仓库 MIT 许可证。未在 Mac 上编译/运行 C#、Windows 大页或无缓冲 I/O 部分。

这些是固定的公开源码快照，不保证恰好是 2015 年每一项发表测量所用的精确提交。原论文测量主要来自 C#，正文为清晰起见展示 Rust；作者博客也区分不同机器和版本。

## 实际运行了什么

1. 在 Mac 上 `cargo build --release` 成功构建作者 Rust 工程，原有算法文件未修改。
2. 生成 4096 节点、32768 条边的图，包含两个连通分量且每个节点有出边；用原始 `to_hilbert` 转换布局。
3. 原始 `pagerank`、`label_prop`、`union_find` 分别运行 vertex/hilbert 模式，均正常退出。两种 CC 算法均报告 4094 个非根节点，即两个连通分量。
4. PageRank 原 CLI 只打印迭代计时，不输出最终向量；该运行是执行检查，不单独构成结果一致性证明。
5. 新增一个仅调用原始 `BytewiseHilbert.entangle` 的检查入口，将 260 个坐标的 Hilbert 编码与教学移植逐项核对，完全一致。
6. `author_smoke_results.json` 保存命令、退出码和输出。构建目录和生成的图位于本对话的 work 目录，未把构建产物纳入项目。

## 教学移植与原始复现的区别

`paper_algorithms.py` 是按上述源码/论文结构编写的 Numba 教学移植，不是把原 Rust 性能数字搬进 notebook。

- PageRank：固定同一内核、float32、源/目标 uint32 数组，只改变边顺序；全体节点正出度；分别校验顺序变换结果与独立 SciPy 稀疏公式。保留作者零初值和 0.15 reset 语义，与原先 demo 的概率归一化版本分开。
- Hilbert：采用易读的逐位编码，并与作者字节查表版本交叉验证；没有移植 `.upper/.lower` 紧凑磁盘格式。
- 布局测试：内存数组、合成图、Numba 代码，不测 SSD、不设置 2 MB 大页；编码、排序、重排成本单列，JIT 预热排除。
- 连通性：原地 LP 与无路径压缩的按 rank UF，统一编译方式。使用 int64 节点/父指针，区别于作者 u32 表示；UF 导出规范化完整标签的额外时间计入计时。
- 并行连通性对照：本项目另写同步标签传播，Numba nogil + 线程池；每轮私有输出后归并。它不是原文 Naiad 的并行并查集，也不是原地 LP 的无改动并行化。
- 性能解释：只在任务语义、图、迭代/停止条件一致的组内比较；代码校验结果，不把 cache/TLB 或内存带宽占比当作已经测到的事实。

## 与 COST 初衷的对应

作者用有竞争力的单线程程序衡量系统要投入多少资源才能超过它。这里分别展示：

1. 构造更有效的单线程基线（PageRank 边顺序）；
2. 换一种完成同一任务的算法（连通分量 LP vs UF）；
3. 将并行时间与有效单线程基线比较，报告已测配置内首次越过的位置，而不是只看自身加速比。

本地连通性测试也保留了 LP 比 UF 更快的图，防止把“有竞争力的基线”误解为“固定指定 UF”。

补充阅读：[作者 2015 年博客](https://www.frankmcsherry.org/graph/scalability/cost/2015/01/15/COST.html)。
