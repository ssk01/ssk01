# Socrates.md - 问答记录

### Q: TF 第一个版本做了哪些推理优化？
TF 初始提交没有显式的 "inference" 模块，推理优化通过 Session.Run 的图重写 + 调度实现：

1. **图裁剪 (Prune)**：`subgraph.cc` 的 `RewriteGraphForExecution` 每次 Run 时执行 —— FeedInputs 把 Placeholder 换成 `_Recv` 注入节点、FetchOutputs 加 `_Send` 拉出结果、`PruneForReverseReachability`（algorithm.cc）从 fetch/target 反向 BFS 只保留可达节点。训练图 → 推理图的关键一步
2. **CSE (optimizer_cse.cc)**：全局值编号，拓扑序扫描按 op+输入+attrs hash，等价则重连输出边删重复节点；stateful/ref-input 节点不合并
3. **Executor 编译缓存 (local_session.cc)**：按 (feeds, fetches, targets) 缓存已编译 executor，同形状推理复用
4. **data-driven 并行调度 (executor.cc)**：pending_count 数依赖、输入就绪即触发；`IsExpensive()` 节点派线程池、便宜节点内联执行（tail-call）
5. **多设备**：simple_placer + cost model 放设备，graph_partition 切图插 Send/Recv，每设备一个 executor 并行

**注意**：初始提交没有常量折叠 / 算子融合（Grappler 是后来的事），推理快主要靠 prune + CSE + 编译缓存 + 并行调度。

→ 落到 dl-from-zero 概念层：prune/CSE 属于**图变换层**，缓存/调度属于**执行引擎层**。已先实现 prune（`dl-from-zero-1-prune`）。 (2026-08-03)

### Q: 有最简单线性模型了，能解哪些经典问题？
线性模型能解的经典问题（及各自需要新增的算子/概念）：

| 问题 | 经典数据集 | 需要新增 |
|------|-----------|---------|
| 单特征线性回归 | 任意回归数据 | 无（已能跑） |
| 多元线性回归 | 房价/销量 | matmul（dot） |
| **逻辑回归（二分类）** | **Wisconsin 乳腺癌、鸢尾花、信用卡欺诈** | **matmul + sigmoid + 二分类交叉熵(log)** |
| Softmax 回归（多分类） | MNIST | matmul + softmax + 交叉熵 |

- 最有教学价值的下一步是**逻辑回归**：线性模型的"分类版"，概念跃迁小但完整覆盖 线性变换→激活→分类损失→SGD
- 需要补 3 个 op：`matmul`（多特征）、`sigmoid`（激活）、`log`（交叉熵），这也正是 TF 初始提交里的核心算子
- Python binding 的意义：能直接接 sklearn 真实数据（如 `load_breast_cancer()`） (2026-08-03)

### Q: pybind11 绑定踩了什么坑？
- 返回 `Node*` 的图方法默认 `return_value_policy::automatic` 会导致 Python 退出时**双重释放**（Graph 和 Node 包装器都释放）→ 进程 SIGABRT
- 修复：`py::return_value_policy::reference`（不转移所有权）+ `py::keep_alive<0,1>`（Node 保活 Graph）
- macOS 编译 Python 扩展需 `-undefined dynamic_lookup`；绑定模块名必须与 `.so` 文件名一致 (2026-08-03)

### Q: 逻辑回归做信用卡欺诈（不平衡二分类）踩了什么坑？
- **数据问题**：sklearn 默认 `class_sep=0.4` + 0.3% 正类时数据几乎不可学（sklearn 自己也只有 AUC 0.52 且全判负）→ 需先验证数据可学性再调框架
- **pos_weight 固定点**：`pos_weight = neg/pos` 时加权 BCE 在 p=0.5 处批均值梯度精确为零 → 模型卡死
  - 修复：**bias 初始化为先验 log-odds**（`ln(pos_frac/(1-pos_frac))`），打破对称
- **评估指标**：欺诈场景看 recall/AUC 而非 accuracy（全判负也有 99.6% 准确率）
- 最终：AUC 0.936 vs sklearn 0.944，证明框架 autodiff 正确 (2026-08-03)

<!-- 以下继续记录 -->
