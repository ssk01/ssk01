# v5-opt — 图优化: CSE + 常量折叠 (dl-from-zero 系列第 5 版)

dl-from-zero 系列第 5 迭代: **执行前的图优化 pass** —— 对应 TF `optimizer_cse.cc`(初始 commit 就有) + `constant_folding.cc`(2016-01-25 加入)。

## 核心概念(meta 问题)

**v0-v4 拿到图就按拓扑执行——图能跑 ≠ 图高效。** 图里充满编译期就能消除的浪费:

1. **同一子表达式被多个消费者各算一遍**(CSE 场景): 多塔模型里共享特征被两个 tower 各自构建一次; loss 函数和 eval 函数各自算了同一个指标
2. **常量表达式每次 run 重算**(常量折叠场景): `2*3`、`mean(square(2))` 这种输入全是常量的子图, 每次 run 都重新执行

TF 的答案(论文 §5 的 master 优化阶段): 执行前加**编译期 pass**。初始 commit 实有 `optimizer_cse.cc`; 常量折叠 2016-01-25 加入(commit 71184628900, 提交说明原文: *"Creates a local executor and executes a copy of the constant 'slice' of the original graph, and replaces nodes in original graph with constant nodes"*)。

**与 v1 的关系**: v1 的 prune = DCE(死代码消除, 对应 TF `PruneForReverseReachability`)。图优化三件套 **DCE / CSE / 常量折叠**, v1 做了第一个, v5 补齐另两个。

## 两个 pass

### CSE(`graph/optimize.h` 的 `cse`)—— 对应 `optimizer_cse.cc`

```
std::unordered_map<size_t, Node*> available
for each node n in topological order:
  h = NodeHash(n)                       // (type, 输入 id, scalar, CONST 值)
  if available[h] exists and Equivalent(available[h], n)
    redirect n 的全部消费边 → available[h]; 删除 n
  else available[h] = n                 // 只保留第一个候选
```

`Equivalent()` 的过滤规则抄自 TF, 逐条对应:

| TF 规则 | 我们的对应 |
|---|---|
| `is_stateful()` 永不合并 | `VARIABLE` / `SGD_STEP` 不参与 |
| `HasRefInput()` 永不合并 | 我们没有 ref 句柄; VARIABLE 自带"读"语义, 其直接消费者等价 TF 里 Read 之后的纯节点(可合并) |
| attrs 相等 | `scalar` + `init` 相等 —— **两个不同的常量不会合并** |
| 输入 (src, src_output) 全等 | 输入节点指针全等 |
| 交换律 op 输入规范化 | `ADD`/`MUL` 按节点 id 排序后比较 → `add(a,b)` ≡ `add(b,a)` |
| `consider_fn` 调用方过滤钩子 | 我们内置规则(demo 不需要) |

demo 级差异: **PLACEHOLDER 不参与合并** —— 我们的 feed 以 `Node*` 为键, 合并两个 placeholder 会破坏 feed 路由; TF 的 feed 以节点名为键, 合并后名字相同, 语义安全, 所以 TF 可以合并。

### 常量折叠(`const_fold`)—— 对应 `constant_folding.cc`

TF 原版: 找出输入全可折叠的节点集 → 拷一份子图 → 用**本地 executor + rendezvous** 执行 → 把原图节点换成同名 Const 节点。

我们的简化(语义等价):
- 不拷子图、不开 executor: 直接在原图上, 输入全为 `CONST` 的纯节点用**运行时同一个 `forward()`** 求值 —— forward 是"这个 op 算什么"的唯一事实来源
- 不删旧建新: **原地把节点改成 CONST**(保持节点身份, 用户持有的指针仍有效)
- 创建序即拓扑序 → 单遍就能折叠整条链(上游先折叠成 CONST, 下游当轮可见)
- 折叠后清掉"没有消费者"的孤儿常量(TF 原版不清理, 靠后续 RemoveDeadNodes pass; 我们顺手做掉, 让节点数可读)

## Session 的变化

- `run()` 前自动跑 pass pipeline: **CSE → 常量折叠 → CSE**(第二轮 CSE 合并折叠产物中彼此重复的常量), 对应 master 在 step 前的优化
- **幂等 + 缓存**: Graph 增加结构代数 `generation`(每次变换 +1), 图没变就跳过优化; 编译缓存键加入 generation, 图被变换后自动失效
- 注意: 优化会删除节点(CSE 合并), 持有被删节点指针再 run 是未定义行为 —— 真实 TF 图构造后不可变, 优化在子图副本上进行

## 运行

```bash
make && ./build/train
```

## 结果

```
== demo 1: CSE 合并重复子表达式 ==
  before 1st run: 4 nodes
  after  1st run (CSE merged sigmoid): 3 nodes
  values == direct 2*sigmoid(x): yes

== demo 2: 常量折叠 ==
  2a before 1st run: 5 nodes
  2a after  1st run (2*3 folded): 3 nodes
   values == x+6: yes
  2b before 1st run: 5 nodes
  2b after  1st run (chain folded + CSE dedup): 3 nodes
   values == x+4: yes

== demo 4: 负例 —— 有状态节点不合并 ==
  graph before 1st run: 4 nodes
  after 1st run (sgd kept as 2 nodes): 4 nodes
  v = 0.6  (expect 0.6: 两个 sgd 各应用一次, 未合并)

== demo 3: prune + CSE + 常量折叠 串联 ==
  full graph (forward + dup monitor + grad + sgd): 25 nodes
  after 1st run (auto optimize: dup monitor merged): 22 nodes
  epoch 0: loss=20.9918
  epoch 40: loss=2.1121
  epoch 80: loss=0.600487
  epoch 120: loss=0.283229
  epoch 160: loss=0.21664
  epoch 199: loss=0.202812
  trained: a=1.97173  b=2.99197
  after prune{y_pred} (inference graph): 5 nodes

  concurrent inference: 8 threads x 5000 runs on the same graph:
    thread 0  x=-4  mean_abs_err=0.105062
    thread 1  x=-3  mean_abs_err=0.0767891
    thread 2  x=-2  mean_abs_err=0.0485153
    thread 3  x=-1  mean_abs_err=0.0202416
    thread 4  x=0  mean_abs_err=0.00803208
    thread 5  x=1  mean_abs_err=0.0363059
    thread 6  x=2  mean_abs_err=0.0645795
    thread 7  x=3  mean_abs_err=0.0928535
```

解读: demo 1/2 验证代数变换前后值一致; demo 4 验证 is_stateful 守卫(合并了会变 0.8); demo 3 串起完整生命周期——25 节点训练图一次 run 自动优化到 22(CSE 合并重复监控链),200 轮训练收敛 a≈1.97 b≈2.99,prune 后推理图只剩 5 节点,8 线程并发推理结果与手算一致。

## 与 TF 的对应

| 我们 | TF |
|---|---|
| `graph/optimize.h` `cse()` | `tensorflow/core/graph/optimizer_cse.cc` `OptimizeCSE`(初始 commit f41959ccb 实有, 但当时**未接入任何执行路径**, 只有单测) |
| 哈希 (type, 输入, scalar, CONST 值) | `NodeHash`(type + 输出类型 + 输入 + attrs) |
| `equivalent()` 过滤规则 | `Equivalent()`: is_stateful / HasRefInput / attrs / 输入 / 控制边 |
| ADD/MUL 交换律规范化 | `FillInputs` 的 commutative sort |
| `const_fold()` 用运行时 forward 求值 + 原地转 CONST | `constant_folding.cc`: 常量子图拷出去用本地 executor 跑, `ReplaceNodeWithConstant` 换节点(2016-01-25 加入, 71184628) |
| Session run 前 pass pipeline | 2016 年底 `common_runtime/graph_optimizer.cc`: `GraphOptimizer::Optimize` 先折叠后 CSE, 由 direct_session 调用; 论文 §5 master 的优化阶段即此 |
| Graph `generation` 缓存失效 | executor 缓存 + 图版本化 |

**历史考证**: 首个 commit(2015-11-07)里唯一的"优化"其实是 `subgraph.cc` 的 `RewriteGraphForExecution`——按 fetch/target 只跑图的一部分(v1-prune 的源头), CSE 写好了但没接线; 常量折叠 2016-01-25 才出现; "Session run 前自动跑 CSE+折叠"要到 2016 年底 GraphOptimizer 才算成型。
