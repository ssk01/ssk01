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
demo 1  CSE:       y = sigmoid(x) + sigmoid(x)        4 → 3 节点, 值一致
demo 2  常量折叠:  y = x + 2*3                        5 → 3 (mul 原地折叠, 孤儿 2/3 被清)
                   y = x + mean(square(2))            5 → 3 (整条链折叠 + 折叠产物被第二轮 CSE 去重)
demo 4  负例:      两个相同的 sgd_step 不被合并        v = 0.6 (合并了会变 0.8, 梯度应用两次)
demo 3  三 pass 串联: 训练图(前向 + 重复监控指标 + 梯度子图 + sgd)
                   25 → 22 (CSE 合并重复的 mean(square(diff)) 监控链)
                   → prune 到推理图: 5 节点
                   训练收敛 a=1.97 b=2.99, 8 线程并发推理不变
```

## 与 TF 的对应

| 我们 | TF |
|---|---|
| `graph/optimize.h` `cse()` | `tensorflow/core/graph/optimizer_cse.cc` `OptimizeCSE`(初始 commit 实有) |
| 哈希 (type, 输入, scalar, CONST 值) | `NodeHash`(type + 输出类型 + 输入 + attrs) |
| `equivalent()` 过滤规则 | `Equivalent()`: is_stateful / HasRefInput / attrs / 输入 / 控制边 |
| ADD/MUL 交换律规范化 | `FillInputs` 的 commutative sort |
| `const_fold()` 用运行时 forward 求值 + 原地转 CONST | `constant_folding.cc`: 常量子图拷出去用本地 executor 跑, `ReplaceNodeWithConstant` 换节点 |
| Session run 前 pass pipeline | 论文 §5 master 的 CSE + 常量折叠优化阶段 |
| Graph `generation` 缓存失效 | executor 缓存 + 图版本化 |
