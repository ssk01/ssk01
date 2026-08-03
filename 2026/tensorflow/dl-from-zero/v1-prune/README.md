# v1-prune — 图裁剪 (dl-from-zero 系列第 1 版)

从 `dl-from-zero` 演进而来的第 1 版:新增**图裁剪 (Graph Pruning)** —— 训练图 → 推理图。

## 概念

TensorFlow 的 Session 在每次 `Run` 时会对图做重写 (`RewriteGraphForExecution`),其中核心一步是
`PruneForReverseReachability` (`tensorflow/core/graph/algorithm.cc`):

> 训练时的图包含 loss、梯度、优化器节点;推理时这些全部无用。
> 从 fetch/target 节点**反向 BFS**,只保留能算出它们的节点,其余一刀剪掉。

本实现对应关系:

| TF 原版 | dl-from-zero-1-prune |
|---------|----------------------|
| `PruneForReverseReachability` (algorithm.cc) | `reachable_from()` + `Graph::keep_only()` (graph/prune.h) |
| 反向 BFS 收集可达集 | 反向 BFS 收集 keep 集 |
| `Graph::RemoveNode` 删非可达节点 | `keep_only()` 保序筛选 |

## 运行

```bash
make && ./build/train       # C++ 版
make pybind && python3 demo_linear.py   # Python binding 版
python3 credit_card_fraud.py            # 逻辑回归: 信用卡欺诈检测
```

输出示例:

```
training graph: 9 nodes        # x y a b mul add sub square mean
before: 9 nodes
after : 5 nodes                # x a b mul add —— loss 子图(sub/square/mean/y)被剪掉
keep: x
keep: a
keep: b
keep: mul
keep: add

Inference on pruned graph      # 推理只需喂 x, y 已被剪掉
x=-3  y_pred=-2.92418  y_true=-3
...
```

## 关键点

- **裁剪是"算得出来"的最小闭包**:保留集合 = targets 沿 `inputs` 反向可达的全部节点
- **保序筛选**:原图创建顺序保留,拓扑排序仍成立
- **推理图不反传**:裁剪后 `loss` 指针已失效,对应 TF 中推理图没有梯度子图
- 下一迭代候选:CSE(公共子表达式消除)、executor 编译缓存、data-driven 并行调度

## Python Binding (pybind11)

对应 TF 三层 API 的 Python 层 (`bindings/lfpy.cpp`),API 镜像 TF 风格:

```python
import lfpy
g = lfpy.Graph()
x = g.placeholder("x", [N]);  y = g.placeholder("y", [N])
a = g.variable("a", 0.0);     b = g.variable("b", 0.0)
y_pred = g.add(g.mul(x, a), b)
loss = g.mean(g.square(g.sub(y_pred, y)))

sess = lfpy.Session()
sess.run(g, [loss], {x: x_data, y: y_data}, loss_node=loss)
a.assign(a.output.data[0] - lr * a.grad.data[0])   # 对应 Variable.assign

lfpy.prune(g, [y_pred])       # 图裁剪
sess.run(g, [y_pred], {x: [0.5]})
```

绑定要点:
- **Graph 方法返回 `Node*`**:`return_value_policy::reference`(不转移所有权,避免退出时双重释放)+ `keep_alive<0,1>`(Node 保活 Graph)
- `Session.run(graph, targets, feeds, loss_node=None)`,`loss_node` 传入才做反向
- 有了 Python 层,可以接 sklearn 等真实数据集做经典任务

## 逻辑回归: 信用卡欺诈检测

`credit_card_fraud.py` — 新增 op `matmul` / `sigmoid` / `log` 后,逻辑回归在 Python binding 上跑通:

- 数据: 真实 Kaggle `creditcard.csv` 不在本地, 用 sklearn `make_classification` 合成同分布数据
  (30 特征 / ~0.4% 欺诈 / 高度不平衡)
- 模型: `p = sigmoid(X@w + b)`, 加权二分类交叉熵 (`pos_weight = neg/pos` 处理不平衡)
- 关键点:
  - **bias 初始化为先验 log-odds**, 否则 p=0.5 处加权损失梯度为零 (固定点), 模型不学
  - 评估看 **recall/AUC** 而非 accuracy (全判负也有 99.6% 准确率)
- 结果: test AUC **0.936** (sklearn 同数据基线 0.944), fraud recall 0.895
- 训练图 20 节点, prune 后推理图 6 节点 (`X w b matmul add sigmoid`)
