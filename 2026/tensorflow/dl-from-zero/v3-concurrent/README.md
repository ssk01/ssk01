# v3-concurrent — TF-like 图/运行时状态分离 (dl-from-zero 系列第 3 版)

dl-from-zero 系列第 3 迭代: 把框架改成 **TF-like 的图/运行时状态分离**,让同一张图可以并发跑。

## 为什么改

之前(第 1/2 迭代)把 `output`/`grad` 存在 Node 上 —— **图本身有状态, 同一张图没法并发跑**。
TF 不是这么干的, 它分三层:

| 层 | 存什么 | 本实现对应 |
|---|---|---|
| 图节点 (Graph Node) | 纯静态描述, 不存值 | `graph/graph.h` 的 `Node`(只有 type/name/inputs/attr) |
| 执行期临时值 (Executor Entry) | 每次 run 的张量 | `runtime.h` 的 `RunState`(outputs/grads) |
| 变量持久状态 (VariableOp 的 Var buffer) | 跨 run 的变量值 | `Session::vars_` |

因为运行值都在每次 run 的 `RunState` 里, **同一张图可以被多个线程同时跑**,互不干扰。

## 主要改动

1. **`Node` 变纯静态** — 删掉 `output`/`grad`, 值全部移入 `RunState`
2. **`Session` 持有变量** — `vars_`(带锁)+ `assign()` / `var_value()`
3. **`sgd_step` 变成真正能用的优化器节点** — 对应 TF `ApplyGradientDescent`, backward 后统一应用梯度; 之前的"死代码"问题解决, demo 不再手动更新参数
4. **编译缓存** — `Session::plan()` 按 (图, 目标集) 缓存拓扑执行计划, 对应 TF `GetOrCreateExecutors`("图只构建一次", 也省掉每步排序)
5. **topo 排序重写** — 两阶段直白版: 反向 BFS 收集可达集 + DFS 后序(=上游在前的执行序); 去掉原来的 children 表 + 入度表
6. **静态空张量** — `kEmptyTensor()`, 应用 Abseil Performance Hints 的 static-empty 模式

## 运行

```bash
make && ./build/train          # 训练 + prune + 8 线程并发推理 (C++)
make pybind && python3 demo_linear.py
python3 demo_concurrent.py     # Python 版并发推理 (4 线程)
python3 credit_card_fraud.py   # 逻辑回归 (新 API, 用 sgd_step)
```

## C++ 并发推理结果

训练收敛 (a=1.97, b=2.99) 后, 8 个线程在**同一张图**上各跑 5000 次推理, 每个线程结果都正确:

```
concurrent inference: 8 threads x 5000 runs on the same graph:
  thread 0  x=-4  mean_abs_err=0.104
  ...
  thread 4  x=0   mean_abs_err=0.009
```

## 设计要点

- **值跟 run 走, 不跟图走**: `RunState` 是每次 run 的局部对象, 天然支持并发; 图只是只读的静态描述
- **变量是唯一的跨 run 状态**: 集中在 Session, 用锁保护; 并发训练同一个变量是另一回事(那是 PS/worker 要解决的, 见 dl-from-zero-2-ps-worker)
- **计划缓存**: 拓扑序只算一次, 复用之 —— 呼应"图只构建一次"和减少每步分配
- 局限: 训练时变量在 forward 里按值拷贝进 RunState, 没有做 buffer 共享(概念优先, 未优化)
