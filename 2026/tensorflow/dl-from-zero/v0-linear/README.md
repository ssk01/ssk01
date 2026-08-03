# v0-linear — 最小可跑框架

dl-from-zero 系列的起点: 用 ~300 行 C++ 讲清"深度学习框架的最小骨架"。

## 概念

| 组件 | 文件 | 对应 TF |
|---|---|---|
| `Graph` + `Node`(type/name/inputs) | `graph/graph.h` | `Graph` / `Node` |
| `Tensor`(data + shape) | `framework/tensor.h` | `Tensor` + `TensorShape` |
| op 前向/反向 kernel | `kernels/kernels.h` | `OpKernel` |
| `Session::run`(拓扑排序 + 反向) | `session.h` | `Session` / `Executor` |

- 7 个 op: placeholder / variable / add / mul / sub / square / mean / sgd_step
- 反向: 每个 op 手写链式法则, Session 逆拓扑遍历, 梯度累加到 `node->grad`
- 训练: main.cpp 里手动 `a = a - lr * grad_a`(优化器还没变成图节点)

## 运行

```bash
make && ./build/train
```

200 轮收敛: a=1.97(真值 2), b=2.99(真值 3)。

## 局限(留给后续版本)

- 值存在 Node 里 → 图有状态, 不能并发(v3 解决)
- backward 是手写单独一趟反向遍历(v4 改成梯度子图)
- 没有 Python 层(v1 加 binding)
- 值/梯度存在 Node::output / Node::grad(v3 挪进 RunState)
