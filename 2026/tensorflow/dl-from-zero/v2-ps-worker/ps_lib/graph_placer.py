"""3.4 图切分 (概念展示)

把训练图按 job 放置, 在切边上插 Send/Recv:
  - PS 子图:  变量节点 (w, b) + 优化器 (参数更新)
  - worker 子图: X, matmul, add, sigmoid, 损失, 梯度
  - 切边: 变量 w/b —— 训练时 worker 从 PS pull(Recv) 参数, 把梯度 push(Send) 给 PS
真实执行时 worker 每步先 pull 参数再算梯度, 再 push 梯度 (见 worker.py)。
"""
import numpy as np


def initial_params(F):
    return {"w": np.zeros(F, dtype=np.float64),
            "b": np.float64(0.0)}


def show_partition(F):
    lines = [
        "graph partition (job placement):",
        f"  ps     : w [{F}], b []        <- 变量节点 + 优化器(SGD/Adam)",
        "  worker : X, matmul, add, sigmoid, loss, backward  <- 计算子图",
        "  cut edges (Send/Recv): pull(w,b) 参数 / push(grad_w, grad_b) 梯度",
    ]
    return lines
