"""worker 子图: 逻辑回归 (numpy 版, 概念实现)

对应 dl-from-zero 框架里 worker 侧的前向+反向。
变量 w/b 位于 PS, 计算 (matmul/add/sigmoid/损失/梯度) 都在 worker。
"""
import numpy as np


class LinearModel:
    def __init__(self, F):
        self.F = F

    def forward(self, X, w, b):
        return 1.0 / (1.0 + np.exp(-(X @ w + b)))

    def loss_and_grads(self, X, y, w, b):
        p = self.forward(X, w, b)
        eps = 1e-12
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        d = p - y
        gw = X.T @ d / len(y)      # [F]
        gb = np.mean(d)            # 标量
        return float(loss), gw, gb

    def predict(self, X, w, b):
        return self.forward(X, w, b) > 0.5

    def accuracy(self, X, y, w, b):
        return float(np.mean(self.predict(X, w, b) == y))
