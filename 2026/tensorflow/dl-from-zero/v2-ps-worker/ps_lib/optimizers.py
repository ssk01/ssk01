"""PS 侧优化器: 收到梯度后如何更新参数

对应 TF 初始提交就有的 6 种优化器概念; 这里实现 SGD 与 Adam。
"""
import numpy as np


class Optimizer:
    def __init__(self, name="sgd", lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
        self.name = name
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = {}   # Adam 一阶矩
        self.v = {}   # Adam 二阶矩
        self.t = 0

    def step(self, name, param, grad):
        if self.name == "sgd":
            return param - self.lr * grad
        if self.name == "adam":
            self.t += 1
            m = self.m.get(name, np.zeros_like(param))
            v = self.v.get(name, np.zeros_like(param))
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * grad ** 2
            self.m[name], self.v[name] = m, v
            mhat = m / (1 - self.beta1 ** self.t)
            vhat = v / (1 - self.beta2 ** self.t)
            return param - self.lr * mhat / (np.sqrt(vhat) + self.eps)
        raise ValueError(f"unknown optimizer {self.name}")
