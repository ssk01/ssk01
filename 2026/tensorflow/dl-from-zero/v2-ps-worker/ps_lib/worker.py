"""3.7 Worker 端

独立进程: 持有本机训练数据分片, 每步:
  1. pull_all  (Recv): 从 PS 拉取参数
  2. forward+backward: 在**当前 mini-batch** 上算梯度 (worker 子图)
  3. push_all  (Send): 把梯度推给 PS

数据分片按 batch 切分, 每个 epoch 重新 shuffle —— 这才是真实的 mini-batch SGD
(不是每步都用整片数据算全量梯度)。
"""
import socket

import numpy as np

from .model import LinearModel
from .transport import recv_msg, send_msg


def run_worker(ps_addr, X, y, steps, batch_size, lr, mode, log_every=20,
               result_q=None, worker_id=0, seed=0):
    sock = socket.create_connection(ps_addr)
    model = LinearModel(X.shape[1])
    rng = np.random.default_rng(seed + worker_id)

    def rpc(msg):
        send_msg(sock, msg)
        return recv_msg(sock)

    n = len(X)
    n_batches = max(1, (n + batch_size - 1) // batch_size)
    order = rng.permutation(n)
    new_params = None

    for k in range(steps):
        # 跨 epoch 时重新 shuffle 分片, 再顺序取 mini-batch
        if k % n_batches == 0:
            order = rng.permutation(n)
        pos = (k % n_batches) * batch_size
        idx = order[pos:pos + batch_size]
        Xb, yb = X[idx], y[idx]

        params = rpc({"type": "pull_all"})["params"]
        w, b = params["w"], params["b"]
        loss, gw, gb = model.loss_and_grads(Xb, yb, w, b)
        new_params = rpc({"type": "push_all",
                          "grads": {"w": gw, "b": gb}, "step": k})["params"]

        if k % log_every == 0 or k == steps - 1:
            # 打日志用整片 loss/acc, 反映真实收敛; 梯度始终来自 mini-batch
            full_loss, _, _ = model.loss_and_grads(X, y, new_params["w"], new_params["b"])
            acc = model.accuracy(X, y, new_params["w"], new_params["b"])
            print(f"[worker{worker_id} {mode}] step {k:4d}: "
                  f"loss={full_loss:.4f} train_acc={acc:.4f}", flush=True)

    sock.close()
    if result_q is not None:
        full_loss, _, _ = model.loss_and_grads(X, y, new_params["w"], new_params["b"])
        result_q.put({
            "worker": worker_id,
            "final_loss": float(full_loss),
            "final_acc": model.accuracy(X, y, new_params["w"], new_params["b"]),
        })
