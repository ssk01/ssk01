"""3.5 Master 协调 + 3.8 同步/异步

Master 职责:
  1. 生成数据并按 worker 分片
  2. 启动 ps 进程 (含初始参数), 拿到真实端口
  3. 启动 n 个 worker 进程 (各持一分片)
  4. 等待 worker 结束, 从 ps 拉最终参数, 在留出集上评估
"""
import socket

import multiprocessing as mp
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .graph_placer import initial_params, show_partition
from .model import LinearModel
from .ps_server import run_ps
from .transport import recv_msg, send_msg
from .worker import run_worker


def run_master(n_workers=2, steps=200, lr=0.5, opt="sgd",
               mode="async", seed=42, batch_size=256):
    # ---- 数据 (Master 侧数据加载; 与 PS 通信无关) ----
    X, y = make_classification(n_samples=20000, n_features=10, n_informative=8,
                               n_classes=2, weights=[0.85, 0.15],
                               class_sep=1.2, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed,
                                          stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    F = Xtr.shape[1]
    for line in show_partition(F):
        print("  " + line)

    # ---- 启动 PS ----
    ctx = mp.get_context("spawn")
    port_q = ctx.Queue()
    ps_proc = ctx.Process(target=run_ps,
                          args=(initial_params(F), n_workers, mode, opt, lr, port_q))
    ps_proc.start()
    ps_addr = ("127.0.0.1", port_q.get())

    # ---- 启动 workers ----
    result_q = ctx.Queue()
    shards = np.array_split(np.arange(len(Xtr)), n_workers)
    workers = []
    for i in range(n_workers):
        idx = shards[i]
        p = ctx.Process(target=run_worker,
                        args=(ps_addr, Xtr[idx], ytr[idx], steps, batch_size,
                              lr, mode, 20, result_q, i))
        p.start()
        workers.append(p)
    for p in workers:
        p.join()

    # ---- 拉最终参数, 在留出集上评估 ----
    sock = socket.create_connection(ps_addr)
    send_msg(sock, {"type": "pull_all"})
    final = recv_msg(sock)["params"]
    sock.close()

    model = LinearModel(F)
    acc = model.accuracy(Xte, yte, final["w"], final["b"])
    w_norm = float(np.linalg.norm(final["w"]))
    print(f"\n[master] final params: ||w||={w_norm:.3f}  b={float(final['b']):.3f}")
    print(f"[master] holdout accuracy = {acc:.4f}   (mode={mode}, "
          f"workers={n_workers}, steps={steps})")

    results = [result_q.get() for _ in workers]
    ps_proc.terminate()
    ps_proc.join()
    return results
