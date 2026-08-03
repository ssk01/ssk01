#!/usr/bin/env python3
"""dl-from-zero-2-ps-worker 入口: 单机多进程 PS 训练

用法:
  python3 train_ps.py --n_workers 2 --steps 200 --lr 0.5 --opt sgd --mode async
  python3 train_ps.py --mode sync --opt adam --lr 0.05

进程拓扑:
  master -> 启动 ps (参数服务) + n 个 worker (数据分片)
  worker -> PS: pull 参数 / push 梯度  (TCP + pickle)
"""
import argparse

from ps_lib.master import run_master


def main():
    ap = argparse.ArgumentParser(
        description="dl-from-zero-2-ps-worker: 单机多进程 PS 分布式训练")
    ap.add_argument("--n_workers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--opt", choices=["sgd", "adam"], default="sgd")
    ap.add_argument("--mode", choices=["async", "sync"], default="async")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=256,
                    help="每个 worker 每步采样的 mini-batch 大小")
    a = ap.parse_args()

    run_master(n_workers=a.n_workers, steps=a.steps, lr=a.lr,
               opt=a.opt, mode=a.mode, seed=a.seed, batch_size=a.batch_size)


if __name__ == "__main__":
    main()
