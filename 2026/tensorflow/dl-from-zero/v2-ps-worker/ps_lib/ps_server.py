"""3.6 参数服务端 (PS)

独立进程: 持有参数表 params: {name: ndarray}, 通过 TCP 服务两个 RPC:
  - pull_all   : worker 拉取全部参数
  - push_all   : worker 推送梯度
     async 模式: 每次推送立即应用 (谁先到谁先更新)
     sync  模式: 同一 step 的所有 worker 梯度都到齐, 取平均后一次性应用 (barrier)
"""
import socket
import threading

import numpy as np

from .optimizers import Optimizer
from .transport import recv_msg, send_msg


class _PsCore:
    def __init__(self, params, n_workers, mode, opt_name, lr):
        self.params = {k: np.array(v, dtype=np.float64) for k, v in params.items()}
        self.n_workers = n_workers
        self.mode = mode
        self.opt = Optimizer(opt_name, lr)
        self.cv = threading.Condition()
        self.pending = {}   # step -> [(conn, grads), ...]  (仅 sync)
        self.applied = -1   # 已应用的全局 step (仅 sync)

    def handle_conn(self, conn):
        try:
            while True:
                msg = recv_msg(conn)
                if msg is None:
                    break
                self._handle(conn, msg)
        except (ConnectionResetError, EOFError, BrokenPipeError):
            pass
        finally:
            conn.close()

    def _handle(self, conn, msg):
        if msg["type"] == "pull_all":
            send_msg(conn, {"type": "params", "params": self._snapshot()})
        elif msg["type"] == "push_all":
            grads, step = msg["grads"], msg["step"]
            if self.mode == "async":
                with self.cv:
                    for k, g in grads.items():
                        self.params[k] = self.opt.step(k, self.params[k], g)
                send_msg(conn, {"type": "ok", "params": self._snapshot()})
            else:  # sync: 到齐再应用
                with self.cv:
                    self.pending.setdefault(step, []).append((conn, grads))
                    if len(self.pending[step]) == self.n_workers:
                        names = list(grads.keys())
                        agg = {k: np.mean([g[k] for _, g in self.pending[step]], axis=0)
                               for k in names}
                        for k, g in agg.items():
                            self.params[k] = self.opt.step(k, self.params[k], g)
                        self.applied = step
                        self.cv.notify_all()
                    while self.applied < step:
                        self.cv.wait()
                send_msg(conn, {"type": "ok", "params": self._snapshot()})
        else:
            raise ValueError(f"unknown message {msg['type']}")

    def _snapshot(self):
        return {k: np.array(v, copy=True) for k, v in self.params.items()}


def run_ps(params, n_workers, mode, opt_name, lr, port_q):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(16)
    port = sock.getsockname()[1]
    port_q.put(port)
    print(f"[ps] listening 127.0.0.1:{port}  params={list(params)}  "
          f"mode={mode} opt={opt_name} lr={lr}")
    core = _PsCore(params, n_workers, mode, opt_name, lr)
    while True:
        conn, _ = sock.accept()
        threading.Thread(target=core.handle_conn, args=(conn,), daemon=True).start()
