"""3.1 集群描述与身份

ClusterSpec: 集群里有哪些 job (ps / worker), 每个 job 下有哪些 task (地址)
每个进程用 (job, task_index) 标识自己是谁。
对应 TF 的 ClusterSpec / ServerDef。
"""
from dataclasses import dataclass


@dataclass
class Task:
    job: str      # "ps" | "worker"
    index: int    # 0-based task 编号
    addr: str     # "host:port"

    def __str__(self):
        return f"{self.job}/{self.index}@{self.addr}"


class ClusterSpec:
    def __init__(self, jobs):
        # jobs: {"ps": ["127.0.0.1:9001"], "worker": ["127.0.0.1:9002", "127.0.0.1:9003"]}
        self.tasks = []
        for job, addrs in jobs.items():
            for i, addr in enumerate(addrs):
                self.tasks.append(Task(job, i, addr))

    def task(self, job, index=0):
        for t in self.tasks:
            if t.job == job and t.index == index:
                return t
        raise KeyError(f"no task {job}/{index}")

    def all_workers(self):
        return [t for t in self.tasks if t.job == "worker"]
