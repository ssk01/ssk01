"""ps_lib - PS/Worker 单机多进程分布式训练 (概念实现)

对应设计文档第 3 节的模块清单:
  cluster.py    -> 3.1 集群描述与身份
  transport.py  -> 3.3 通信层 (TCP + pickle)
  graph_placer.py -> 3.4 图切分 (概念展示)
  optimizers.py -> 3.6 PS 侧优化器 (SGD / Adam)
  ps_server.py  -> 3.6 参数服务端
  worker.py     -> 3.7 Worker 端
  master.py     -> 3.5 Master 协调 + 3.8 同步/异步
"""
