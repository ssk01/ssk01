# Plato.md - 项目约定

### PS/Worker 概念优先
- 先用文档讲清"分布式训练涉及哪些模块"，再动手；实现顺序：cluster → transport → ps_server → 图切分 → master → 同步/异步训练 (2026-08-03)

### 单机多进程起步
- 通信层先不用 gRPC，socket/pickle 即可；把"图切分 + Send/Recv"这个核心概念跑通 (2026-08-03)

<!-- 以下继续记录 -->
