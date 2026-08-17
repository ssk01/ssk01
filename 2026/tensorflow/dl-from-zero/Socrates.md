# Socrates.md - 问答记录

### Q: 怎么指定哪些算子在 GPU 哪些在 CPU? 还是整张图设定?
per-op 形式, TF 没有"整张图设备"概念。每个 NodeDef 带 `device` 字符串 (如 "/gpu:0"), 不指定则由 simple_placer 自动放置: FilterSupportedDevices 按 kernel 注册表筛出该 op 支持哪些设备, 再按优先级 Order 选 (GPU=2 < CPU=3, 有 GPU kernel 就优先 GPU); 显式指定优先, 错误时 soft placement 兜底回退。设备是 op 级属性, 图切分 (graph_partition) 只是把落在不同设备的 op 拆成多个执行片段, 跨设备边补 _Send/_Recv 节点对通信 —— 切分是放置的后果, 不是放置的前提。
(2026-08-14)

<!-- 以下继续记录 -->
