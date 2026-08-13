# Plato.md - 项目约定

### dl-from-zero 是总目录, 版本按子目录演进
- `dl-from-zero/` 是总目录, 下面按 `vN-名称/` 分子目录(v0-linear, v1-prune, v2-ps-worker, v3-concurrent, v4-grad)
- 新版本 = 上一版 + 一个新概念, 从上一版拷贝代码
- 总目录 README 是主文档, 讲演进脉络; 各版本 README 讲自己 (2026-08-03)

### feature 文档的叙事方式: 从 0 到 1 的 meta 问题
- 写 feature/概念文档时, 每个 feature 按"从 0 到 1"叙事: 先讲前一版卡在哪个 meta 问题(打破了什么假设), 再讲为什么必须引入新机制, 最后讲怎么实现
- 核心概念要写详细; 技术细节(机制/op/源码)要考证到初始 commit 真实源码, 不凭记忆 (2026-08-12)

<!-- 以下继续记录 -->
