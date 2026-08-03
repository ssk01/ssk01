# Plato.md - 项目约定

### TF-like 的三层分离
- 图节点纯静态(不存值); 每次 run 的临时值在 RunState; 变量持久状态在 Session。这是 dl-from-zero-3 起的地基, 后续迭代沿用 (2026-08-03)

<!-- 以下继续记录 -->
