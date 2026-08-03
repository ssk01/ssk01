# Plato.md - 项目约定

### 梯度子图是新的反向机制
- 从 dl-from-zero-4 起, 反向不再用手写 backward 遍历, 而是 build_gradients 生成梯度子图(与正向同图, 可被 prune) (2026-08-03)

### RunState 用稠密数组
- 运行期值按 node->id 存 vector, 不用 unordered_map; 编译期定槽、运行时纯下标 (2026-08-03)

<!-- 以下继续记录 -->
