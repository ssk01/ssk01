# Socrates — Q&A 记录

### Q: 为什么 spill 活得最远的那个？
线性扫描的核心启发式是 farthest-end —— 在寄存器不够用时，spill 掉 live range 终点最远的变量。直觉：如果必须让一个人没柜子用，选需要柜子时间最长的——让他费点事（栈上），柜子留给很快就能用完的人。这是 interval scheduling 的贪心：你有一组时间区间，要在有限的机器槽位上安排它们。每次冲突时踢掉"结束时间最晚"的区间，能最大化当前槽位的利用率——等价于经典的"最早结束时间优先"调度策略的对偶形式。已知这类贪心策略在 interval graph coloring 上是精确最优的，live ranges 的结构近似于 interval graph，所以启发式在实践中接近最优。
(2026-06-23 10:00)

### Q: spill_scratch2 有什么用？什么情况下会出现？
`spill_scratch2` 是第二个临时寄存器（如 R7/x26），用于处理同一条指令引用多个 spilled 变量的情况。当前简化版只用 `spill_scratch` 一个寄存器做 reload/store，如果一条指令同时用到两个 spilled 变量，第二次 load 会覆盖第一个 load 的结果。例如 `add r1, r2, r3` 中 r2 和 r3 都被 spill 了，需要交替使用两个 scratch 寄存器。在实践中 spill 率通常只有 5-10%，同一条指令中两个操作数同时被 spill 的概率是 spill 率的平方——大约 1% 不到。
(2026-06-23 10:07)

### Q: 线性扫描复杂度是 O(n) 还是 O(n log n)？
是 O(n)。Poletto-Sarkar 原论文的结论就是 O(n)——唯一排序是按 start 排序 intervals，这在编译器里天然有序（指令顺序 = interval 顺序）；active 列表大小 ≤ 物理寄存器数（常数），所有 per-iteration 操作都是 O(1)。笔记里之前写 O(n log n) 是错的。
(2026-06-23 10:15)

### Q: V8 的线性扫描实现与 demo 有什么异同？
结构同源（Wimmer 线性扫描），但 V8 成熟得多。相同：按 start 排序处理 interval，先 expire 再 allocate/spill，都是 O(n)。四大升级：(1) Active 管理三分——active/inactive/unhandled 三个列表，inactive 随着扫描推进切换回 active；(2) Split 取代全量 spill——把 live range 切成多个子段，只 spill 真正冲突的那一小段；(3) Spill 启发式用"free-until"而非 farthest-end——对每个寄存器计算下次冲突点，选空闲窗口最长的；(4) 无专用 scratch 寄存器——spill load/store 通过 gap move 框架插入。最关键架构决策是 live range splitting + inactive 列表。
(2026-06-23 10:25)

### Q: V8 没有预留 scratch 寄存器，spill move 要用寄存器中转，全用完了怎么办？
正常情况不会发生——gap move 插在指令之间，此时前一条指令刚完成（输入寄存器已释放），后一条还没开始，可用临时寄存器比指令内多。极端情况有兜底：GapResolver 如果遇到 move 循环且没有空闲寄存器，会 `assembler_->Push()` 把一个值临时推到硬件栈上，释放出它的寄存器当 scratch，完成循环后再 `PopTempStackSlots()` 恢复。不依赖预留给寄存器分配器的 scratch，用的是硬件栈——无限容量、无规划开销。
(2026-06-23 10:30)

### Q: greedy_color 里为什么没看见"寄存器用完"的 case？
图着色没有全局寄存器池——它是并行分配，所有节点一起着色。每个节点只看邻居：`for c in range(K): if c not in neighbor_colors: assign; break; else: spill`。`if not assigned` 就是"寄存器用完"的 case——K 个颜色全被邻居占了，只能 spill。和线性扫描表象不同，但等价。
(2026-06-23 10:32)

### Q: 图着色为什么不需要"归还"寄存器？
干涉图已经把时间维度的冲突全部编码为空间维度的边——两个 interval 不重叠就不会有边，自然可以共享同一个颜色（物理寄存器）。着色是一次性的全局分配，没有"先用后还"的时序。
(2026-06-23 10:35)

<!-- 以下继续记录 -->
