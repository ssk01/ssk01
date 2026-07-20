# Euler Tour / DFS序 三题

三题的共同工具：**tin/tout 区间编号**（也叫 Euler Tour、DFS序）。

对树做一次 DFS，进入节点时分配 `tin`，离开时记录 `tout`（子树内最大 tin）。则：

- **"v 在 u 的子树中" ⇔ tin[u] ≤ tin[v] ≤ tout[u]**（O(1) 判定）
- 子树对应一个连续的 `[tin, tout]` 区间

```
        0 [tin=1, tout=6]
       / \
   1[2,3] 2[4,6]
          / \
     3[5,5] 4[6,6]
```

---

## #331 · Verify Preorder Serialization

**题意**：给定先序遍历序列（`"1,2,#,#,3,#,#"` 形式），判断是否合法二叉树。

**解法**：slot 法。初始 1 个空位，遍历每个节点：
- 消耗 1 个 slot，若 slot < 0 返回 false
- 非 `#` 节点生成 2 个 slot（左右子树）
- 遍历结束时 slot == 0 返回 true

O(n)，O(1)。与 tin/tout 的关系：前序序列化本质就是 dfs 的轨迹，slot 法模拟了 dfs 栈的消去过程。

---

## #2003 · Smallest Missing Genetic Value

**题意**：给树（parent 数组）和每个节点互不相同的基因值 nums[i]。对每个节点，找出其子树中缺失的最小正整数。

**解法**：

1. 跑一次 DFS 分配 tin/tout
2. 关键观察：只有含值 **1** 的节点到根的路径上答案 > 1。其他节点答案必然是 1（因为 1 不在它们的子树里）。
3. 从值 1 所在节点出发，沿父链向上逐级合并。每级维护一个 `seen` 集合（或直接用一个渐进递增的指针），碰到 seen 中已有值就跳过，找第一个缺失值。
4. 路径外的所有节点答案直接是 1。

时间复杂度 O(n)。核心是 tin/tout 不用来枚举子树（那会 O(n²)），而是用来**判定节点是否在值 1 所在子树内**。

---

## #2322 · Minimum Score After Removals

**题意**：删两条边把树切成三部分，使三部分异或值的 max-min 最小。

**解法**：

1. 任选根做 DFS，分配 tin/tout，同时计算每个节点子树的 XOR（`subtree_xor[node]`）
2. 枚举两条要删除的边。每条边连接父子 (u,v)，约定 v 是 u 的子节点（tin[u] < tin[v]）
3. 枚举时用 tin/tout 判断两条边的位置关系，分三种情况计算三部分 XOR：

| 关系 | 条件 | 三部分 XOR |
|---|---|---|
| 边 B 在边 A 的子树内 | `tin[a_v] ≤ tin[b_v] ≤ tout[a_v]` | `sub[b_v]`, `sub[a_v] ^ sub[b_v]`, `total ^ sub[a_v]` |
| 边 A 在边 B 的子树内 | `tin[b_v] ≤ tin[a_v] ≤ tout[b_v]` | 对称 |
| 两棵子树无包含关系 | 互不包含 | `sub[a_v]`, `sub[b_v]`, `total ^ sub[a_v] ^ sub[b_v]` |

4. 计算得分 = max(a,b,c) - min(a,b,c)，取最小。O(n²)

三道题都依赖于同一个核心能力：**用 tin/tout 在 O(1) 时间内判定树上节点的祖先-后代关系**。
