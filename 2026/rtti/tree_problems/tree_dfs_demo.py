"""
Tree DFS Demo — 树的深度优先遍历

树结构:
        A
      / | \
     B  C  D
    / \    / \
   E   F  G   H

DFS 输出 (前序): A B E F C D G H
"""

from __future__ import annotations
from typing import List


class TreeNode:
    def __init__(self, val: str):
        self.val = val
        self.children: List[TreeNode] = []


def build_tree() -> TreeNode:
    A = TreeNode("A")
    B = TreeNode("B"); C = TreeNode("C"); D = TreeNode("D")
    E = TreeNode("E"); F = TreeNode("F")
    G = TreeNode("G"); H = TreeNode("H")

    A.children = [B, C, D]
    B.children = [E, F]
    D.children = [G, H]
    return A


# ─── 1. 递归 DFS ───
def dfs_recursive(node: TreeNode) -> List[str]:
    """TODO: 你来实现"""
    pass


# ─── 2. 迭代 DFS (显式栈) ───
def dfs_iterative(root: TreeNode) -> List[str]:
    """TODO: 你来实现"""
    pass


# ─── 3. DFS 分配 tin/tout (Euler Tour 区间) ───
def dfs_order(root: TreeNode) -> List[str]:
    """
    进入节点时记录 tin, 离开子树时记录 tout.
    返回 [(val, tin, tout), ...]
    TODO: 你来实现
    """
    pass


# ─── 测试 ───
if __name__ == "__main__":
    root = build_tree()

    tests = {
        "递归DFS": dfs_recursive(root),
        "迭代DFS": dfs_iterative(root),
    }
    for name, result in tests.items():
        print(f"{name}: {result}  (expected ['A','B','E','F','C','D','G','H'])")

    orders = dfs_order(root)
    if orders:
        for val, tin, tout in orders:
            print(f"  {val}: tin={tin}, tout={tout}")
