"""
Tree BFS Demo — 树的广度优先遍历 (层序遍历)

树结构:
        A
      / | \
     B  C  D
    / \    / \
   E   F  G   H

BFS 输出 (按层): A → B C D → E F G H
"""

from collections import deque
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


# ─── 1. 基础 BFS (逐节点) ───
def bfs_basic(root: TreeNode) -> List[str]:
    """TODO: 你来实现"""
    pass


# ─── 2. BFS 按层输出 ───
def bfs_by_level(root: TreeNode) -> List[List[str]]:
    """
    返回: [['A'], ['B','C','D'], ['E','F','G','H']]
    TODO: 你来实现
    """
    pass


# ─── 3. BFS 同时记录层号 ───
def bfs_with_level(root: TreeNode) -> List[tuple]:
    """
    返回: [(val, level), ...]
    TODO: 你来实现
    """
    pass


# ─── 测试 ───
if __name__ == "__main__":
    root = build_tree()

    print(f"基础BFS: {bfs_basic(root)}  (expected ['A','B','C','D','E','F','G','H'])")
    print(f"按层BFS: {bfs_by_level(root)}  (expected [['A'],['B','C','D'],['E','F','G','H']])")
    print(f"带层号:  {bfs_with_level(root)}  (expected [('A',0),('B',1),('C',1),('D',1),('E',2),('F',2),('G',2),('H',2)])")
