"""
LeetCode #2322 · Minimum Score After Removals on a Tree
https://leetcode.com/problems/minimum-score-after-removals-on-a-tree/

给定无向树 edges，节点值 nums。删除两条边，树被分成三个连通分量。
每个分量的异或值分别为 a, b, c。得分 = max(a,b,c) - min(a,b,c)。
求最小可能得分。
"""

from typing import List


def minimumScore(nums: List[int], edges: List[List[int]]) -> int:
    """TODO: 你来实现"""
    pass


# ─── 测试 ───
if __name__ == "__main__":
    tests = [
        ([1, 5, 5, 4, 11],
         [[0, 1], [1, 2], [1, 3], [3, 4]],
         9),
        ([5, 5, 2, 4, 4, 2],
         [[0, 1], [1, 2], [5, 2], [4, 3], [1, 3]],
         0),
        ([29, 29, 13, 31, 17],
         [[2, 0], [2, 3], [1, 4], [3, 4], [3, 1]],
         33),
    ]

    for nums, edges, expected in tests:
        result = minimumScore(nums, edges)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] nums={nums!r}, edges={edges!r}")
        print(f"       result={result}, expected={expected}")
