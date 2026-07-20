"""
LeetCode #2003 · Smallest Missing Genetic Value in Each Subtree
https://leetcode.com/problems/smallest-missing-genetic-value-in-each-subtree/

给定树 parents（根为 -1），每个节点有一个唯一基因值 nums[i] ∈ [1, 10^5]。
对每个节点，找出其子树中未出现的最小正整数。
"""

from typing import List


def smallestMissingValueSubtree(parents: List[int], nums: List[int]) -> List[int]:
    """TODO: 你来实现"""
    pass


# ─── 测试 ───
if __name__ == "__main__":
    tests = [
        ([-1, 0, 0, 2], [1, 2, 3, 4], [5, 1, 1, 1]),
        ([-1, 0, 1, 0, 3, 3], [5, 4, 6, 2, 1, 3], [7, 1, 1, 4, 2, 1]),
        ([-1, 0, 0, 1, 1, 2], [1, 5, 2, 3, 4, 6], [7, 1, 1, 1, 1, 1]),
        ([-1, 0, 0], [3, 1, 2], [4, 2, 1]),
    ]

    for parents, nums, expected in tests:
        result = smallestMissingValueSubtree(parents, nums)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] parents={parents!r}, nums={nums!r}")
        print(f"       result={result}, expected={expected}")
