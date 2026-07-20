"""
LeetCode #331 · Verify Preorder Serialization of a Binary Tree
https://leetcode.com/problems/verify-preorder-serialization-of-a-binary-tree/

给定 preorder 序列化字符串（节点值或 '#' 表示 null），判断是否合法。
"""


def isValidSerialization(preorder: str) -> bool:
    """TODO: 你来实现"""
    pass


# ─── 测试 ───
if __name__ == "__main__":
    tests = [
        ("9,3,4,#,#,1,#,#,2,#,6,#,#", True),
        ("1,#", False),
        ("9,#,#,1", False),
        ("#", True),
        ("1,#,#", True),
        ("1,#,#,#", False),
        ("#,1", False),
    ]

    for preorder, expected in tests:
        result = isValidSerialization(preorder)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] {preorder!r:40s} → {result} (expected {expected})")
