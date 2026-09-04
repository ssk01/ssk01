# Aristotle

- 先写 demo 单测，验证当前 Uniqlo 实现（raw pointer 构造 + 析构 + 拷贝删除）
- 让我列出 Uniqlo 还需要实现哪些 feature（对照 std::unique_ptr）
- 让我继续测下去，看还有哪些单测跑不过
- 先讨论 EBO 怎么实现（先学老版本继承式，`[[no_unique_address]]` 新版本以后再说）[decision]
- swap 要连同 deleter 一起交换，pointer 和 deleter 是一对 [decision]
- 总结实现过程到文档、看看还有哪些 demo 跑不通、commit；剩下的 feature 后续再处理 [decision]
- 让我看 shy_ptr.h 能不能跑通 cppreference shared_ptr 页面的 demo；有问题先别修，先汇报，让我决定怎么处理 [decision]
(2026-09-04 16:56)
<!-- 以下继续记录 -->
