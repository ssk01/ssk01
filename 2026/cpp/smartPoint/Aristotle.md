# Aristotle

- 先写 demo 单测，验证当前 Uniqlo 实现（raw pointer 构造 + 析构 + 拷贝删除）
- 让我列出 Uniqlo 还需要实现哪些 feature（对照 std::unique_ptr）
- 让我继续测下去，看还有哪些单测跑不过
- 先讨论 EBO 怎么实现（先学老版本继承式，`[[no_unique_address]]` 新版本以后再说）[decision]
- swap 要连同 deleter 一起交换，pointer 和 deleter 是一对 [decision]
- 总结实现过程到文档、看看还有哪些 demo 跑不通、commit；剩下的 feature 后续再处理 [decision]
- 让我看 shy_ptr.h 能不能跑通 cppreference shared_ptr 页面的 demo；有问题先别修，先汇报，让我决定怎么处理 [decision]
- a、b 修好后先 commit 一版（我自己改的 friend/const/expired 判定）；下一步做类型擦除：构造时按真实类型 Y 记录 deleter，delete 按 Y 走，对齐 std，让 ~Base 非虚的 demo 也能正确析构 [decision]
- 类型擦除做完后：拷贝构造/赋值改收 const&、观察者标 const、assign 补释放旧引用与 return *this、lock 改按值返回，一次性 commit [decision]
- 我加的 deleter 构造修语法后仍炸（无捕获 lambda 引用局部 d），让我去读官方 libc++ 源码看 deleter 怎么存，然后按官方案子实现自定义 deleter [decision]
- deleter 存储不用 [[no_unique_address]]，改用 libc++ 老派 __compressed_pair_elem 的 CanBeEmptyBase 分派（空且非 final → 继承式 EBO，否则存成员）[decision]
(2026-09-04 18:15)
<!-- 以下继续记录 -->
