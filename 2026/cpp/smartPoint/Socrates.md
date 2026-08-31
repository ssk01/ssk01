# Socrates

### Q: `Uniqlo& operator=(const Uniqlo& t) = delete;` 和原来的 `Uniqlo operator=(Uniqlo& t) = delete;` 差别在哪？需要再加 `(Uniqlo& t)` 或 `(Uniqlo t)` 重载吗？
- 差别有三点：① const 限定——`const Uniqlo&` 能同时匹配非 const 左值 / const 左值 / 右值，`Uniqlo&` 只能匹配非 const 左值；② 返回类型——规范 copy assignment 返回 `Uniqlo&`（支持链式 `a=b=c`），原版返回 `Uniqlo`（按值）；③ 错误提示——`const Uniqlo&` 删除后所有赋值路径都报清晰的 "use of deleted function"，`Uniqlo&` 对 const 对象会报 "no matching function"（语义混乱）。
- 不需要加重载。单个 `const Uniqlo&` 已覆盖所有左值/右值匹配路径；后续加 move 赋值 `operator=(Uniqlo&&)` 后，右值会优先走 move，const 右值才会落到被删的 copy 上，行为正确。
(2026-08-31)

### Q: move 构造用 `std::swap(u.t_, t_)` 会出什么问题？
- 模板类的成员函数是惰性实例化的：只要没人调用，编译错误/UB 不会暴露——旧 demo 全过，但新特性测试一触发就现形。
- `std::swap(u.t_, t_)` 在 move 构造里是错的：`t_` 此时尚未初始化，swap 会把未初始化的垃圾指针写进被移动对象 `u`，其析构 `delete` 垃圾指针 = UB（实测 SIGABRT）。正确写法：`t_(u.t_) { u.t_ = nullptr; }`——把指针搬走、源置空。
- move 赋值用 swap 恰好"能工作"（被移动方持有对方旧资源、由其析构收尾），但语义和 std 不一致（std 要求 moved-from 为空）；仍建议按标准语义写。
- macOS + 新版 Xcode 的 ASan 启动即挂是工具链已知问题，UBSan 正常可用。
(2026-08-31)
<!-- 以下继续记录 -->
