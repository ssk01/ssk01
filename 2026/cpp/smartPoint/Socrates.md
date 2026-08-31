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

### Q: 值类别：为什么"命名变量"一定是左值？std::move/forward 到底是什么？
- 值类别是**表达式**的属性，不是对象的属性。对象有名字 → 用它名字写出的表达式永远是左值（有身份、有地址、持久存在）；临时量没名字 → 右值。
- `int&& r = 5;` 里 `&&` 是**类型**（右值引用类型）不是值类别；`r` 一旦绑定就是命名对象 = 左值。这就是 `std::move` 存在的意义——把命名左值转回右值表达式。
- 按值收参：实参左值 → 拷贝构造；实参右值 → 移动构造（`T(const T&)` vs `T(T&&)` 重载决议）。`int` 拷贝==移动所以无差别；unique_ptr 拷贝被删所以必须右值。
- `std::forward<T>(x)` = `static_cast<T&&>(x)`——一个 cast，输出类型**只由 T 决定**，与实参值类别无关。推导语境（T 从实参推）表现成"左保左/右保右"；显式指定语境（T 写死）普通类型就是 move。
- move 构造 `forward<Deleter>(u.get_deleter())`：get_deleter 返回左值引用，但 T 写死为普通类型 → cast 成右值 → 移动。**不能用 `std::move`**：无脑转右值，引用 deleter `D&` 会编译错；`forward<D&>` 折叠回 `D&` 保左。
(2026-08-31)

### Q: EBO 到底"零"了什么？
- "zero" 不是整个对象变 0 字节——是**空 deleter 贡献 0 字节**。`t_` 指针 8 字节是物理下限。
- 空类独立存在要 1 字节（两对象不能同址）；**空基类子对象可 0 字节**，与首个数据成员重叠（`&基类 == &t_`，实测同地址）。
- 继承式 `class Uniqlo : private Deleter { T* t_; }`：空 deleter 贴到 t_ 上 → sizeof == 8 == 裸指针。
- 传 deleter 给基类 = 构造初始化列表 `: Deleter(d)`；用 deleter = `static_cast<Deleter&>(*this)` 上转型。
- 现代 libc++ 不用继承，用 C++20 `[[no_unique_address]]` 成员属性，效果相同且支持引用 deleter（引用不能当基类）。
(2026-08-31)

### Q: swap 为什么必须连带交换 deleter？
- pointer 和 deleter 是一对（指针 → 释放方式）。只换指针不换 deleter，函数指针 deleter（`void(*)(T*)`）场景会错配：`free` 释放 `new` 内存 = UB（编译能过，类型系统抓不到，纯运行时炸）。
- 空 deleter 之间 swap = no-op（三次 move 全空），无害且免费——所以无条件写 `std::swap(ptr) + std::swap(deleter)` 两端通吃。
- `delete t_` 不会置空指针（悬空地址还在）；`delete nullptr` 是安全 no-op 不用判空。
(2026-08-31)
<!-- 以下继续记录 -->
