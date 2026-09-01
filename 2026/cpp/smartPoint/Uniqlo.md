# Uniqlo — 手写 unique_ptr 学习笔记

> C++17 手写 `std::unique_ptr`，类名 `Uniqlo`。完整移植了 cppreference 主页面 + 全部成员函数子页面的 demo 作为测试基准。

## 当前实现状态

`unique_ptr.h` 已实现：

| 功能 | 状态 |
|---|---|
| 构造（裸指针 / 传 deleter） | ✅ |
| 析构 / reset / release / get | ✅ |
| move 构造 / move 赋值（拷贝已删除） | ✅ |
| `operator*` / `operator->`（含 const） | ✅ |
| `explicit operator bool` | ✅ |
| swap（指针 + deleter 一起换） | ✅ |
| 自定义 deleter 模板参数 + `get_deleter()` | ✅ |
| EBO（空 deleter 不占空间，`sizeof == 8`） | ✅ |
| `make_uniqlo<T>(args...)` 工厂 + 完美转发 | ✅ |

**Demo 测试**：17 个 cppreference demo 移植到 `demos/`，当前 7 过 11 挂（挂的全是未实现 feature，非回归）。

**编译命令**：`g++ -std=c++17 -Wall -Wextra -I. demos/demo_xxx.cpp`

## 未实现 feature（TODO，后续再做）

1. 数组特化 `Uniqlo<T[]>`（`operator[]`、`delete[]` 元素指针类型、`make_uniqlo<T[]>(n)`）
2. 默认构造 + `reset()` 无参（`void reset(T* t = nullptr)`）
3. 比较运算符（`==`/`!=`/`<` 与另一个 unique_ptr 或 nullptr）
4. 多态转换 move 构造（`Uniqlo<D> → Uniqlo<B>`）
5. 引用 deleter `Uniqlo<T, D&>`
6. `std::hash<Uniqlo<T>>` 特化

---

## 实现过程中的关键知识点

### 1. move 语义与所有权

- move 构造必须**搬走指针 + 源置空**：`t_(u.t_) { u.t_ = nullptr; }`
- **不能用 `std::swap(u.t_, t_)` 实现 move 构造**：move 构造时 `t_` 未初始化，swap 会把未初始化垃圾写进 `u`，其析构 `delete` 垃圾指针 = UB
- move 赋值要防自移动 `if (this != &u)`，先 `delete` 旧资源再接管
- 模板类成员函数**惰性实例化**：buggy 函数不调用就不报错——这就是旧 demo 全过、新特性测试一跑就炸的原因

### 2. 值类别（value category）

- **值类别是表达式的属性，不是对象的属性**。对象被"命名"时，该表达式永远是左值（有身份、有地址）
- `int&& r = 5;` — `&&` 是**类型**（右值引用类型），`r` 一旦绑定就是命名对象 = 左值
- **`std::move` 存在的意义**：把命名左值转回右值表达式
- 按值收参的函数：实参左值 → 拷贝构造；实参右值 → 移动构造（重载决议 `T(const T&)` vs `T(T&&)`）
- `int` 拷贝 == 移动，所以左右值无差别；`unique_ptr` 拷贝被删，所以必须右值（`std::move`）

### 3. `std::forward` 的本质

- `std::forward<T>(x)` = `static_cast<T&&>(x)`——**一个 cast**
- 输出类型**只由 `T` 决定，与实参值类别无关**（引用折叠规则）
- **推导语境**（`template<class T> f(T&& x)`）：T 从实参推，左值 → `T=T&` → 保左；右值 → `T=T` → 保右（"完美转发"的表现）
- **显式指定语境**（`forward<Deleter>(...)`）：T 写死，普通类型 → `Deleter&&` → move；引用类型 → `D& &&` 折叠 `D&` → 保左
- move 构造里 `forward<Deleter>(u.get_deleter())`：`get_deleter()` 返回左值引用，但 T 写死为 `Deleter`（普通类型）→ cast 成右值 → 移动 deleter
- **为什么不用 `std::move`**：`move` 无脑转右值；对引用 deleter `D&` 会编译错（非 const 左值引用不能绑右值）。`forward` 能移就移、是引用就绑引用
- `std::forward` 必须写显式模板参数（故意不能推导）

### 4. 完美转发（make_uniqlo）

- `args` 在函数体内是命名变量 = 左值；不 forward 就一律拷贝
- 后果：① move-only 类型（如 `std::unique_ptr`）编译不过；② 白白多一次拷贝
- 转发的价值不在参数多少，而在**参数的值类别**——同一工厂同时支持左值（拷）/右值（移）/move-only（只能移）

### 5. EBO（Empty Base Optimization）

- **空类独立存在要 1 字节**（两个对象不能同址）；**空基类子对象可以 0 字节**，与第一个数据成员重叠
- `class Uniqlo : private Deleter { T* t_; }` — 空 deleter 贴到 `t_` 上，`sizeof == 8 == 裸指针`
- 关键语法：构造初始化列表 `: Deleter(d)` 把外部 deleter 种进基类子对象；用 `static_cast<Deleter&>(*this)` 上转型取回
- `get_deleter()` const 版必须 `static_cast<const Deleter*>(this)`，不能去 const
- **现代 libc++ 不用继承，用 C++20 `[[no_unique_address]]`** 成员属性，效果相同且支持引用 deleter（引用不能当基类）
- **本项目最终采用"分派式"**（老版 libc++ `__compressed_pair_elem` 思路，C++17 可写）：`DeleterHolder<D>` 用 `is_class && is_empty && !is_final` 判断——能继承就继承（EBO 压 0），否则存成员（函数指针/引用/非空类）。纯继承式只支持类类型 deleter，函数指针 `void(*)(T*)` 会"base specifier must name a class"，这就是需要分派的原因

### 6. 删除器（deleter）

- `DefaultDeleter<T>`：`void operator()(T* p) const { delete p; }`；数组版用偏特化 `DefaultDeleter<T[]>` 做 `delete[]`（不能靠重载，`T[]*` 语法非法）
- **unique_ptr 从不直接 `delete`，而是 `deleter_(ptr_)`**——删除动作委托给 deleter 对象，这是自定义 deleter 能工作的原因
- **`delete` 不会把指针置空**：`delete t_` 后 `t_` 仍持有悬空地址。先 delete 再 swap 会把悬空地址换进别的对象 → 双重释放
- **`delete nullptr` 是安全 no-op**，不需要判空；真正要防的是"值未初始化"，不是"值为 null"
- 有状态 deleter 按值存进 unique_ptr，修改 `get_deleter()` 不影响外部原对象（与 std 语义一致）

### 7. `forward` vs `move`（deleter 移动）

- move 构造移 deleter 用 `forward<Deleter>`：普通类型 → 移动；引用类型 `D&` → 折叠保左，绑外部对象
- `std::move` 对引用 deleter 会炸（编译错），因为 `D&` 不能绑右值

### 8. swap 的配对语义

- **pointer 和 deleter 必须一起换**——它们是一对（指针 → 释放方式）
- 只换指针不换 deleter：函数指针 deleter（`void(*)(T*)`）场景下会错配 → `free` 释放 `new` 内存 = **UB**（编译能过，运行时炸，类型系统抓不到）
- 空 deleter 之间 swap = no-op（三次 move 全空），无害且免费——所以无条件写 `std::swap(ptr) + std::swap(deleter)` 两端通吃

---

## 实现变更记录（CHANGELOG）

> 每次为实现某个 demo 的改动，在这里记一版简单 diff 介绍，方便回看。

### 2026-08-31 · commit 78faf4b — 重构 + 补全 4 个 feature

**改了什么**（`unique_ptr.h`）：
1. **重构 UniqloBase 公共层**：把两个类重复的"存储 + 所有权"逻辑抽到 `UniqloBase<T, Deleter>`（EBO 继承 deleter），`Uniqlo<T>` 和 `Uniqlo<T[]>` 变薄包装，只加各自的存取运算符（`*`/`->` vs `[]`）
2. **多态转换 move 构造**：`template<class U, class E> Uniqlo(Uniqlo<U,E>&&)`，条件 `is_convertible_v<U*, T*>`；同时给 `DefaultDeleter<T>` 加了转换构造/赋值（`default_delete<D> → default_delete<B>`）
3. **比较运算符**：`==`/`!=`/`</>`/`<=`/`>=`（owner 语义，比 `get()`）+ 与 `nullptr` 比较
4. **std::hash 特化**：单一特化 `hash<Uniqlo<T,D>>` 同时覆盖单对象和数组
5. **make_uniqlo 数组工厂修正**：数组重载的模板参数方向改对（`Uniqlo<T>` 而非 `Uniqlo<T[]>`），变参版加 `enable_if<!is_array_v<T>>` 排除数组——否则 `make_uniqlo<int[]>(5)` 精确匹配变参版（`int&&` 优于 `int→size_t` 转换）而走 `new int[](5)` 报错

**效果**：17/17 demo 全过。

### 2026-08-31 · commit 7b1fca4 — 分派式 EBO（函数指针/引用 deleter）

**改了什么**（`unique_ptr.h`）：
1. 新增 `DeleterHolder<D>` 存储层（老 libc++ `__compressed_pair_elem` 思路）：
   - `CanEbo = is_class && is_empty && !is_final` → true 时 private 继承（EBO 压缩，0 字节）
   - false 时存成员（函数指针/引用/非空/final 类走这里，函数指针 8 字节、引用存引用）
2. `UniqloBase` 从 `: private Deleter` 改为 `: private DeleterHolder<Deleter>`
3. deleter 构造统一 `std::forward<Deleter>`（引用 deleter 保持引用、普通类型走移动）

**恢复了注释段落**：
- `demo_main.cpp` §3（FILE 函数指针 deleter）、§4（lambda 转函数指针）
- `demo_ctor.cpp` `Uniqlo<Foo, D&>` 引用 deleter 两处

**效果**：函数指针（16B）/引用 deleter（写回外部对象）解锁，空 deleter 仍 8B，17/17 demo 全过。

---

## 参考

- 官方实现分析：libc++ `__memory/unique_ptr.h`（`_LIBCPP_COMPRESSED_PAIR`）+ `__memory/compressed_pair.h`（`[[no_unique_address]]` 实现压缩）
- cppreference：https://en.cppreference.com/cpp/memory/unique_ptr
