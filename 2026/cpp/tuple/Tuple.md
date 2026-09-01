# Tuple — 手写 std::tuple 学习笔记

> 目标：C++20 手写 `std::tuple`。类名 `Tuple`，实现放 `namespace mytup`。
> 前置参考：同层 `smartPoint` 项目手写 unique_ptr 时对 EBO 的完整讨论，tuple 是 EBO 最典型的使用场景，务必复用。

## 当前实现状态

`tuple.h` 已实现（对应 std::tuple 全部常用接口）：

| 功能 | 状态 |
|---|---|
| 递归存储（`Tuple<T, Rest...>` 继承 `Tuple<Rest...>`）+ EBO 压缩（`sizeof == std::tuple`） | ✅ |
| `get<I>`（左值/const/右值）+ `get<T>` 按类型取 | ✅ |
| 值构造（条件 explicit）/ 默认（值初始化）/ 拷贝 / 移动 / 转换构造 / pair 构造 | ✅ |
| 拷贝/移动赋值 / 转换赋值 / pair 赋值 | ✅ |
| `make_tuple` / `tie` / `ignore` / `forward_as_tuple` / `make_from_tuple` | ✅ |
| `apply` / `tuple_cat` / 比较运算符 / 成员+自由 `swap` | ✅ |
| `std::tuple_size` / `tuple_element` 特化 + 结构化绑定 | ✅ |
| CTAD 推导指引（含 pair） | ✅ |
| 引用元素 `int&` / 函数指针元素 / move-only 元素 | ✅ |

**Demo**：
- `demos/demo_tuple.cpp`（主 demo，8 节）+ `demos/demo_vs_std.cpp`（对照 `std::tuple` 的 sizeof 与结构化绑定）
- `demos/demo_01_main.cpp` ~ `demo_16_deduction_guides.cpp`：cppreference tuple 全部页面的 Example 移植基准，见下表

**编译命令**：`g++ -std=c++20 -Wall -Wextra -I. demos/demo_xxx.cpp`

## cppreference demo 基准（15 过 1 挂）

把 cppreference 上 tuple 主页面 + 各成员/非成员函数页面的 Example 逐个移植（`std::tuple`→`mytup::Tuple`、`std::get`→`mytup::get`、`std::apply`→`mytup::apply`）。输出与 cppreference 逐字一致（含 tuple_cat 的引用保留语义）。

| demo | 来源页面 | 状态 | 说明 |
|---|---|---|---|
| demo_01_main | tuple 主页（get_student） | ✅ | 条件 explicit 让 `return {…}` 可隐式转换 |
| demo_02_ctor | tuple::tuple 构造 | ✅ | 默认/值/转换/pair 构造全过 |
| demo_03_assign | tuple::operator= | ✅ | 同型+转换+pair 赋值 |
| demo_04_swap | tuple::swap 成员 | ✅ | 成员 swap + make_tuple |
| demo_05_get | get(std::tuple) | ✅ | 按索引 get + 按类型 get<T> + tie |
| demo_06_apply | std::apply | ✅ | 折叠表达式打印 |
| demo_07_make_tuple | std::make_tuple | ✅ | unwrap_ref_decay 解包 std::ref |
| demo_08_tie | std::tie | ✅ | tie/ignore/CTAD/operator</tuple 字典序 |
| demo_09_forward_as_tuple | std::forward_as_tuple | ❌ | **std 互操作边界**：forward_as_tuple 本身可用，但 std::map 的 piecewise 构造只认 std::tuple，喂不进去我们的 Tuple |
| demo_10_tuple_cat | std::tuple_cat | ✅ | 元素类型保留引用（tie 的 int& 拼进结果，输出 42） |
| demo_11_make_from_tuple | std::make_from_tuple | ✅ | |
| demo_12_cmp | operator==,<,… | ✅ | 字典序递归比较 |
| demo_13_swap2 | swap(std::tuple) | ✅ | 成员 + 自由 swap |
| demo_14_tuple_size | tuple_size | ✅ | 编译期/运行期 |
| demo_15_tuple_element | tuple_element | ✅ | 含 long&/bool&&/volatile 元素 |
| demo_16_deduction_guides | 推导指引 | ✅ | CTAD + pair 推导指引 |

**待做**：`std::hash<Tuple>`、allocator 扩展构造（`uses_allocator`）、对照 libc++ tuple 源码逐行分析。

## Pair（2 元素特例）—— 与 Tuple 的对比

`pair.h` 里的 `mytup::Pair<T1, T2>`：概念上就是 `Tuple<T1,T2>`，但实现完全不同。

| | Tuple<T, Rest...> | Pair<T1, T2> |
|---|---|---|
| 元素个数 | 任意 N（变参模板） | 固定 2 |
| 存储 | 递归继承 `Tuple<Rest...>` + `head_` | 两个普通成员 `first` / `second` |
| 类型工具 | `TupleTypeAt` / `TupleTypeIndex` / `index_sequence` | 一个 2 行的 `PairElement` |
| `get<I>` | 递归下钻 `get<I-1>(tail())` | `if constexpr (I==0)` 硬编码 |
| 转发构造偷拷贝 | **有坑**（`Tuple(U&&, URest&&...)` 空包退化成 1 参数） | **没坑**（值构造固定 2 参数，永不跟拷贝/移动抢） |
| 条件 explicit | 需要（`return {…}` 场景） | 不需要（隐式转换无歧义场景少） |
| EBO | `[[no_unique_address]] head_` + 空基类 | `[[no_unique_address]] first/second` |

**可互转**：标准里 `std::pair` 和 2 元 `tuple` 可互相转换。本实现里 `Tuple` 有 pair 构造（demo_02 展示），`Pair` 有从 `Tuple<T1,T2>` 的构造（demo_pair 展示）。

**Demo**：`demos/demo_pair.cpp`（6 节：构造/get/结构化绑定、拷贝移动转换、比较+sort、swap+make_pair、与 tuple 互转、EBO 对比）。

## Pair2（继承式 EBO 版）—— 与 [[no_unique_address]] 的实测对比

`pair2.h` 的 `mytup::Pair2`：老版 libc++ `__compressed_pair_elem` 思路——`is_empty && !is_final` 就继承存（空基类压成 0），否则存成员。元素是基类子对象，所以只能暴露 `first()`/`second()` 访问器（没有 std::pair 的 `.first`/`.second` 成员），`get<I>` 走公开访问器。

**实测 sizeof（Clang，`demos/demo_pair2.cpp`）**：

| 元素 | Pair（no_unique_address） | Pair2（继承式） |
|---|---|---|
| `Empty1` + `int` | 4 | 4 |
| `Empty1` + `Empty2`（不同空类） | 1 | 1 |
| `Empty1` + `Empty1`（同类型空类） | 2 | 2 |
| `int` + `string` | 32 | 32 |
| `int&` + `string` | 32 | 32 |
| **`EmptyFinal`（final 空类）+ `int`** | **4** | **8** |

结论（推翻本笔记早先"继承式对同类型空类更优"的猜想）：
- **同类型空类 `Empty1+Empty1` 两种都是 2**——继承式也压不到 1：两个包装基类（`Pair2Elem<0,Empty1>` / `Pair2Elem<1,Empty1>`）虽类型不同，但各自内嵌同类型 `Empty1` 子对象，同类型对象不能同址，传递约束强制它们错开
- **final 空类是继承式唯一输的场景**（8 vs 4）：final 不能继承只能存成员，而纯继承式又不用 no_unique_address
- 结论：**`[[no_unique_address]]` 成员式全面不输继承式**（final 场景严格更优 + 实现更简单 + 能保留 `.first`/`.second` 成员）。继承式唯一的现实价值是 C++20 之前没有 no_unique_address 的历史方案；且带 `is_empty&&!is_final` 分派后，引用/函数指针元素也照常工作（对比 smartPoint Uniqlo 无分派直接继承的坑）

---

## 实现过程中的关键知识点

### 1. 递归存储结构

- `Tuple<T, Rest...> : public Tuple<Rest...>`，自己只持有 `head_`（第一个元素），剩下的全在基类里
- `tail()` 直接上转型返回基类引用；`get<I>` 递归下钻：`I==0` 取 `head()`，否则 `get<I-1>(tail())`
- `Tuple<>` 单独偏特化（空），作为递归终点；`Tuple<>` 占 1 字节（C++ 规定对象不能 0 尺寸）
- 类型工具 `TupleTypeAt<I, Types...>` 递归取第 I 个类型，用于 `get` 返回类型和 `tuple_element`

### 2. 转发构造函数会偷拷贝/移动（本实现踩的第一个坑）

- `Tuple(U&& u, URest&&... rest)` 对 `Tuple<int,string> b(a)` 的解析：
  转发模板绑定 `Tuple& → Tuple&` 是**纯 identity**；拷贝构造 `const Tuple&` 是 **identity + const 限定**
- 标准转换序列里 identity 严格优于 identity+qualification → **模板赢**，拷贝被偷 → 编译错
- 解法：SFINAE 排除首参是 Tuple 的情况
  `enable_if_t<!is_same_v<decay_t<U>, Tuple>, int> = 0`（`decay_t` 同时盖住 `Tuple&`/`Tuple&&`）
- 这是 C++ 所有"万能转发构造 + 拷贝构造"组合的通用坑（std::tuple/vector 内部同样有约束）

### 3. 命名空间：`get`/`apply` 不能放全局

- libc++ 的 `<memory>` **传递 include `<tuple>`**，`using namespace std;` 后 `std::apply` 可见
- 我们特化了 `std::tuple_size`，所以 `std::apply` 对 `mytup::Tuple` **可用**（有 size 即满足它的约束）→ 非限定 `apply(...)` 两个候选都 viable = 歧义
- 而 `get` 不冲突：`std::get` 的各个重载都要求 `std::tuple`/`pair`/`array`，对我们的类型全不 viable，ADL 会挑到 `mytup::get`
- 结论：实现放 `namespace mytup`，demo 里 `apply` 显式写 `mytup::apply`，`get` 靠 ADL 不用限定

### 4. EBO 两种路线的最终选择

- 继承式（smartPoint Uniqlo 用的）：**函数指针/引用类型做不了基类**，本实现直接放弃
- 成员式 `[[no_unique_address]] T head_`（C++20）：空元素与相邻对象重叠 → `Tuple<Empty1, Empty2, int>` 压到 4 字节，和 `std::tuple` 一致；引用/函数指针元素照常工作
- 对照：朴素成员实现 16 字节（1+pad+1+pad+4+pad）；EBO 后 4 字节，demo 里三者并排打出来

### 5. 结构化绑定的协议

- 结构化绑定要求：`std::tuple_size<E>` 完整 + 能通过 ADL 找到 `get<I>`
- 所以必须**特化 `std::tuple_size` / `std::tuple_element`**（放进 `namespace std`），且 `Tuple` 必须在某个有 `get` 的命名空间里
- `auto& [r] = t` 走 `get` 左值版返回可写引用 → 写穿生效（demo 3 展示了）

### 6. `apply` 的实现

- `apply(f, t)` 用 `make_index_sequence<tuple_size<decay_t<TupleLike>>::value>` 生成 `0..n-1`
- `apply_impl` 里 `std::invoke(f, get<Is>(std::forward<TupleLike>(t))...)` 包展开成 `f(get<0>, get<1>, ...)`
- `decltype(auto)` 返回保留引用/值语义

### 7. 转换构造/赋值：递归的魔力

- `Tuple<T, Rest...>` 的转换构造：`Tuple<Rest...>(static_cast<const Tuple<TailFrom...>&>(other))` + `head_(get<0>(other))`
  —— **首元素进 head，其余整个上转型成 tail 再递归**，一次转换构造就写成递归
- 约束用 `TupleConstructible`/`TupleAssignable`（逐元素 `is_constructible_v<To, From>` 的 bool_constant 递归），否则错会埋在模板实例化深处
- **声明了 move 构造后，隐式拷贝赋值会被 delete**（C++ 规则）——必须显式 `operator=(const Tuple&) = default`，否则 `t1 = t2` 直接挂（demo_04 踩过）
- pair 构造/赋值同样走"head + 尾巴（get<0> 写 tail）"

### 8. 条件 explicit（C++20 `explicit(E)`）

- std 的值构造只在**某个元素不可隐式转换**时才 explicit；全可转换时允许 `return {…}` 拷贝列表初始化
- 写法：`explicit(!is_convertible_v<U, T> || (... || !is_convertible_v<URest, Rest>))` 直接挂在模板参数后
- 主页 demo 的 `return {3.8, 'A', "Lisa Simpson"}` 靠它才过得去（string 可隐式从 const char* 构造）

### 9. `tuple_cat` 的元素类型是"源 tuple 的声明类型"，不是 get 的 decltype

- 结果元素类型 = 各源 tuple 的**元素类型包直接拼接**（所以 `tie(n)` 的 `int&` 会进结果，输出 42）
- 每个元素用 `get<I>(std::forward<Ti>(tpl))` 构造：左值源拷贝、右值源移动、引用源绑引用
- 实现：`Cat<Res, Tuples...>` 逐 tuple 消耗，`build_impl` 用 `index_sequence<Is...>` 把当前 tuple 元素展开追加到已累积元素后，最后一层 `Res(args...)` 收尾
- 两个坑：① 参数包后面的固定参数（`Tail&&..., index_sequence<Is...>`）会推导失败，**把包挪到最后**；② 引用元素让右值 `get` 不能用 `std::move`（`int&` 绑不了 `int&&`），要 `std::forward<T>`

### 10. 派生族函数与推导指引

- `make_tuple` 用 `unwrap_ref_decay_t`（自实现 `UnwrapRef`）：`std::ref(n)` → `int&` 元素
- `get<T>` 按类型取用 static_assert 卡"唯一出现"（比 SFINAE 报错清晰），`TupleTypeIndex` 递归找下标
- pair 构造**不会自动生成 implicit 推导指引**（其模板参数 U1/U2 推导不出类模板参数 T/Rest），要手写 `Tuple(const pair<U1,U2>&) -> Tuple<U1,U2>` 才能 `mytup::Tuple(pair)` CTAD

## EBO（Empty Base Optimization）—— 实现 tuple 的关键前置

### 问题

tuple 要同时存储多个不同类型的元素。当元素里有**空类型**（无数据成员，如 `struct empty {}`、无捕获 lambda）时，若按普通成员存储，每个空类型占 1 字节 + padding，tuple 会白白膨胀：

```cpp
struct Empty1 {};   // sizeof = 1
struct Empty2 {};   // sizeof = 1
// 朴素 tuple<Empty1, Empty2, int>: 1+pad+1+pad+4+pad = 16
```

### 机制

- **空类独立存在时最小 1 字节**（C++ 规定两个对象不能同址）
- **空类作为基类子对象时可以 0 字节**，与派生类第一个数据成员重叠——这就是 EBO

```cpp
struct AsMember { Empty1 e; int x; };      // 1+pad+4 = 8
struct AsBase : private Empty1 { int x; }; // 0+4    = 4（空基类压成 0）
```

### 在 tuple 中的两种实现路线

1. **继承式 EBO**（老版 libc++ `__compressed_pair_elem` 思路）：
   用 `is_class && is_empty && !is_final` 判断，能当空基类就继承、否则存成员。
   缺点：**函数指针/引用类型的元素不能作为基类**——smartPoint 里就栽在这（`void(*)(D*)` deleter 继承不了）。
2. **`[[no_unique_address]]` 成员式**（C++20 现代 libc++）：
   成员标属性即可，编译器自动做"空则重叠、非空则占位"，函数指针/引用/空类全兼容。**本实现采用这一路线。**

### smartPoint 踩坑回顾

- 继承式 EBO 只支持"类类型"deleter，函数指针 `void(*)(T*)` 和引用 `D&` 继承不了 → 相关 demo 被注释
- `delete` 不置空指针、`delete nullptr` 安全、move/forward 语义等，tuple 实现同样会遇到，见 `smartPoint/Uniqlo.md`

## 参考

- 官方实现分析（待做）：libc++ `tuple`（`_TupleLeaf` + `[[no_unique_address]]`）
- cppreference：https://en.cppreference.com/w/cpp/utility/tuple

