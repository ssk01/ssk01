# Optional 手动实现与原理

## 是什么

`std::optional<T>` 表示"**可能有一个 T，也可能没有**"。它把"有没有值"从调用约定（比如返回 `-1`、`nullptr`、`"not found"`）变成类型层面的事实：

```cpp
Optional<int> find(...);   // 一眼看出可能没有值
if (opt) { ... }           // 先判断，再放心取
```

## 核心原理：空间与生命周期分离

朴素想法：

```cpp
template <typename T>
class Optional {
    bool has_value_;
    T value_;   // 问题：为空也要默认构造一个 T
};
```

问题 1：要求 `T` 可默认构造（很多类型不行）；问题 2：为空时白构造/白析构一个 `T`。

**正解**：不给 `T` 分配"活的"对象，只预留空间，有值时才在空间里构造 `T`。官方 libstdc++ 和本 demo 都这么干：

```cpp
template <typename T>
class Optional {
    union Storage {
        T value;          // 占 sizeof(T)，union 自动保证对齐
        Storage() {}      // 不构造 value
        ~Storage() {}     // 不析构 value
    };
    Storage storage_;     // 只留空间
    bool has_value_ = false;  // 有没有值的标志
};
```

`union` 的好处：空间由 `T value` 天然对齐（不用手写 `alignas`），且语义上"要么是值要么是空的"。

## 三种关键操作

| 操作 | 手法 | 说明 |
|------|------|------|
| 放值 | `new (&storage_.value) T(v)` | **placement new**：在已有空间上构造，不分配新内存 |
| 取值 | `&storage_.value` | 直接取 union 里的成员 |
| 清空 | `(&storage_.value)->~T()` | **显式调用析构**，再 `has_value_ = false` |

```cpp
Optional(const T& v) : has_value_(true) { new (get()) T(v); }   // 放值

void reset() {
    if (has_value_) { get()->~T(); has_value_ = false; }        // 清空
}

T& value() {
    if (!has_value_) throw bad_optional_access();               // 空则抛异常
    return *get();
}
```

## 与官方 `std::optional` 的对比

### 内存布局：完全一样

实测（本机，与 libstdc++ 一致）：

| 类型 | sizeof(T) | sizeof(optional<T>) |
|------|-----------|---------------------|
| char | 1 | 2 |
| int | 4 | 8 |
| double | 8 | 16 |
| string | 24 | 32 |
| 本 demo 的 Optional<Logged> | 4 | 8 |

规律：`sizeof(optional<T>) = sizeof(T) + 1 字节 bool，对齐到 alignof(T)`。官方源码核心也是 `union 存储 + bool _M_engaged`，**没有更省**。这个 bool 开销是 optional 的**固有成本**——必须记录"有没有值"，标准明确接受 `sizeof(optional<T>) > sizeof(T)`。

### 真正的差别（不在内存）

| 能力 | 官方 | 本 demo |
|------|------|---------|
| constexpr 编译期使用 | ✅ | ❌ |
| 异常安全：拷贝赋值时若 T 拷贝抛异常，保持原状态 | ✅（按 `is_nothrow_copy_assignable` 分两条路） | ❌（reset+重建） |
| `emplace` / `std::in_place_t` / 完美转发 | ✅ | ❌ |
| 移动语义、比较运算符、C++23 monadic (`transform`/`or_else`) | ✅ | ❌ |
| T 平凡时 optional 也平凡（性能点） | ✅ | ❌ |

## Demo 运行输出

`g++ -std=c++17 Optional_demo.cpp -o build/optional_demo && ./build/optional_demo`

```
== 1. default construct: empty, and T is NOT constructed ==
    (no Logged ctor log below = T never constructed)
    o1.has_value() = 0

== 2. construct with value (placement new) ==
    Logged(42) constructed, alive=1
    Logged(42) copy-constructed, alive=2
    Logged(42) destroyed, alive=1
    o2.has_value() = 1
    o2.value().id = 42

== 3. operator bool ==
    o2 is truthy
    o1 is falsy

== 4. value() on empty throws ==
    caught: bad_optional_access

== 5. value_or ==
    a.value_or(99) = 99
    b.value_or(99) = 7

== 6. reset: explicit ~T, then flag cleared ==
    o2 has value, calling reset():
    Logged(42) destroyed, alive=0
    o2.has_value() = 0

== 7. copy construct ==
    Logged(100) constructed, alive=1
    Logged(100) copy-constructed, alive=2
    Logged(100) destroyed, alive=1
    Logged(100) copy-constructed, alive=2
    o4.value().id = 100

== 8. copy assign (reset + rebuild) ==
    Logged(100) copy-constructed, alive=3
    o5.value().id = 100

== 9. sizeof: union storage + 1 bool ==
    sizeof(Logged)           = 4
    sizeof(Optional<Logged>)  = 8
    (extra bytes = the 'has value' bool, same as std::optional)

== 10. use with std::string ==
    empty name -> (none)
    name -> C++17

== main ends: o5/o4/o3 destroyed, alive back to 0 ==
    Logged(100) destroyed, alive=2
    Logged(100) destroyed, alive=1
    Logged(100) destroyed, alive=0
```

### 输出解读

- **第 1 节**：默认构造空 optional，`Logged` 的构造函数**一次都没被调用**——空间与生命周期分离的直接证据。
- **第 2 节**：`o2(42)` 里 `42` 先构造临时 `Logged(42)`，再 placement-new **拷贝**进 storage，临时对象随后析构（alive 1→2→1）。这就是官方引入 `emplace` 的原因——少一次拷贝。
- **第 6 节**：`reset()` 显式调用 `~Logged()`，alive 回到 0。
- **第 9 节**：`sizeof(Optional<Logged>) = 8`，比 `Logged` 的 4 多 1 字节 bool + 填充，与官方 `std::optional<int>` 一致。
- **结尾**：三个持值的 optional 依次析构，`alive` 1→2→3→0，证明每个值的构造与析构严格配对、没有泄漏。

## 编译运行

```bash
g++ -std=c++17 Optional_demo.cpp -o build/optional_demo
./build/optional_demo
```
