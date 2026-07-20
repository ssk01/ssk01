# C++ 原生 RTTI 实现原理 (vptr / vtable / type_info / dynamic_cast)

## 谁有 RTTI

C++ 只有**多态类型**才有 RTTI——即至少有一个虚函数的类。没有虚函数 → 没有 vptr → 没有 type_info → `dynamic_cast` 不能用（编译报错）。

`-fno-rtti` 关闭后所有类退化为无 RTTI，`typeid` 和 `dynamic_cast` 全部失效。

## 对象内存布局

```
         Orc 对象 (栈/堆)                    vtable_Orc (只读数据段)
        ┌──────────┬──────────┐             ┌─────────────────────────────┐
        │  vptr    │  data    │             │ [0] &type_info_Orc          │  ← RTTI 入口
        │  8 bytes │          │      ┌───→  │ [1] &Orc::speak()           │
        └────┬─────┴──────────┘      │      │ [2] &Orc::attack()          │
             │                       │      └─────────────────────────────┘
             └───────────────────────┘
```

- **vptr**：每个多态对象的第一个字段（8 字节，编译期插入，对程序员不可见）
- **vtable 首个槽**：永远指向 `type_info`（这是 ABI 规范）
- **虚函数** 从 vtable[1] 开始按声明顺序排列

sizeof(Orc) = 指针大小 + 成员数据。一个空的虚析构函数就值 8 字节。

## type_info 结构

```cpp
// 实际 Itanium C++ ABI 定义 (简化)
struct __class_type_info : std::type_info {
    // base_list 存储所有直接/间接基类
    // dynamic_cast 沿此递归查找
};
```

关键方法：`__do_dynamic_cast`——沿 `base_list` 递归遍历，对比目标 `type_info`：

```
Orc.type_info → base_list[0] → Enemy.type_info → base_list[0] → Entity.type_info
```

## dynamic_cast 的完整执行路径

```cpp
dynamic_cast<Orc*>(entity_ptr)
```

汇编级别（简化）：

```
1. 读 entity_ptr 的 vptr:
      mov rax, [entity_ptr]         // rax = vtable 地址

2. 读 vtable[0]:
      mov rdi, [rax]                // rdi = type_info 地址

3. 读目标 type_info:
      lea rsi, [type_info_Orc]

4. __dynamic_cast(rdi, rsi, ...):
      → 沿 type_info 的 base_list 递归遍历
      → 如果 rdi 的基类链中包含 rsi → 返回偏移后的指针
      → 否则 → 返回 nullptr
```

第 4 步是 `libstdc++` / `libc++abi` 的运行时函数，不在编译期内联——所以任何 `dynamic_cast` 都是一次函数调用 + 多次指针追踪。

## 为什么 dynamic_cast 慢

| 操作 | 成本 |
|---|---|
| 读 vptr（解引用） | 1 次 load |
| 读 vtable[0]（再解引用） | 1 次 load |
| 调用 `__dynamic_cast` | 函数调用开销 |
| 遍历 base_list（再解引用） | n 次 load + strcmp |
| 偏移计算 + 返回 | O(1) |

与区间方案的两次寄存器比较相比，多出 3+ 次内存访问和一次函数调用。

## 运行时 typeid 也是同一套

```cpp
typeid(*ptr).name()
```

流程完全一样：`ptr → vptr → vtable[0] → type_info → name()`。没有虚函数的类 `typeid` 在编译期求值，不走 vptr。

## 对应代码

`cpp_rtti_demo.cpp` 手动模拟了这个过程，删掉编译器魔法后核心不足 100 行：

```
TypeInfo (name + base_list)     ← 类型元数据
VTable   (rtti 指针)            ← 虚表首个槽
Object   (vptr)                 ← 多态对象
dyn_cast (读 vptr → type_info → 沿 base_list 查找)
```

与真正的 `dynamic_cast` 行为一致（继承链判断、失败返回 nullptr）。
