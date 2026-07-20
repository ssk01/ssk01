# JVM RTTI 实现原理 (HotSpot 模拟)

## 对象头布局

每个 Java 对象在堆上的结构：

```
┌──────────────────┬──────────────────┬─────────────┐
│    Mark Word     │    klass ptr     │  instance   │
│   (8 字节)       │  (压缩后 4 字节)  │    data     │
└──────────────────┴──────────────────┴─────────────┘
```

- **Mark Word**：GC 标记、锁状态、hashCode 等
- **klass ptr**：指向方法区的 `InstanceKlass` 元数据对象

开启压缩指针 (`-XX:+UseCompressedOops`) 后 klass ptr 是 4 字节。这个指针就是 JVM RTTI 的全部——一切类型判断都从它开始。

## instanceof 的完整链路

```java
obj instanceof Enemy
```

对应 JVM 字节码 `instanceof`，在 HotSpot 中的执行路径：

```
obj.klass                              // 解对象头, 取 klass ptr
  → InstanceKlass::is_subtype_of(target_klass)
      → 1) walk _super 链              // 主继承: O(depth)
      → 2) check _secondary_super_cache // 热路径缓存: O(1)
      → 3) scan _secondary_supers[]     // 接口查找: O(#interfaces)
```

## 三个核心数据结构

### 1. `_super` 链 — 主继承

每个 `Klass` 指向它的直接父类。`is_subtype_of` 沿这条链上行比较指针。

```
Orc._super → Enemy._super → Entity._super → nullptr
```

判断 `obj instanceof Enemy`：从 `Orc.klass` 出发上一格就命中了，O(1)。最坏情况 O(继承深度)，通常 < 10。

### 2. `_secondary_supers[]` — 接口查找

一个 `Klass` 可以实现多个接口。这些接口不在 `_super` 链上（Java 是单继承），而是存为一个数组。

```
Orc._secondary_supers = [ICollidable]
```

当判断 `obj instanceof ICollidable` 时，`_super` 链走到底都找不到，于是扫描 `_secondary_supers` 数组。

### 3. `_secondary_super_cache` — 热路径优化

每次扫描 `_secondary_supers` 成功匹配后，把目标类型写入一个缓存字段。下次同一个 `instanceof` 检查可以先看缓存，命中率极高。

```cpp
// 第一次: 扫描数组 → 找到 → 写缓存
// 第二次: 读缓存 → O(1) 命中
if (_secondary_super_cache == target) return true;
```

## 对比 Preorder 区间方案

| | JVM (klass ptr) | Preorder 区间 |
|---|---|---|
| 数据结构 | 对象 → pointer → Klass 链 | 对象 → int tag |
| instanceof | 遍历指针链 | 两次 int 比较 |
| 内存访问 | 多次间接寻址 (cache miss) | 寄存器操作 |
| 每对象开销 | 8B (未压缩) / 4B (压缩) | 4B |
| 动态加类 | 天然支持 | 需预留/溢出 |
| 接口 | 天然支持 | 需额外处理 |

## 为什么 JVM 选了指针链？

JVM 的设计目标是**无限制的动态类加载**。你随时可以 `ClassLoader.defineClass()` 往继承树上插入一个新类。如果用区间方案，插入可能触发全局 IDF 重分配（代价 O(加载过的所有类)），这对 JVM 不可接受。

HotSpot 的折中是：继承链很短（< 10 层是常态），遍历它比维护区间更便宜，还天然支持任意动态插入。

## 为什么我们又做了区间方案？

因为很多场景不需要动态类加载——游戏引擎的 ECS、嵌入式系统、序列化框架。在这些场景中，类型系统在编译时就已经封死了。放弃动态能力换 O(1) 两次比较是划算的。

## 对应代码

`jvm_rtti_demo.cpp` 中 `Klass::is_subtype_of()` 直接对应 HotSpot 的 `InstanceKlass::is_subtype_of()`：

```cpp
bool Klass::is_subtype_of(Klass* target) const {
    // 1) _super 链
    for (const Klass* k = this; k != nullptr; k = k->_super)
        if (k == target) return true;

    // 2) 缓存命中
    if (_secondary_super_cache == target) return true;

    // 3) 扫描接口列表
    for (auto* iface : _secondary_supers)
        if (iface == target) {
            _secondary_super_cache = target;  // 写入热路径
            return true;
        }

    return false;
}
```
