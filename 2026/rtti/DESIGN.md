# Java RTTI System: Minimal Per-Object Overhead

---

## 1. What is RTTI?

**RTTI (Run-Time Type Information)** is the mechanism that lets a program discover the actual type of an object **at runtime** — not just the declared type used at compile time.

你写了一个方法接受 `Animal` 参数，但实际传入的可能是 `Dog`、`Cat`、`Bird`。RTTI 让你在运行时知道它到底是什么，并据此做不同的事。

Java 内置的 RTTI 表现为三个东西：
- `instanceof` — 运行时类型判断
- `getClass()` — 获取 Class 对象
- 反射 (`Class.forName`, `getDeclaredMethods`, etc.) — 所有围绕 Class 对象的元数据访问

```java
void handle(Animal a) {
    if (a instanceof Dog)   ((Dog) a).bark();   // 运行时才知道是不是 Dog
    if (a instanceof Cat)   ((Cat) a).meow();
}
```

## 2. What is it used for?

| 场景 | 说明 |
|---|---|
| **序列化/反序列化** | Jackson、Protobuf 等框架需要知道字段类型才能读写 |
| **游戏引擎 ECS** | 遍历 Entity 列表时判断哪些是 Player、哪些是 Enemy |
| **事件系统** | 不同 Event 子类分发给不同的 Handler |
| **依赖注入** | Spring 扫描注解、注入正确的 Bean |
| **对象池/内存管理** | 回收对象时需要知道类型以调用正确的 cleanup |
| **命令模式/访问者模式** | 处理异构对象集合时的分派 |
| **ORM** | Hibernate 需要知道实体类型来做表映射 |

一句话概括：**当你有一个指向基类的引用，却需要根据实际子类型做不同处理时，就需要 RTTI。**

## 3. How is it implemented (broad categories)?

### 3a. 语言内置方式（C++/Java/Go）

**Java/JVM**: 每个对象的对象头（object header）中有一个 **klass pointer**（压缩后 4 字节，未压缩 8 字节），指向方法区的 Class 元数据。`instanceof` 实际是遍历 klass 中的继承链（或 itable/vtable）做比较。

```
Object header
┌───────┬──────────────┐
│ Mark  │ klass ptr    │  ← 指向 Class<Dog>
│ Word  │ (compressed) │
└───────┴──────────────┘
```

**C++ (vtable 方式)**：每个多态对象头部有一个 **vptr**（8 字节），指向虚函数表。`dynamic_cast` 通过 vtable 中的 `type_info` 指针做跨层次转换。没有虚函数的类没有 RTTI（除非编译时开 `/GR`）。

**Go**: 每个接口值是一个 `(type, data)` 对，其中 type 指针指向 runtime 的类型描述符。类型断言通过比较 type 指针或遍历 method set 实现。

### 3b. 自定义方式（手动打的 Tag）

很多引擎/框架自己实现 RTTI，而不是依赖语言内置机制，原因有三：
1. **控制开销**：不想在每个对象上存 8 字节指针
2. **跨语言/序列化**：需要自定义类型标识，不绑死语言
3. **避免 RTTI 被禁用**：某些环境（嵌入式、-fno-rtti）不能用内置 RTTI

常见做法：

| 方式 | 每对象开销 | instanceof 开销 | 说明 |
|---|---|---|---|
| **enum/int tag** | 4 字节 | O(1) 但只支持 flat 层次 | 类型之间没有父子关系，switch 分派 |
| **字符串 name** | 引用+字面量 | String.equals O(n) | 慢但调试友好 |
| **BitSet 掩码** | 4 字节 + 全局掩码 | 1 次 bit test | 需要知道最大类型数 |
| **Preorder 区间** | 4 字节 | 2 次 int 比较 | 本方案 |

---

## 4. This Design: Preorder-Interval Type IDs

### 核心思想

通过一次 DFS 预序遍历类型树，给每个类型分配一个整数 ID，**所有子类型的 ID 都落在父类型的 `[low, high]` 区间内**。

于是 `instanceof` 退化成了两道 int 比较：`tag >= type.low && tag <= type.high`

```
          Entity [1, 6]
         /         \
   Player[2,3]    Enemy[4,6]
                  /        \
            Orc[5,5]    Dragon[6,6]
```

- `Entity.isAssignableFrom(Orc)` → `5 ∈ [1, 6]` ✓
- `Player.isAssignableFrom(Dragon)` → `6 ∈ [2, 3]` ✗

每对象：4 字节的 `int` 类型标签。无间接寻址、无 hash 查找、无 bitmask。

### 架构

```
┌──────────────────────────────────────────────────────┐
│ TypeRegistry (static, thread-safe)                    │
│  - assignIds(rootTypes) → preorder DFS                │
│  - register(type) → TypeInfo                          │
│  - lookup(id) / lookup(name) / lookup(class)          │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│ TypeInfo                                              │
│  - id: int         (类型自身的 ID = low)               │
│  - high: int       (子树中最大子类型 ID)               │
│  - name: String                                       │
│  - parent: TypeInfo (nullable)                        │
│  - javaClass: Class<?> (nullable)                     │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│ RttiObject (abstract base class)                      │
│  - int __rttiTypeId   ← 4 bytes per object            │
│                                                       │
│  - isInstance(TypeInfo t):  t.id <= id && id <= t.high│
│  - typeOf():                TypeRegistry.lookup(id)   │
│  - <T> cast(Class<T>):      check + unchecked cast     │
└──────────────────────────────────────────────────────┘
```

### 类型检查 (type-safe cast)

```java
public <T extends RttiObject> T cast(Class<T> cls) {
    TypeInfo target = TypeRegistry.lookup(cls);
    if (!TypeRegistry.isAssignableFrom(target, this.__rttiTypeId)) {
        throw new ClassCastException(typeOf().name() + " cannot be cast to " + target.name());
    }
    @SuppressWarnings("unchecked")
    T result = (T) this;
    return result;
}
```

### 动态注册新类型

初始编号完成后，如果父类型在注册时预留了区间，新子类型才能就地插入。

| 策略 | 方式 | 适用场景 |
|---|---|---|
| **预留区间** | `register(parent, "FutureType", reserveSize=10)` 让父类型区间多留 10 个空位 | 已知会扩展的类型族 |
| **溢出映射** | 对超出区间的类型，维护 `Map<Integer, Set<Integer>>` 做祖先关系查询 | 完全不可预料的动态加载 |

预留区间意味着 high 值拉大，`isAssignableFrom` 仍然 O(1)。溢出映射回到 map 查找，退化到 O(1) hash。

### 性能特征

| 操作 | 成本 | 说明 |
|---|---|---|
| `instanceof` 检查 | 2 次 int 比较 | 理想状况下 ~1-2 CPU cycle |
| `typeOf()` 查询 | 1 次数组索引 | 按需使用，不在热路径上 |
| 注册 | 一次性 DFS | 启动时完成 |

### 与 Java 内置 instanceof 的对比

| | Java instanceof | 本方案 |
|---|---|---|
| 每对象额外内存 | 0（对象头已有 klass ptr） | 4 字节 int |
| interesting 开销 | 2-3 次内存间接访问（klass ptr → class → superclass chain） | 2 次寄存器比较 |
| 支持接口/多继承 | 是（迭代 itable） | 否（严格单继承树） |
| 动态加载新类 | 天然支持 | 需预留区间 |
| 关闭编译器优化 | 不能 | 可以完全内联 |

### 要实现的组件

1. `TypeInfo` — 不可变类型描述符，包含区间
2. `TypeRegistry` — 全局注册表，DFS 区间分配，查询 API
3. `RttiObject` — 抽象基类（带 `int __rttiTypeId`）
4. `Rtti` — 静态工具类：`isInstance`、`cast`、`typeOf`
5. 基准测试 — 对照原生 `instanceof` 的热循环性能

### 待确认

1. **基类 vs 接口**？基类共享 `int` 字段定义，省内存；接口更灵活不占继承位。倾向基类（`RttiObject`）。
2. **接口/多继承**：严格单继承树，还是用溢出映射回退？设计上先做单继承。
3. **预留区间 API**：是否需要？是——`TypeRegistry.reserve(parent, count)`。
