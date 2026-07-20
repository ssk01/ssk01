# RTTI 实现教程：从零到一

## 0. 先搞清楚：Runtime 你需要知道什么？

写代码的时候，你手里只有一个**基类指针/引用**。你不知道它指向的究竟是哪个子类对象。

```java
void process(Animal a) {
    // a 可能是 Dog、Cat、Bird……
    // 你怎么知道它到底是啥？
}
```

RTTI 做的事情就是：让你在**运行时**（不是编译时）判断"这个对象到底是什么类型"。

原生做法：
- Java: `a instanceof Dog`
- C++: `dynamic_cast<Dog*>(a)`

原生的问题：
- Java `instanceof` 性能不差（~6ns），但内存上每个对象已经自带 klass 指针了，没得省
- C++ `dynamic_cast` 很慢（~18ns），因为它要走 vtable → type_info → strcmp 整条链
- 而且你不能自己控制类型 ID——无法跨语言序列化，类型标识绑死在编译器上

**我们要做的：自己给每个类型分配一个整数 ID，存到对象里。然后类型判断 = 整数比较。**

---

## 1. 核心思想：Preorder 区间编号

给你一棵类型树，做一次 DFS 遍历，按访问顺序编号：

```
          Entity[1, 5]
         /            \
   Player[2,2]    Enemy[3,5]
                  /        \
            Orc[4,4]    Dragon[5,5]
```

**关键性质**：一个类型的所有子类型的 ID，全部落在 `[自己的ID, 子树最大ID]` 区间内。

所以 `isAssignableFrom` 的判断变成了：
```
问：Orc 是不是 Enemy 的子类？
答：Orc.id (4) 在不在 Enemy 的区间 [3, 5] 里？在！→ true

问：Player 是不是 Enemy 的子类？
答：Player.id (2) 在不在 Enemy 的区间 [3, 5] 里？不在！→ false
```

**不需要遍历继承链。不需要查 hash 表。两条 int 比较指令。**

---

## 2. 三个核心组件

### 2.1 TypeInfo：类型描述符

就是一个数据对象，描述一个类型。类比 Java 的 `Class<?>` 对象，但因为是我们自己造的，更轻量。

```
TypeInfo {
    id:    int      // 这个类型自己的编号
    high:  int      // 子树中最大的编号（决定了区间右端点）
    name:  String   // 人类可读的名字
    parent: TypeInfo // 父类型（null 就是根）
}
```

最核心的方法：
```java
boolean isAssignableFrom(int otherId) {
    return otherId >= this.id && otherId <= this.high;
}
```

### 2.2 TypeRegistry：类型注册表

负责三件事：
1. **注册类型**：`define(name, parent)` 把新类型登记进去（此时 id 还是 0）
2. **封存 + 编号**：`seal()` 做一次 DFS，给所有类型分配 ID 和区间
3. **查询**：`lookup(id)` / `lookup(name)` 查类型信息

`seal()` 的核心逻辑——DFS 递归：

```
dfsAssign(node):
    node.id = nextId++        // 占领当前编号
    node.high = node.id       // 初始区间 = [自己的id, 自己]

    for each child (按名字排序):
        childHigh = dfsAssign(child)   // 递归处理子树
        node.high = childHigh          // 把自己的 high 推到子树最右端

    return node.high
```

### 2.3 RttiObject：带类型标签的对象

每一层继承链上的类，在构造时把自己的 `TypeInfo.id` 存到一个 `int` 字段里。

```
class RttiObject {
    int __rttiTypeId;    // ← 每对象 4 字节

    boolean isInstance(TypeInfo type) {
        return type.isAssignableFrom(this.__rttiTypeId);
    }
}
```

因为子类构造器会覆盖父类写的值，最终存的一定是**最具体类型**的 id：

```
new Orc():
  RttiObject() → __rttiTypeId = 0
  Entity()     → __rttiTypeId = 1 (Entity.id)
  Enemy()      → __rttiTypeId = 3 (Enemy.id)
  Orc()        → __rttiTypeId = 4 (Orc.id)   ← 最终值
```

---

## 3. 使用流程（从用户视角）

```
// 1. 启动时注册所有类型（一次性）
reg.define("Entity", null)
reg.define("Player", entity)
reg.define("Enemy",  entity)
reg.define("Orc",    enemy)
reg.seal()   // ← 执行 DFS，分配 id

// 2. 每个业务类构造时注入 typeId
class Orc extends Enemy {
    Orc() { super(Orc.TypeInfo); }  // id=4
}

// 3. 运行时直接用
void handle(Entity e) {
    if (e.isInstance(enemyType)) { ... }   // 两次 int 比较
    Orc o = e.cast(Orc.class);
}
```

---

## 4. Java 版本的具体实现

### TypeInfo.java

```java
public final class TypeInfo {
    private int id;          // 非 final——seal() 时通过 setter 赋值
    private int high;
    private final String name;
    private final TypeInfo parent;
    private final Class<?> javaClass;   // 可选，用于 cast()

    // 核心判断方法
    public boolean isAssignableFrom(int typeId) {
        return typeId >= id && typeId <= high;
    }
}
```

之所以 `id`/`high` 不是 `final`：`define()` 时先创建 (id=0, high=0)，然后 `seal()` 时才 DFS 赋值。如果用 `final`，Java 17+ 的反射修改会失败。

### TypeRegistry.java

```java
public final class TypeRegistry {
    // 数据
    Map<String, TypeInfo> byName;    // 按名查找
    Map<Class<?>, TypeInfo> byClass; // 按 Java Class 查找
    TypeInfo[] byId;                 // 按 ID 查找（O(1) 数组索引）

    // seal() — DFS 分配 ID
    public void seal() {
        List<TypeInfo> roots = 所有 parent==null 的类型;
        AtomicInteger counter = new AtomicInteger(1);  // ID 从 1 开始
        for (root : roots) dfsAssign(root, ordered, counter);
        // 填入 byId 数组
    }

    int dfsAssign(TypeInfo node, counter) {
        int id = counter.getAndIncrement();
        node.setId(id);
        node.setHigh(id);

        // 找所有直属子类型，按名排序保证确定性
        List<TypeInfo> children = 所有 parent==node 的类型（按名排）;
        for (child : children) {
            int childHigh = dfsAssign(child, counter);
            node.setHigh(childHigh);  // 向右扩张区间
        }
        return node.high();
    }
}
```

### RttiObject.java

```java
public abstract class RttiObject {
    protected final int __rttiTypeId;   // 每对象 4 字节

    protected RttiObject(TypeInfo type) { this.__rttiTypeId = type.id(); }

    public boolean isInstance(TypeInfo type) {
        return type.isAssignableFrom(__rttiTypeId);
    }

    public <T> T cast(Class<T> cls) {
        TypeInfo target = registry.lookup(cls);
        if (!isInstance(target)) throw new ClassCastException(...);
        return (T) this;  // 类型安全已由区间判断保证
    }
}
```

---

## 5. C++ 版本的具体实现

C++ 版的思想完全一样，区别在于语法和内存管理。

### type_info.h

```cpp
struct TypeInfo {
    int id = 0;
    int high = 0;
    std::string name;
    const TypeInfo* parent = nullptr;

    bool is_assignable_from(int type_id) const noexcept {
        return type_id >= id && type_id <= high;
    }
};
```

`id` 和 `high` 是普通字段，`seal()` 时直接赋值即可。没有 Java 的 `final` 限制。

### type_registry.cpp — DFS 分配

```cpp
int TypeRegistry::dfs_assign(TypeInfo* node, int& next_id) {
    int id = next_id++;
    int high = id;
    node->id = id;
    node->high = high;

    // 收集子节点，按名字排序保证确定性
    std::vector<TypeInfo*> children;
    for (auto& [name, info] : by_name_) {
        if (info->parent == node) children.push_back(info);
    }
    std::sort(children.begin(), children.end(),
              [](auto* a, auto* b) { return a->name < b->name; });

    for (auto* child : children) {
        int child_high = dfs_assign(child, next_id);
        high = child_high;
        node->high = high;
    }
    return high;
}
```

### rtti_object.h — 基类 + 宏

```cpp
class Object {
protected:
    int rtti_type_id_ = 0;   // 每对象 4 字节
public:
    Object() {}
    explicit Object(const TypeInfo* type) : rtti_type_id_(type->id) {}

    bool is_instance(const TypeInfo& type) const noexcept {
        return type.is_assignable_from(rtti_type_id_);
    }

    template<typename T>
    T* as() {                              // RTTI 安全转型
        const TypeInfo& target = T::rtti_type_info();
        if (!is_instance(target)) throw std::bad_cast();
        return static_cast<T*>(this);
    }
};

// 业务类中只需写这一个宏
#define RTTI_ENABLE(ClassName)                              \
    static const TypeInfo& rtti_type_info() {               \
        static const TypeInfo* ti =                         \
            type_registry().lookup(#ClassName);             \
        return *ti;                                         \
    }                                                       \
    ClassName() { rtti_type_id_ = rtti_type_info().id; }
```

---

## 6. Java vs C++ 实现差异总结

| 方面 | Java | C++ |
|---|---|---|
| TypeInfo 可变性 | setter（因 final 字段反射受限） | 直接赋值 |
| 注册表隔离 | 构造函数 public，可 new 独立实例 | 同上 |
| 类型安全转型 | `cast(Class<T> cls)` 泛型 | `as<T>()` 模板 |
| 子类注入 typeId | 手动 `super(type)` | `RTTI_ENABLE` 宏自动注入 |
| 查找 typeInfo | 存储在 `byId[id]` 数组 | 同上 |
| 构造器链 | `super(TypeInfo)` 逐层传递 | 每层构造器体内覆盖 `rtti_type_id_` |

---

## 7. 适用场景 vs 不适用场景

**适合：**
- 游戏引擎 ECS（成千上万个 Entity，频繁判断类型）
- 自定义序列化（需要跨语言的类型标识）
- 嵌入式环境（禁用了原生 RTTI 的 C++ 项目）
- 事件系统 / 命令模式（switch 太重，需要 dispatch）

**不适合：**
- 需要 `instanceof` 检查接口类型（本方案只支持单继承树）
- 运行时有大量动态加载新类型（预分配区间不够灵活）
- 简单项目——Java 原生 `instanceof` 已经够快

---

## 8. Benchmark 结论

| | C++ (ns/check) | Java (ns/check) |
|---|---|---|
| RTTI interval | 3.56 | 3.58 |
| 原生 (dynamic_cast / instanceof) | 17.70 | 6.66 |

- C++：RTTI interval 比 `dynamic_cast` 快 ~5x
- Java：比 `instanceof` 快 ~2x（但 Java 的 instanceof 本身也不慢，主要优势是可控的 ID 体系）
