# DuckDB C++ 现代特性实战手册

每个特性附真实代码位置和用法。

---

## 1. `unique_ptr` / `make_uniq` — 独占所有权

```cpp
// 文件: src/execution/operator/persistent/physical_insert.cpp:119
unique_ptr<GlobalSinkState> PhysicalInsert::GetGlobalSinkState(ClientContext &context) const {
    auto result = make_uniq<InsertGlobalState>(context, GetTypes(), *table);
    return std::move(result);
}
```

`make_uniq<T>(args...)` 是 DuckDB 对 `std::make_unique` 的包装。操作符的 Sink 状态（每线程一个 LocalState，全局一个 GlobalState）全部用 `unique_ptr` 管理——操作符创建它，独享所有权，析构时自动释放。

## 2. `shared_ptr` / `weak_ptr` — 共享所有权与无锁探测

```cpp
// 文件: src/include/duckdb/storage/buffer/block_handle.hpp:251,289-294,314
class BlockHandle : public enable_shared_from_this<BlockHandle> {
public:
    weak_ptr<BlockMemory> GetMemoryWeak() const {
        return weak_ptr<BlockMemory>(memory_p);
    }
private:
    const shared_ptr<BlockMemory> memory_p;    // 底层内存, 多个 ColumnSegment 共享
    BlockMemory &memory;                        // 非持有引用, 快速访问
};

// 文件: src/include/duckdb/storage/table/column_segment.hpp:169
class ColumnSegment : public SegmentBase<ColumnSegment> {
    shared_ptr<BlockHandle> block;             // 多个 Segment 共享同一 Block
};
```

一个 256KB DataBlock 可能被多个 ColumnSegment 共同引用（PartialBlockManager 分配）。`shared_ptr` 保证最后一个 Segment 释放后才释放 Block。`weak_ptr` 用于 eviction queue 无锁探测：`weak_ptr::lock()` 失败说明 Block 已析构，队列节点就是 dead node，直接丢弃。

```cpp
// 文件: src/storage/buffer/buffer_pool.cpp:47-59
bool BufferEvictionNode::IsDeadNode() {
    auto shared_memory_p = memory_p.lock();    // weak_ptr → shared_ptr
    if (!shared_memory_p) return true;         // Block 已析构
    if (handle_sequence_number != shared_memory_p->GetEvictionSequenceNumber())
        return true;                           // Block 重新加载过, 此节点过期
    return false;
}
```

## 3. `optional_ptr<T>` — 可空的非持有引用

```cpp
// 文件: src/include/duckdb/common/optional_ptr.hpp:17-82
template <class T, bool SAFE = true>
class optional_ptr {
public:
    optional_ptr() noexcept : ptr(nullptr) {}
    optional_ptr(T *ptr_p) : ptr(ptr_p) {}          // 隐式从裸指针
    optional_ptr(T &ref) : ptr(&ref) {}              // 隐式从引用
    optional_ptr(const unique_ptr<T> &p) : ptr(p.get()) {}
    optional_ptr(const shared_ptr<T> &p) : ptr(p.get()) {}

    operator bool() const { return ptr; }            // if (ptr) 判断
    T &operator*() { CheckValid(); return *ptr; }    // *ptr 解引用(DEBUG 检查非空)
    T *get() { return ptr; }
private:
    T *ptr;
};

// 用法: src/include/duckdb/storage/table/column_segment.hpp:102
void ConvertToPersistent(QueryContext context,
                         optional_ptr<BlockManager> block_manager,  // 可空——transient segment 传 nullptr
                         const block_id_t block_id);
```

**为什么不用裸指针 `T*`？** 一眼看不出参数能不能为空——`void Foo(T*)` 是可为空还是禁止为空？靠注释, 不可靠。

**为什么不用 `std::optional<std::reference_wrapper<T>>`？**

```cpp
// 方案 A: 裸指针 — 语义模糊
void Foo(T *ptr);          // 允许为空? 还是禁止为空? 猜不出

// 方案 B: std::optional<reference_wrapper<T>> — 太重
void Foo(std::optional<std::reference_wrapper<T>> ptr);  // 写起来要命, 内部至少 16 字节

// 方案 C: optional_ptr<T> — 明确语义 + 零额外开销
void Foo(optional_ptr<T> ptr);  // 签名自解释: "可以没有"
```

`optional_ptr` 的实际价值是**类型系统级别的合同**：`optional_ptr<T>` = 可空, `reference<T>` = 不可空。读者不看实现就知道约定:

## 4. `reference<T>` — 不可空的非持有引用

```cpp
// 文件: src/include/duckdb/common/helper.hpp:46-47
template<typename T>
using reference = std::reference_wrapper<T>;

// 用法: src/include/duckdb/storage/table/column_segment.hpp:171
class ColumnSegment {
    reference<const CompressionFunction> function;  // 不能为 null, 必须有效
};

// 用法: src/execution/operator/persistent/physical_insert.cpp:309
static void CheckDistinctnessInternal(ValidityMask &valid,
    vector<reference<Vector>> &sort_keys,  // 容器里存引用——原生引用不能放进 vector
    idx_t count, ...);
```

`optional_ptr` 管"可能没有"，`reference` 管"一定有"。

**为什么放 vector？** C++ 不允许 `vector<T&>`——引用不是对象, 不能存容器:

```cpp
vector<int&> v;  // 编译错误! 引用不是可存储对象
```

但有时确实需要"在容器里存一组对象的引用"：

```cpp
// physical_insert.cpp:309
static void CheckDistinctnessInternal(
    ValidityMask &valid,
    vector<reference<Vector>> &sort_keys,  // "一组 Vector 的引用", 不能是 vector<Vector*> (编码规范禁裸指针)
    idx_t count, ...);
```

`reference_wrapper` = 能放进容器的引用：

```cpp
Vector a, b, c;
vector<reference<Vector>> keys = {a, b, c};
keys[0].get().SomeMethod();  // 等价于 a.SomeMethod()
```

## 5. `atomic<T>` — 无锁并发读写

### atomic<SegmentNode<T>*> next — 无锁链表

```cpp
// 文件: src/include/duckdb/storage/table/segment_tree.hpp:29-34,66-68,79,480-487

// 读者 — 无需加锁
optional_ptr<SegmentNode<T>> Next() const {
    return next.load();               // atomic load, memory_order_seq_cst
}

// 写者 — 持有 SegmentLock 后写
void SetNext(optional_ptr<SegmentNode<T>> ptr) {
    this->next = ptr.get();           // atomic store, memory_order_seq_cst
}

// 声明
atomic<SegmentNode<T>*> next;         // 指针原子, 保证读到完整地址

// 追加新节点 (持有锁)
void AppendSegmentInternal(SegmentLock &l, shared_ptr<T> segment, ...) {
    auto node = make_uniq<SegmentNode<T>>(row_start, std::move(segment), nodes.size());
    if (!nodes.empty()) {
        nodes.back()->SetNext(*node);  // 旧尾 → 新节点
    }
    nodes.push_back(std::move(node));  // 同时存数组 (二分查找用)
}
```

**为什么读者无锁安全**：`next` 只在追加时写入, 写者持有 `SegmentLock`。写完新节点的 `next = nullptr` 后, 才把旧尾的 `next` 指向新节点。读者不管何时读, 要么看到 `nullptr`（链表尾）, 要么看到已完全初始化的节点。不存在"读到半成品"。

### atomic<idx_t> count — 行计数

```cpp
// 文件: src/storage/table/column_data.cpp:600-606 — 写者 (append 线程)
void ColumnData::AppendData(ColumnAppendState &state, ...) {
    while (true) {
        auto &segment = state.current->GetNode();
        idx_t copied = segment.Append(state, vdata, offset, append_count);
        this->count += copied;           // atomic fetch_add, 并发安全
        if (copied == append_count) break;
        ...
    }
}

// 文件: src/storage/table/row_group.cpp:1123 — 读者 (scan 线程)
idx_t row_group_start = this->count.load();  // 无锁读当前总行数
```

append 线程逐 chunk 累加, scan 线程随时读。不需要锁——`atomic<idx_t>` 保证完整读写, 不会撕裂。

### 文件: src/include/duckdb/storage/table/segment_base.hpp:23

```cpp
template <class T>
class SegmentBase {
    atomic<idx_t> count;              // 每个 segment 的行数, 并发安全
};
```

## 6. `SegmentLock` / `StorageLock` — RAII 锁

RAII **不在 `SegmentLock` 自己的代码里**, 在它持有的 `unique_lock<mutex>` 里:

```cpp
// 文件: src/include/duckdb/storage/table/segment_lock.hpp:15-39
struct SegmentLock {
    explicit SegmentLock(mutex &lock) : lock(lock) {}     // 构造 = 加锁 (unique_lock 构造即 lock)

    SegmentLock(const SegmentLock &) = delete;             // 禁止复制 (双解锁会 UB)
    SegmentLock(SegmentLock &&other) noexcept {            // 允许移动, 用于返回值
        std::swap(lock, other.lock);
    }

private:
    unique_lock<mutex> lock;           // ← RAII 在这里: ~unique_lock() 自动 unlock()
};

// 用法: src/include/duckdb/storage/table/segment_tree.hpp:117-119
SegmentLock Lock() const {
    return SegmentLock(node_lock);    // 返回已加锁对象, 移动语义传递
}

// 实际使用:
{
    auto l = tree.Lock();              // SegmentLock 构造 → unique_lock 构造 → mutex.lock()
    tree.AppendSegment(l, seg);        // 持锁期间操作
}                                      // l 析构 → unique_lock 析构 → mutex.unlock()
```

**析构链**: `SegmentLock::~SegmentLock()` (编译器生成) → `unique_lock<mutex>::~unique_lock()` → `mutex.unlock()`

**StorageLock — R/W 锁的 RAII 版**：

```cpp
// 文件: src/include/duckdb/storage/storage_lock.hpp:19-52
class StorageLockKey {                                  // RAII Key: 持有 = 锁定
public:
    StorageLockKey(shared_ptr<StorageLockInternals> internals, StorageLockType type);
    ~StorageLockKey();                                  // 析构 = 自动解锁 (实现在 storage_lock.cpp)
};

class StorageLock {
    unique_ptr<StorageLockKey> GetExclusiveLock();      // 返回 Key, 持有则独占
    unique_ptr<StorageLockKey> GetSharedLock();          // 返回 Key, 持有则共享
    unique_ptr<StorageLockKey> TryGetExclusiveLock();    // 失败返回 nullptr
};

// 调用者:
{
    auto key = storage_lock.GetSharedLock();             // 获取读锁
    // ... 读操作 ...
}  // key 析构 → ~StorageLockKey() → 释放锁
```

## 7. Template Policy: `SegmentTree<T, bool SUPPORTS_LAZY_LOADING>`

```cpp
// 文件: src/include/duckdb/storage/table/segment_tree.hpp:103-104,184-199
template <class T, bool SUPPORTS_LAZY_LOADING = false>
class SegmentTree {

    optional_ptr<SegmentNode<T>> GetNextSegment(SegmentNode<T> &node) const {
        if (!SUPPORTS_LAZY_LOADING) {        // 编译期常量分支!
            return node.Next();               // 快路径: 直接读 next 指针
        }
        if (finished_loading) {
            return node.Next();
        }
        auto l = Lock();                      // 慢路径: 加锁 + 惰性加载
        return GetNextSegment(l, node);
    }
};
```

当 `SUPPORTS_LAZY_LOADING = false`（Transient Segment 使用），编译器生成代码中不存在 `Lock()` 调用——整个分支被优化掉。这是编译期策略选择：恢复数据库时 Persistent SegmentTree 用 `true`，运行时追加的 Transient SegmentTree 用 `false`。同一套模板,编译期决定行为,零运行时开销。

## 8. `std::function` + Lambda: 控制反转

```cpp
// 文件: src/storage/table/column_data_checkpointer.cpp:92-115
void ColumnDataCheckpointer::ScanSegments(
    const std::function<void(Vector &)> &callback) {   // 接受回调

    for (auto &segment_node : col_data.data.SegmentNodes()) {
        auto &segment = segment_node.GetNode();
        segment.InitializeScan(scan_state);

        for (base_row = 0; base_row < segment.count; base_row += 2048) {
            col_data.CheckpointScan(segment, scan_state, count, scan_vector);
            callback(scan_vector);                      // 每个 Vector 回调一次
        }
    }
}

// 调用方——用 Lambda 同时分析所有候选压缩算法:
// 文件: 同文件 195-207
ScanSegments([&](Vector &scan_vector) {
    for (idx_t i = 0; i < checkpoint_states.size(); i++) {
        auto &states = analyze_states[i];
        for (idx_t j = 0; j < functions.size(); j++) {
            functions[j]->analyze(*states[j], scan_vector);  // 并行评估 RLE/ALP/ZSTD/...
        }
    }
});
```

`ScanSegments` 负责遍历逻辑（谁在迭代），Lambda 负责计算逻辑（对每个 Vector 做什么）。一次遍历,十几个压缩算法同时采样。

## 9. Variadic Templates — 序列化接口

```cpp
// 文件: src/include/duckdb/common/serializer/serializer.hpp:91-148
template <class T>
void WriteProperty(const field_id_t field_id, const char *tag, const T &value) {
    OnPropertyBegin(field_id, tag);
    WriteValue(value);                       // SFINAE 分发: int→写4字节, string→写长度+内容
    OnPropertyEnd();
}

template <class T>
void WritePropertyWithDefault(field_id_t id, const char *tag,
                              const T &value, const T &default_value) {
    if (value == default_value) return;      // 等于默认值 → 不写, 省空间
    OnOptionalPropertyBegin(id, tag, true);
    WriteValue(value);
    OnOptionalPropertyEnd(true);
}

// 实际调用 (DataPointer 序列化):
serializer.WriteProperty(100, "row_start",    row_start);
serializer.WriteProperty(101, "tuple_count",  tuple_count);
serializer.WriteProperty(102, "block_pointer", block_pointer);
serializer.WriteProperty(103, "compression_type", compression_type);
serializer.WriteProperty(104, "statistics",      statistics);
serializer.WriteProperty(105, "segment_state",   segment_state);
```

一个 `WriteProperty` 方法接受任何类型，编译期决定序列化方式。每个属性有 `field_id`（二进制格式）和 `tag`（文本格式），向前兼容新增字段。

## 10. CRTP: `SegmentBase<T>`

```cpp
// 文件: src/include/duckdb/storage/table/segment_base.hpp:16-24
template <class T>
class SegmentBase {
public:
    explicit SegmentBase(idx_t count) : count(count) {}
    atomic<idx_t> count;                      // 所有 segment 类型共享的成员
};

// 用法: src/include/duckdb/storage/table/column_segment.hpp:39
class ColumnSegment : public SegmentBase<ColumnSegment> {
    // 继承 atomic<idx_t> count —— 无需重复声明
};

// 未来如果加共享方法, 可以 static_cast<T*>(this) 调用子类方法:
// template<class T> void SegmentBase<T>::Compact() {
//     static_cast<T*>(this)->CompactImpl();   // 编译期多态, 零虚函数开销
// }
```

CRTP 是 DuckDB 唯一使用的"类继承"模式——没有虚函数（除少数接口类），继承只用来共享数据成员。

## 11. Lock-Free Queue — 缓冲区淘汰队列 (近似 LRU)

### 架构: 三级优先级队列

```cpp
// 文件: src/include/duckdb/storage/buffer/buffer_pool.hpp:115-122
static constexpr idx_t EVICTION_QUEUE_TYPES = 3;

// 队列 0: BLOCK (1 个) — 最先驱逐 (普通数据块, 便宜)
// 队列 1: MANAGED_BUFFER (6 个) — 中等优先级
// 队列 2: TINY_BUFFER (1 个) — 最后驱逐 (碎片, 收益低)
const array<idx_t, EVICTION_QUEUE_TYPES> eviction_queue_sizes = {1, 6, 1};
```

### 入队 (Block 被 unpin 时)

```cpp
// 文件: src/storage/buffer/buffer_pool.cpp:271-297
bool BufferPool::AddToEvictionQueue(BlockLock &lock, shared_ptr<BlockHandle> &handle) {
    auto &memory = handle->GetMemory();

    // ① 旧队列节点标记为 dead — 序列号不一致时自动过滤
    if (memory.HasLiveQueueEntry(lock)) {
        queue.IncrementDeadNodes();         // 旧节点变 dead
    }

    // ② 递增序列号 + 记录 LRU 时间戳
    auto ts = memory.NextEvictionSequenceNumber();  // eviction_seq_num += 1
    memory.SetLRUTimestamp(steady_clock::now());    // 记录本次访问时间
    memory.SetHasLiveQueueEntry(lock, true);         // 标记为活跃

    // ③ 创建新节点入队 — 存 weak_ptr, 不阻止 Block 析构
    BufferEvictionNode node(handle->GetMemoryWeak(), ts);
    return queue.AddToEvictionQueue(std::move(node));
}
```

### 出队 & 驱逐 (内存不够时)

```cpp
// 文件: src/storage/buffer/buffer_pool.cpp:464-509
template <typename FN>
void EvictionQueue::IterateUnloadableBlocks(FN fn) {
    for (;;) {
        BufferEvictionNode node;
        if (!q.try_dequeue(node)) {        // 无锁出队
            if (!TryDequeueWithLock(node)) return;  // 最后一次尝试, 加锁
        }

        auto handle = node.memory_p.lock(); // weak_ptr → shared_ptr

        if (!handle) {                      // ① Block 已被析构 → 死节点
            DecrementDeadNodes();
            continue;
        }

        auto lock = handle->GetLock();
        if (node.handle_sequence_number != handle->GetEvictionSequenceNumber()) {
            // ② 序列号不匹配 → 已入队过新版本 (此节点过期)
            DecrementDeadNodes();
            continue;
        }

        // ③ 确认是当前活跃节点
        handle->SetHasLiveQueueEntry(lock, false);
        if (!handle->CanUnload()) {         // ④ 仍被 pin 住 → 跳过, 让 unpin 时重新入队
            continue;
        }

        if (!fn(node, handle, lock)) {      // ⑤ 真正驱逐: unload block → 释放内存
            break;                           // fn 返回 false → 停止迭代
        }
    }
}

// 文件: src/storage/buffer/buffer_pool.cpp:391-402
EvictBlocks(context, tag, extra_memory, memory_limit) {
    for (auto &queue : queues) {             // 按优先级 0→1→2 依次尝试
        if (EvictBlocksInternal(queue).success) return;
    }
    EvictObjectCacheEntries();               // 还不够 → 驱逐 ObjectCache (metadata/config)
}
```

### 基于年龄的驱逐 (PurgeAgedBlocks)

```cpp
// 文件: src/storage/buffer/buffer_pool.cpp:436-462
idx_t BufferPool::PurgeAgedBlocks(uint32_t max_age_sec) {
    int64_t limit = now() - (max_age_sec * 1000);  // 过期时间阈值
    for (auto &queue : queues) {
        queue.IterateUnloadableBlocks([&](node, handle, lock) {
            if (handle->GetLRUTimestamp() >= limit) {
                handle->Unload(lock);              // 即使未过期也一并驱逐
                return false;                       // 但遇到未过期就停止
            }
            handle->Unload(lock);                  // 淘汰过期 Block
            return true;                            // 继续找下一个
        });
    }
}
```

### 近似 LRU 的效果

不依赖严格时间戳排序。通过**序列号 + 死节点**组合：

1. Block 被 unpin → 序列号递增 → 新节点入队尾
2. 最先入队的节点在队头 → 最先被出队 → 大概率 Block 已被重新访问 (序列号已变) → 死节点 → 跳过
3. 如果还是活节点且没人 pin → 驱逐它
4. `PurgeAgedBlocks` 兜底: 超过 `max_age_sec` 从未访问, 主动踢掉

```cpp
// 文件: src/storage/buffer/buffer_pool.cpp:42-59 — 节点死亡检测
bool BufferEvictionNode::IsDeadNode() {
    auto shared = memory_p.lock();         // 尝试获取 shared_ptr
    if (!shared) return true;              // Block 已析构 → 死
    if (handle_sequence_number != shared->GetEvictionSequenceNumber())
        return true;                       // 序列号变了 → 有新版本 → 死
    return false;                          // 活着
}
```

为什么用 `weak_ptr`：Block 被外部 `shared_ptr` 持有时不析构, 淘汰队列不能阻止 Block 消亡。`weak_ptr` 恰好满足: (1) 能探测 Block 是否活着, (2) Block 析构时 `lock()` 返回空, (3) 不占引用计数——不阻止 Block 被正确释放。

```cpp
// 文件: src/storage/buffer/buffer_pool.cpp:61 — moodycamel 无锁 MPMC 队列
typedef duckdb_moodycamel::ConcurrentQueue<BufferEvictionNode> eviction_queue_t;
```

---

## 12. `if constexpr` — 编译期分支 (C++17)

DuckDB 中 `if constexpr` 出现超过 60 处，替代了传统 SFINAE 和 tag dispatch。

```cpp
// 文件: src/include/duckdb/common/vector_operations/aggregate_executor.hpp:376-409
template <class OP, class STATE_TYPE, class INPUT_TYPE, class RESULT_TYPE, class ... EXTRA>
static void Execute(...) {
    for (idx_t i = 0; i < count; i++) {
        if constexpr (CHECK_VALIDITY) {         // 编译期常量 → 不编译另一半
            if (!validity.RowIsValid(i)) continue;
        }
        OP::template Operation<INPUT_TYPE, STATE_TYPE, RESULT_TYPE>(...);
    }
}

// 文件: src/include/duckdb/common/checked_integer.hpp:333-336
template <typename A, typename B>
static CheckedInteger Add(A a, B b) {
    if constexpr (std::is_signed_v<A> == std::is_signed_v<B>) {
        return AddSameSign(a, b);               // 同符号 → 简单路径
    } else if constexpr (std::is_signed_v<A>) {
        return AddSignedUnsigned(a, b);         // A 有符号 → 需要额外检查
    }
}
```

对比 C++14 写法——需要用 `std::enable_if` 或 tag dispatch 把不同分支拆成不同函数重载。`if constexpr` 把编译期分支留在同一个函数里，可读性好得多。

## 13. Structured Bindings — 解包返回值 (C++17)

```cpp
// 文件: src/function/function_binder.cpp:449-465
auto [args, kwargs] = GetArgumentsFromExpressions(regular_args, keyword_args);

// 文件: src/execution/operator/persistent/physical_copy_to_file.cpp:2337
auto [unique_count, total_count] = merged_state->GetCounts();

// 文件: src/planner/expression/bound_window_expression.cpp:338
auto [bound_func, bind_info] = function_binder.ResolveFunction(win_func, result->children);
```

替代了传统的 `std::tie` 或先声明变量再分别赋值。DuckDB 里 16 处使用。

## 14. `_v` Type Traits — 省 `::value` (C++17)

```cpp
// 文件: src/include/duckdb/common/helper.hpp:258
static_assert(std::is_same_v<T, U> || std::is_base_of_v<T, U> || std::is_base_of_v<U, T>,
              "RefersToSameObject requires T and U to be related by inheritance");

// 文件: src/include/duckdb/common/checked_integer.hpp:38
static_assert(std::is_integral_v<T>, "CheckedInteger only supports integral types");
```

C++14 写法: `std::is_same<T, U>::value`，多 7 个字符。这里用 `_v` 后缀别名（C++17 标准库提供的 inline variable template）。

## 15. `inline constexpr` — 类内静态常量 (C++17)

```cpp
// 文件: src/include/duckdb/common/file_open_flags.hpp:61
inline constexpr FileOpenFlags operator|(FileOpenFlags b) const {
    return FileOpenFlags(flags | b.flags, MergeLock(lock, b.lock), ...);
}

// 文件: src/include/duckdb/common/types.hpp:53
inline constexpr bool operator==(const list_entry_t &other) const {
    return offset == other.offset && length == other.length;
}

// 文件: src/include/duckdb/common/enums/database_modification_type.hpp:31
inline constexpr DatabaseModificationType operator|(DatabaseModificationType b) const;
```

C++14 里 `constexpr` 成员函数隐式是 `inline`，但 C++17 允许显式写明 `inline constexpr`——语义完全一致，纯粹是文档意图更清晰："这是 inline 的 constexpr 函数"。

## 16. `std::optional` — 可能没有的值 (C++17)

```cpp
// 文件: src/include/duckdb/common/optional.hpp:16
using std::optional;

// 文件: src/include/duckdb/optimizer/type_pushdown.hpp:105
std::optional<GetBinding> Resolve(ColumnBinding binding, ...);
```

DuckDB 直接 `using std::optional`，没有自己实现。用于返回值可能为空的场景（函数可能找不到 binding），和 `optional_ptr` 的定位互补——`optional` 持有值（值语义），`optional_ptr` 持有引用。

---

## DuckDB 没用的 C++17 特性

| 特性 | 为什么没用 |
|------|-----------|
| `[[nodiscard]]` | 代码库里零出现 |
| `std::string_view` | 用自己的 `string_t`（短字符串优化） |
| `std::variant` / `std::any` | 用模板 + 继承替代 |
| `std::byte` | 用 `data_t` (`uint8_t`) 替代 |
| fold expressions | 有少量使用（serializer），但不是主力 |
| `std::filesystem` | 跨平台兼容性考虑，未使用 |
