# moodycamel ConcurrentQueue 实现解析

DuckDB 用 `moodycamel::ConcurrentQueue` 作为 BufferPool 淘汰队列。
源码: `third_party/concurrentqueue/concurrentqueue.h` (3684 行)

---

## 整体架构

```
ConcurrentQueue<T>
  │
  ├─ Producer 链表 (无锁, 每个线程一个)
  │   ├─ ExplicitProducer (主动创建 token 的生产者)
  │   │   └─ Block[32] × N (每个 Block 存 32 个 T, 用 bitmask 标记空/满)
  │   └─ ImplicitProducer (无 token, 按 thread_id 自动分配)
  │       └─ 同上链路
  │
  └─ Consumer (消费者按 round-robin 轮转所有 Producer)
```

核心思路: **每个生产者独立子队列 + 消费者轮转** = 多生产者间零竞争。

---

## 生产者: 独立子队列

每个线程入队时只操作自己的 Producer 对象，不碰其他线程的:

```cpp
// 文件: concurrentqueue.h:1639-1697
template<typename T>
struct ProducerBase {
    ProducerBase *next;                             // 所有 Producer 串成链表
    std::atomic<bool> inactive;                     // 标记是否活跃
    ProducerToken *token;

    ConcurrentQueue *parent;                        // 指回队列
    bool isExplicit;                                // explicit(有token) vs implicit
};
```

### ExplicitProducer 入队 (concurrentqueue.h:1712-1800)

```cpp
// 入队核心: 往自己的 Block 链写数据
template<AllocationMode canAlloc, typename U>
inline bool enqueue(U&& element) {
    auto currentTailIndex = this->tailIndex.load(std::memory_order_relaxed);
    auto block = this->tailBlock;

    // ① 先试着往当前尾 Block 的空位写
    auto index = currentTailIndex & (BLOCK_SIZE - 1);   // 取模
    if (block->elements[index].sequence == 0) {          // 这个位置是空的?
        block->elements[index].element = element;
        block->elements[index].sequence = currentTailIndex + 1;  // 写入 sequence 标记
        this->tailIndex.fetch_add(1, std::memory_order_release);
        return true;
    }

    // ② Block 满了 → 分配新 Block，链在链表尾
    auto newBlock = parent->template requisition_block<canAlloc>();
    newBlock->elements[0].element = element;
    newBlock->elements[0].sequence = currentTailIndex + 1;
    block->next = newBlock;
    this->tailBlock = newBlock;
    this->tailIndex.store(currentTailIndex + 1, std::memory_order_release);
    return true;
}
```

Block 是一个定长数组 + 每个元素带 `sequence` 标签:

```cpp
struct Block {
    struct Element {
        std::atomic<index_t> sequence;    // 0=空, N=写入完成
        T element;
    };
    Element elements[BLOCK_SIZE];         // 默认 32 个元素
    Block *next;                          // 链表
};
```

### 为什么多个生产者不冲突?

- Producer A 只写自己的 `tailBlock` 和 `tailIndex`
- Producer B 只写自己的 `tailBlock` 和 `tailIndex`
- 消费者遍历生产者链表，轮流从各生产者取数据

---

## 消费者: Round-Robin 轮转

```cpp
// 文件: concurrentqueue.h:1332-1370
inline bool update_current_producer_after_rotation(consumer_token_t& token) {
    auto tail = producerListTail.load(std::memory_order_acquire);
    auto prodCount = producerCount.load(std::memory_order_relaxed);
    auto globalOffset = globalExplicitConsumerOffset.load(std::memory_order_relaxed);

    // 从当前 Producer 前进 1 个 (round-robin)
    token.currentProducer = token.currentProducer->next_prod();
    token.itemsConsumedFromCurrent = 0;
    globalExplicitConsumerOffset.fetch_add(1, std::memory_order_release);

    return true;
}
```

消费者按链表顺序轮转: 每个消费者线程固定消费 `EXPLICIT_CONSUMER_CONSUMPTION_QUOTA_BEFORE_ROTATE` (256) 个元素后换下一个 Producer。

出队核心:

```cpp
template<typename U>
inline bool try_dequeue(U& element) {
    auto block = currentProducer->headBlock;
    auto index = block->headIndex.load(std::memory_order_relaxed);

    // 读 sequence: 如果 sequence == headIndex + 1，说明这个位置已被写入且有效
    auto seq = block->elements[index].sequence.load(std::memory_order_acquire);
    if (seq != headIndex + 1) {
        return false;  // 还没写入完成，或为空
    }

    // 取元素，清除 sequence
    element = block->elements[index].element;
    block->elements[index].sequence = 0;  // 标记为空
    block->headIndex.store(index + 1, std::memory_order_release);
    return true;
}
```

---

## 无锁的关键: Sequence 标签 + CAS

不靠全局互斥锁，靠每个 Block slot 的 `sequence` 原子变量:

```
Block:
  slot[0]: sequence=0      ← 空
  slot[1]: sequence=42     ← 已被写入 (入队时 seq = tailIndex + 1)
  slot[2]: sequence=0      ← 空
```

- **入队**: 生产者看到 `sequence == 0` → 写入数据 → 设 `sequence = tailIndex + 1`
- **出队**: 消费者看到 `sequence == headIndex + 1` → 读数据 → 设 `sequence = 0`
- **冲突**: 如果两个线程同时看同一个 slot, CAS 保证只有一个成功

---

## 无锁 Freelist (ConcurrentQueue 内存管理)

Blocks 用完后归还到 freelist 而非 free 掉，下次重用:

```cpp
// 文件: concurrentqueue.h:1389-1410
template<typename N>
struct FreeList {
    std::atomic<N*> freeListHead;     // 无锁栈

    void add(N* node) {
        // CAS 将 node 推到栈顶
        auto oldHead = freeListHead.load(std::memory_order_relaxed);
        do {
            node->freeListNext.store(oldHead, std::memory_order_relaxed);
        } while (!freeListHead.compare_exchange_weak(
            oldHead, node, std::memory_order_release, std::memory_order_relaxed));
    }

    N* try_get() {
        // CAS 从栈顶取出一个
        auto head = freeListHead.load(std::memory_order_acquire);
        while (head != nullptr) {
            auto next = head->freeListNext.load(std::memory_order_relaxed);
            if (freeListHead.compare_exchange_weak(
                head, next, std::memory_order_release, std::memory_order_relaxed)) {
                return head;  // 拿到了
            }
        }
        return nullptr;  // 栈空
    }
};
```

经典的无锁栈——只用一个 `atomic<N*> freeListHead` + CAS 循环。

---

## DuckDB 的使用方式

```cpp
// buffer_pool.cpp:61
typedef duckdb_moodycamel::ConcurrentQueue<BufferEvictionNode> eviction_queue_t;

// 入队 (任意线程, 无锁)
queue.AddToEvictionQueue(BufferEvictionNode(handle->GetMemoryWeak(), ts));
//    └→ q.enqueue(std::move(node));

// 出队 (任意线程, 无锁)
queue.try_dequeue(node);
//    └→ 轮转各 Producer, 读 sequence 标签, 取出元素

// 暴力出队 (只有 purge 线程, 加锁保底)
queue.TryDequeueWithLock(node);
//    └→ lock_guard + try_dequeue, 保证不会因为 CAS 失败而漏掉元素
```

DuckDB 用 `ConsumerToken` 和 `ProducerToken` 做 bulk purge 优化（`purge_consumer_token`、`purge_producer_token`），让 purge 线程在队列里快速定位自己的子队列，不用每次都从链表头扫。

---

## 为什么是 lock-free 而不是 wait-free?

- **lock-free**: 至少有一个线程能推进（不会全局死锁），但单个线程可能 CAS 失败重试
- **wait-free**: 每个线程都能在有限步内完成

ConcurrentQueue 是 lock-free：出队遇到无效 sequence 返回 false（非阻塞），入队 CAS 失败重试。不是 wait-free，因为在极端竞争下某个线程可能一直 CAS 失败。

---

## DuckDB 为什选它?

1. **MPMC** — 多生产者多消费者，BufferPool 的 eviction queue 正是这种场景
2. **无锁** — 不会因为队列操作导致 buffer eviction 路径被阻塞
3. **benchmark 验证过** — 作者 blog 有详尽性能分析，lock-free 设计久经考验
4. **Header-only** — 没有链接负担，集成成本低
