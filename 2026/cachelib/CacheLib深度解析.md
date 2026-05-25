# CacheLib 深度解析

> Meta 开源的 in-process C++ 高性能缓存库。这份文档总结对它存储引擎、I/O 架构、并发模型、设计取舍的探究。

---

## 一、项目定位

**CacheLib = 嵌入式 C++ 库，不是网络服务。**

- 形态：链接进你自己的进程里，通过函数调用使用 (`cache->find(key)`)
- 与 Memcached 的关系：Meta 内部的 Memcached fork **用 CacheLib 作为存储引擎**，外面套 Memcached 协议提供网络服务
- 类比：Memcached 是"成品轿车"，CacheLib 是"高性能发动机"

源码里 thrift 只用作 `frozen2` 序列化（持久化共享内存/磁盘格式），**没有任何 RPC**。

---

## 二、整体架构

```
                  ┌──────────────────────────┐
                  │     业务代码              │
                  └──────────────────────────┘
                       │       │       │
       ┌───────────────┘       │       └────────────────┐
       ▼                       ▼                        ▼
┌──────────────┐      ┌─────────────────┐      ┌────────────────┐
│ object_cache │      │   datatype/     │      │ compact_cache  │
│ (存 C++ 对象)│      │ (Map/List 视图) │      │ (超小固定 KV)  │
└──────┬───────┘      └────────┬────────┘      └────────┬───────┘
       │                       │                        │
       │ 都基于                │                        │ 独立实现
       └──────────► ┌──────────▼──────────┐ ◄──────────┘
                    │  CacheAllocator     │
                    │   (DRAM slab)       │
                    └──────────┬──────────┘
                               │ evict
                               ▼
                    ┌─────────────────────┐
                    │   Navy (SSD/NVM)    │
                    │ ┌─────────────────┐ │
                    │ │ BlockCache      │ │ ← 大对象
                    │ │ BigHash         │ │ ← 小对象
                    │ └─────────────────┘ │
                    └─────────────────────┘
```

---

## 三、存储引擎

### 3.1 DRAM 层 —— Slab Allocator（自研）

位置：`cachelib/allocator/memory/`

- **`SlabAllocator`**：内存切成 **4MB Slab**
- **`AllocationClass`**：每个 class 把 slab 切成固定 size 小块
- **`MemoryPool`**：多池隔离，业务可分池
- **`CompressedPtr`**：32-bit 压缩指针
- 可挂共享内存（`shm/`）—— **进程崩溃/重启 cache 存活**

淘汰策略外挂：**LRU / LRU-2Q / 5B-LRU / TinyLFU / W-TinyLFU**（见 `CacheAllocator*.cpp`）

#### 为什么不用 jemalloc

| 维度 | jemalloc | Slab |
|---|---|---|
| 碎片回收粒度 | 单个 object | 整个 4MB Slab |
| Cache workload 调优 | 不支持 size-class 级 hit ratio 决策 | 支持 `MarginalHitsStrategy` 等 |
| 共享内存重启 | metadata 散布无法序列化 | header 集中存储可重建 |
| Hot path 方差 | 最坏情况要 chunk 拆分 | 单 class 链表 pop 几乎恒定 |

**jemalloc 优化通用程序平均效率，slab 优化 cache workload 可控性。** 不同目标。

### 3.2 SSD 层 —— Navy（自研）

位置：`cachelib/navy/`，**两套引擎按对象大小路由**。

#### BlockCache —— 大对象（≥ 几 KB）

- 物理布局：**16MB Region**（默认），日志结构追加写
- 写入：在 active region 顺序追加 → region 写满切换 → 异步整 region 刷盘
- 内存索引：`SparseMapIndex` / `FixedSizeIndex` 必须有 `key → (region, offset, size)`
- 淘汰：以 region 为单位，FIFO/LRU + `HitsReinsertionPolicy` 把热对象 reinsert 到新 region
- **写放大低**（区块整擦），**热点聚集能力强**（reinsertion）

#### BigHash —— 小对象（< 几 KB）

- 物理布局：**4KB Bucket**（默认）
- 写入：hash 到 bucket → 整 bucket 读-改-写
- **零内存索引**（位置由 hash 函数决定）
- 唯一内存数据：per-bucket Bloom Filter（过滤 miss I/O）
- 淘汰：bucket 内 FIFO，写满踢最老
- **几乎零元数据**，但**没有热点聚集机制**

#### 关键对比

| 维度 | BlockCache | BigHash |
|---|---|---|
| 目标对象 | 大对象 | 小对象 |
| 物理单位 | 16MB Region | 4KB Bucket |
| 内存索引 | 必须 (~10–20B/entry) | 无 |
| 写放大 | 低 | 中 |
| 读放大 | 低 | 高 (4KB/op) |
| 热点聚集 | ✅ Reinsertion | ❌ 无 |
| 典型用途 | CDN body / 模型分片 | 社交边 / 元数据 |

---

## 四、Bucket 内部冲突与查找（BigHash 细节）

**冲突处理 = 不处理，挤在一起线性扫。**

```
┌────────────────────────────────────────────────────────────┐
│ Bucket header (24B): checksum + generationTime + storage   │
├────────────────────────────────────────────────────────────┤
│ Entry0: [16B meta][key0 bytes][value0 bytes]               │
│ Entry1: [16B meta][key1 bytes][value1 bytes]               │
│ ...                                                        │
└────────────────────────────────────────────────────────────┘
```

`Bucket::find` 是 O(n) 线性扫，靠 entry 内置的 8B `keyHash_` 当 fingerprint 加速比较。

**为什么这种"看似很蠢"的设计是对的**：SSD 最小 I/O 单元就是 4KB，读 1 个 entry 和读 40 个 entry 的硬盘成本一样（~100μs vs <1μs 扫描）。**Scan 完全免费**，建二级索引反而是浪费。

---

## 五、I/O 架构（绕过 page cache）

### 5.1 完全旁路 OS page cache

`navy/common/Device.cpp:1166`：
```cpp
f = folly::File(fileName.c_str(), flags | O_DIRECT);
```

兜底 `posix_fadvise(POSIX_FADV_DONTNEED)` 即使 O_DIRECT 失败也尽量清 page cache。

**为什么不用 page cache**：
- 内核 LRU 对 cache workload 不友好
- DRAM 层 + page cache 双重缓存浪费
- 写时机不可控，加剧 SSD 写放大
- 与 Kafka/PostgreSQL "故意依赖 page cache 做读缓存" 的哲学**相反**

### 5.2 io_uring 集成

#### 几个核心设计

1. **不直接用 liburing**，用 `folly::IoUring`（和 libaio 共抽象，配置切换）
2. **Thread-local ring**：每线程独占一个 io_uring 实例，**零锁**
3. **EventBase + eventfd 收割**：io_uring CQ 注册到 epoll，零额外线程
4. **背压**：qDepth 满了挂 WaiterList，天然限流

#### 三种"挂起" CPU 给谁

```
模型 A：阻塞 syscall          → 另一个 OS 线程 (内核调度, ~1-3μs)
模型 B：异步 + 回调            → 同线程下一个 callback (~100ns，但 callback hell)
模型 C：异步 + fiber (CacheLib)→ 同线程下一个 fiber (~50ns，代码同步写法)
```

CacheLib 选 C。**OS 线程从不阻塞**，靠 folly fiber 在用户态做栈切换。

#### io_uring 完成路径的 4 种模式

| 模式 | 中断 | CPU 开销 | 适用 |
|---|---|---|---|
| **默认** | 有 (NVMe MSI-X) | 中 | 通用 |
| **IOPOLL** | 无（kernel 轮询设备）| 高（一核 100%）| 极低延迟 |
| **SQPOLL** | 无 syscall 提交 | 高 | 提交侧优化 |
| **io-wq** | 内核线程池兜底 | 中 | 不支持 nowait 的 syscall |

**CacheLib 选默认模式 + eventfd**：中断驱动 → cqe 入 ring → eventfd 触发 → EventBase epoll 唤醒 fiber。

#### "u" = userspace

io_uring 的 SQ/CQ 是 **mmap 到用户态的 ring buffer**，用户态可以直接读写 sqe/cqe 不需要 syscall。这是它比 libaio 快几倍的根本原因。

### 5.3 NVMe 高级特性

`Device.cpp:1101` 通过 io_uring 的 `URING_CMD` 直接给 NVMe 控制器发命令（passthrough），配合 **NVMe FDP (Flexible Data Placement)** 把不同 region 的数据隔离到不同 erase block —— **写放大趋近于零**。这是 libaio 路径完全做不到的。

---

## 六、并发模型 —— folly fiber

### 全局 M:N，每线程内 M:1

```
全局看：几百 fiber 跑在 ~16 个 OS 线程
每线程看：几十 fiber + 1 个独占 io_uring + 1 个 EventBase
```

**线程间不迁移 fiber、不共享 io_uring** —— 所有 hot path 数据结构都是 thread-local，**零跨核同步开销**。

### 与微信 libco 对比

| 维度 | libco | folly fiber |
|---|---|---|
| 思想 | stackful coroutine + 事件驱动 | 同 |
| 主战场 | 网络 I/O | 网络 + 存储 + 锁 + 任何事件 |
| Hook 方式 | **dlsym 劫持 libc**，业务零改 | **不 hook**，必须显式用 folly API |
| 同步原语 | 极简 | 全套 fiber-aware (`Baton` / `TimedMutex` / `Semaphore`) |

**libco 走"零侵入劫持"路线；folly fiber 走"重定义并发原语生态"路线。** 思想同源，工程哲学一轻一重。

### 阻塞陷阱

stackful coroutine 通病：在 fiber 里调用真正阻塞的东西会卡死整个 OS 线程。

| 危险操作 | 解法 |
|---|---|
| `std::mutex` 撞竞争 | 用 `folly::fibers::TimedMutex` |
| 阻塞 socket read | 用 `folly::AsyncSocket` + EventBase |
| 阻塞磁盘 read | 必走 `O_DIRECT` + io_uring |
| `sleep()` | 用 `Baton::try_wait_for` |
| 第三方库内部 `pthread_mutex_lock` | 外包到独立 thread pool |

逃生通道：`folly::via(blockingExecutor, blocking_fn)`。

---

## 七、持久化语义

**Navy 是 "best-effort 持久化"，没有强 fsync 路径。**

| | BlockCache | BigHash |
|---|---|---|
| 写路径有 DRAM buffer? | 有（per-region 16MB） | 无 |
| `set()` 返回 = 已落 SSD? | ❌（在 DRAM buffer 里） | ❌（在 SSD controller cache 里） |
| 正常写路径有 fsync? | **无** | **无** |
| 进程崩溃丢多少? | 最近 N 个 region | 最近若干 in-flight bucket |
| 正常 shutdown | `persist()` fsync + dump metadata | 同 |
| 崩溃后启动 | **整个 cache 丢，从零开始** | 同 |

**Cache 的可靠性来自下游 source-of-truth，不来自自己的 fsync**。加 fsync 会让吞吐降 5–10 倍，对 cache 没意义。

---

## 八、TTL 过期

### 三层设计

1. **DRAM**：`CacheItem` 自带 4B `expiryTime_`
2. **Navy**：业务注入 `ExpiredCheck` 回调（Navy 看不懂字节流里 expiry 在哪）
3. **Admission**：`nvmAdmissionMinTTL` 拦短 TTL，避免无意义下盘

### 触发时机 —— **全是懒过期**

| 时机 | 行为 |
|---|---|
| `find()` 读到过期 | 当场返回 miss，物理空间不动 |
| LRU 淘汰扫尾部 | 顺手处理 |
| BigHash `makeSpace` | 整 bucket 扫一遍过期 entry 一起清 |
| BlockCache region reclaim | 整块回收时自然消失 |
| **定时器主动扫** | **没有** |

**和 Memcached 一样的哲学**：cache 不需要"过期即消失"的精确语义，省后台线程开销。

---

## 九、特化模块

### compact_cache —— 给超小对象省 metadata

- N-way 组相联哈希表，每 bucket 紧密堆 N 个固定大小 entry
- **零元数据**（普通 cache 每 item 30–50B metadata，存 8B value 浪费 5 倍）
- bucket 内 LRU（读 → 提到顶部）
- **value 大了退化**（每次访问要 memmove N 个 entry）
- 典型场景：UserId → score (8B → 4B)、A/B 实验分桶

### object_cache —— 缓存活的 C++ 对象

- 直接存 `T*` 指针，cache 拿对象所有权
- **零序列化** —— hit 后直接当 C++ 对象用
- **嵌套指针不管**：靠 jemalloc 自然回收 + `ThreadMemoryTracker` 估算对象大小
- **不能落盘**（指针无意义）—— 真正的卖点是"复用 CacheAllocator 一整套淘汰/并发/统计基础设施"
- 典型场景：ML 模型权重、解析好的配置树

### datatype —— cache item 内部的结构化容器

- `Map<K,V>`（robin-hood hashing）/ `RangeMap` / `List` / `FixedSizeArray`
- 在一个 cache item 内部建小型数据结构，**避免"全量读改写"**
- 用 chained item 拼接大对象
- 典型场景：社交图谱邻接表（按好友 id 查/插/删，不用 get+反序列化+set）

---

## 十、设计取舍的真相

### BigHash 的"零内存索引"是用什么换来的？

**用"放弃热点优化"换来的**。

| 问题 | 答案 |
|---|---|
| Hash 均匀分布会让热 key 散落、每次 hit 都打 SSD 吗？ | **会**，这是真实缺陷 |
| 为什么不聚集？ | 要 per-key 热度元数据，**违背"零内存索引"宗旨** |
| 怎么活下来？ | 押注 DRAM 层挡住所有热 key，BigHash 只处理"温冷数据" |
| 押注失败场景 | DRAM 装不下热工作集 → 退化为每次 4KB 随机读 |
| 真要解决怎么办 | 换 BlockCache（有 region-level reinsertion，但必须维护 in-memory index） |

### 两层 BigHash 设想 —— 学术界已经做了

思路：tier 1 小、tier 2 大，先查小后查大，热点自然沉到大层。这就是 **SLRU / W-TinyLFU 的 window+main** 模式。

**CacheLib DRAM 层已经这么做了**（`CacheAllocatorWTinyLFUCache.cpp`），SSD 层没做，原因：

| 代价 | 单层 | 两层 |
|---|---|---|
| 读 I/O（命中） | 1 次 | ~1.01 次（Bloom filter 顶住）|
| 读 I/O（miss） | 0 | 0 |
| **写 I/O（promote）** | 0 | **4 次/promote** ← 真痛点 |
| Bloom filter 内存 | N×几B | **2×N 几B**（翻倍） |
| SSD 寿命 | 基准 | 减半 |

**真正的代价不是读放大（Bloom 顶住了），是写放大 + BF 内存翻倍**。

**Meta 自己也认这是问题**，专门发了 Kangaroo 论文（FAST '21）做两层 BigHash，靠"批量 promote"摊薄写放大，实测 miss ratio 降 29%、WAF 降 56%。但**没合并进 CacheLib 主线**。

---

## 十一、CacheLib vs Memcached

| 维度 | Memcached | CacheLib |
|---|---|---|
| 形态 | 独立 daemon + TCP | 嵌入式 C++ 库 |
| 访问 | 网络协议 | 函数调用 |
| 跨语言 | 任何语言 | 仅 C++（有 Rust 绑定） |
| 存储介质 | 纯 DRAM | DRAM + SSD/NVM |
| 进程崩溃 | 全丢 | 共享内存存活 |
| 淘汰策略 | LRU | LRU / 2Q / TinyLFU / W-TinyLFU |
| 复合数据结构 | 无 | Map / List / RangeMap |
| 网络协议 | 内置 | **无**，需自包 |
| 单次 hit 延迟 | ~80–150μs（网络）| ~100–500ns（函数调用）|
| 单机 QPS | ~500K–1M | 几千万 |

**两者不是替代关系**。Meta 内部用 CacheLib 当发动机，外面套 Memcached/Thrift 协议当车壳。

---

## 十二、关键源码索引

| 想看 | 路径 |
|---|---|
| DRAM 分配器 | `cachelib/allocator/memory/SlabAllocator.cpp` |
| 各种淘汰策略 | `cachelib/allocator/CacheAllocator*.cpp` |
| BigHash 实现 | `cachelib/navy/bighash/{BigHash,Bucket,BucketStorage}.cpp` |
| BlockCache 实现 | `cachelib/navy/block_cache/{BlockCache,RegionManager,Region}.cpp` |
| 索引（BlockCache 专用） | `cachelib/navy/block_cache/{SparseMapIndex,FixedSizeIndex}.cpp` |
| Reinsertion 策略 | `cachelib/navy/block_cache/{Hits,ReuseTime}ReinsertionPolicy.cpp` |
| **Device + io_uring 集成** | `cachelib/navy/common/Device.cpp` |
| **NVMe FDP** | `cachelib/navy/common/FdpNvme.cpp` |
| Admission policy | `cachelib/navy/admission_policy/` |
| 共享内存 | `cachelib/shm/` |
| 高级数据结构 | `cachelib/datatype/{Map,RangeMap,List}.h` |
| 对象缓存 | `cachelib/object_cache/ObjectCache.h` |
| 紧凑型缓存 | `cachelib/compact_cache/CCache.h` |

---

## 十三、一句话总结的总结

- **CacheLib = 嵌入式 C++ cache 引擎库**，不是网络服务
- **DRAM 用 slab**，不是 jemalloc —— 因为 cache 要可控碎片、可调优、可共享内存重启
- **SSD 用 Navy**：大对象 BlockCache（日志结构 + 内存索引）、小对象 BigHash（bucket hash + 零索引）
- **io_uring + folly fiber + EventBase**：用户态零锁、零阻塞、每线程独占 ring，能用十几个线程喂饱 NVMe 几十万 IOPS
- **不刷盘、不持久化** —— 崩了就丢，cache 的可靠性来自下游
- **TTL 全靠懒过期** —— 没有后台线程主动扫
- **BigHash 不优化热点是真缺陷**，靠"DRAM 层挡住热 key"押注；如果押注失败应该用 BlockCache 或学术界的 Kangaroo
- **设计哲学**：用 specialization 换性能（compact / object / datatype 三个特化模块），用上层智能换下层简单（admission policy / DRAM tier 帮 SSD 引擎挡住复杂度）
