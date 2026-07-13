# PinnableView: RocksDB 风格的零拷贝字符串视图

## 设计来源

RocksDB 中 `Slice` / `PinnableSlice` / `Cleanable` 三层设计，用于在读路径上避免字符串拷贝——值直接从 block cache 返回给用户，不经过 memcpy。

## 三层架构

```
┌─────────────────────────────────────────────┐
│                 PinnableView                │
│  ┌─────────────────┐  ┌──────────────────┐  │
│  │   StringView     │  │    Cleanable     │  │
│  │  (ptr + size)    │  │ (cleanup chain)  │  │
│  └─────────────────┘  └──────────────────┘  │
│  ┌────────────────────┐                      │
│  │  std::string buf   │  ← PinSelf 回退拷贝  │
│  │  bool pinned       │  ← 当前模式标记      │
│  └────────────────────┘                      │
└─────────────────────────────────────────────┘
```

| 类 | 职责 |
|---|---|
| `StringView` | 基础类型：非拥有指针 + 长度 |
| `Cleanable` | 链表式清理回调，析构/Reset 时依次执行 |
| `PinnableView` | 继承两者，Pin（零拷贝）/ Self（拷贝）双模式 |

## Cleanable 工作方式

```cpp
// 链式注册，后注册的先执行
cleanable.RegisterCleanup(fn1, arg1, nullptr);
cleanable.RegisterCleanup(fn2, arg2, nullptr);

// 析构或 Reset 时，fn2 先执行，fn1 后执行
cleanable.Reset();  // → fn2(arg2) → fn1(arg1)
```

`DelegateCleanupsTo(other)` 将当前对象的所有回调**转移**给另一个 Cleanable，这样被引用的资源只有当最终的 PinnableView 销毁时才释放。

## PinnableView 双模式

### Pin 模式（零拷贝）

```
BlockCache 中的数据块
  │  data_ = 0x1000 ──────┐
  │                        │
  │  PinnableView          │
  │    data_ = 0x1000 ─────┘  指向同一块内存，不拷贝
  │    pinned_ = true
  │    Cleanable 链 ← 持有 Cache::Handle 释放函数
  │
  │  用户使用 value...
  │  用户销毁 PinnableView
  │    → Cleanable::DoCleanup()
  │    → 释放 Cache::Handle
  │    → Block 可被淘汰
```

### Self 模式（回退拷贝）

```
孤立的 std::string
  │
  │  PinnableView
  │    buf_.assign(src)       ← 拷贝到内部 buffer
  │    data_ = buf_.data()    ← 指向自己的 buffer
  │    pinned_ = false
```

当外部数据没有生命周期保障时，自动回退到拷贝模式。

## Demo 流程

`PinnableView_demo.cpp` 模拟了 RocksDB 的 `Get` 完整流程：

### 1. BlockCache

```cpp
class BlockCache {
    // unordered_map<int, unique_ptr<Block>>  持有数据块
    int Pin(int id);    // 引用计数 +1，返回数据指针
    void Unpin(int id); // 引用计数 -1，为 0 时淘汰
};
```

### 2. DB::Get（模拟 RocksDB table reader）

```cpp
bool Get(const string& key, PinnableView* result) {
    const char* data = cache.Pin(block_id);
    if (data) {
        StringView sv(data + offset, length);
        // PinSlice: 零拷贝 + 注册 BlockHolder 作为清理回调
        result->PinSlice(sv, cleanup_fn, new BlockHolder(&cache, block_id), nullptr);
        return true;
    }
    return false;
}
```

### 3. BlockHolder（Cleanable 子类）

```cpp
class BlockHolder : public Cleanable {
    ~BlockHolder() { cache_->Unpin(block_id_); }  // 析构时释放 pin
};
```

### 4. 生命周期

```
DB::Get("key", &result)
  │
  ├─ cache.Pin(block_id)            refcount: 0→1
  ├─ result->PinSlice(sv, cleanup)
  │    └─ RegisterCleanup(delete BlockHolder)
  │
  └─ 用户使用 result.value()        ← 零拷贝，直接读 block 内存

~PinnableView() / Reset()
  │
  └─ Cleanable::DoCleanup()
       └─ delete BlockHolder
            └─ ~BlockHolder()
                 └─ cache.Unpin(block_id)    refcount: 1→0
                      └─ evict block
```

## RocksDB 源码分析：Cleanable 的使用

### 为什么需要清理链条

一次读请求会穿透多层系统，每层独立持有资源，各层互不知道对方的存在。例如 `Get("foo")` 的真实读路径上：

```
MemTable::Get → 引用了 memtable → 释放时需 UnrefMemTable
    没命中，继续
BlockCache::Get → 引用了数据 block (ref++, cache handle) → 释放时需 cache->Unpin
    值里有 merge 操作数
MergeOperator → 分配了临时 buffer → 释放时需 free(tmp)
```

如果只支持单个清理回调，最外层 `GetContext` 就得逐一知道内层所有资源，耦合极深。清理链条让**每层只管注册自己的清理函数**：

```
RegisterCleanup(UnrefMemTable, memtable)
RegisterCleanup(UnpinBlock, cache_handle)
RegisterCleanup(FreeTemp, tmp)
```

最终 `PinnableSlice` 析构时，LIFO 依次执行，所有资源自动释放。本质是 C++ 版的 RAII 组合器——不用 centralized 管控，各自独立注册，析构时自动级联清理。

注意这里不是「不管用没用都试着去析构」——**只有实际命中并贡献了数据的路径才会注册**。MemTable 命中就只注册了 UnrefMemTable；BlockCache 命中就只注册了 UnpinBlock；Merge 路径命中才注册 UnpinBlock + FreeTemp 两个。顶层的 `GetContext` 完全不关心底层走了哪条路径。

### 哪些类继承了 Cleanable

RocksDB 中有 4 个基类直接继承 Cleanable：

| 类 | 文件 | 职责 |
|---|---|---|
| `PinnableSlice` | `include/rocksdb/slice.h:179` | 双继承 Slice + Cleanable，Get 返回值零拷贝 |
| `IteratorBase` | `include/rocksdb/iterator_base.h:14` | 用户层迭代器基类 |
| `InternalIteratorBase<TValue>` | `table/internal_iterator.h:24` | 内部迭代器基类 |
| `PinnedIteratorsManager` | `db/pinned_iterators_manager.h:19` | 批量管理多个迭代器的 pin |

`InternalIteratorBase` 的继承链最广——`BlockIter`、`BlockBasedTableIterator`、`MultiScanIndexIterator`、`PartitionedIndexIterator`、`EmbeddedBlobResolvingIterator` 全都通过它间接继承了 Cleanable。

另外有持有 Cleanable 作为成员的类：
- `ReadPathBlobResolver` — `Cleanable cleanable_` 成员，用于 pin SuperVersion 等资源
- `PlainTableReader` — `std::unique_ptr<Cleanable> dummy_cleanable_`

### IteratorBase 的 Cleanable 机制

迭代器为什么是 Cleanable？因为每次 Seek/Next/Prev 都可能加载新的 data block，需要 pin 住当前 block 防止被淘汰，同时释放旧 block 的 pin。

具体流程（以 `BlockIter::Seek` 为例）：

```
Seek(target)
 → 需要新 block → 从 BlockCache 取出 block (ref++)
 → CachableEntry::TransferTo(it)
     → RegisterCleanup(&ReleaseCacheHandle, cache, cache_handle)
         ↑ 注册到迭代器的 Cleanable 链

下次 Seek 到别的 block 时:
 → BlockIter::Invalidate()
     → Cleanable::Reset()  → 执行旧 unpin → 释放旧 block 引用
 → 加载新 block → 注册新的 cleanup
```

迭代器生命周期内，当前指向的 block 自动保持在缓存中不被淘汰。移动到新位置时，旧 block 自动释放。用户完全不感知。

### Cleanable vs SharedCleanablePtr

| | Cleanable | SharedCleanablePtr |
|---|---|---|
| 语义 | unique_ptr 语义，独占清理责任 | shared_ptr 语义，引用计数共享 |
| 可复制 | 否（copy deleted） | 是（原子引用计数） |
| 使用场景 | 单一所有者，如 PinnableSlice | 多个对象共享同一组清理责任 |

`SharedCleanablePtr::Impl` 的结构（`util/cleanable.cc:96-107`）：

```cpp
struct SharedCleanablePtr::Impl : public Cleanable {
  std::atomic<unsigned> ref_count{1};
  void Ref() { ref_count.fetch_add(1); }
  void Unref() {
    if (ref_count.fetch_sub(1) == 1) {
      delete this;  // 最后一个引用释放才执行 cleanup
    }
  }
  static void UnrefWrapper(void* arg1, void* /*arg2*/) {
    static_cast<SharedCleanablePtr::Impl*>(arg1)->Unref();
  }
};
```

与 Cleanable 的桥接通过两个方法：
- `RegisterCopyWith(target)`：将自己 Ref + 注册一个 Unref 到 target 的 Cleanable 链——"虚拟副本，只有 target 也析构后才能释放"
- `MoveAsCleanupTo(target)`：将清理责任直接转移给 target，不再 Ref——"真实转移"

典型使用场景是异步读路径 `io_dispatcher_imp.cc`：多个 IO 请求共享同一个 read buffer，最后一个请求释放时才真正 clean。

### 代码演进：三个时代的 BlockReader

同一个语义——「把 block 的释放责任挂到 iterator 上」——在三个时代的表达方式完全不同。

#### 1. LevelDB 时代：手写 if-else

`table.cc:219-312`，`Table::BlockReader` 最原始的写法。顶层定义了三个静态 cleanup 函数：

```cpp
static void DeleteBlock(void* arg, void* ignored) {
  delete reinterpret_cast<Block*>(arg);
}
static void DeleteCachedBlock(const Slice& key, void* value) {
  Block* block = reinterpret_cast<Block*>(value);
  delete block;
}
static void ReleaseBlock(void* arg, void* h) {
  Cache* cache = reinterpret_cast<Cache*>(arg);
  Cache::Handle* handle = reinterpret_cast<Cache::Handle*>(h);
  cache->Release(handle);
}
```

然后 BlockReader 函数末尾是手写的分支：

```cpp
Iterator* Table::BlockReader(void* arg, const ReadOptions& options,
                             const Slice& index_value, bool* didIO,
                             bool for_compaction, const bool no_io) {
  Table* table = reinterpret_cast<Table*>(arg);
  Cache* block_cache = table->rep_->options.block_cache.get();
  Block* block = nullptr;
  Cache::Handle* cache_handle = nullptr;

  BlockHandle handle;
  Slice input = index_value;
  Status s = handle.DecodeFrom(&input);

  if (s.ok()) {
    if (block_cache != nullptr) {
      // 构造 cache key
      char cache_key[kMaxCacheKeyPrefixSize + kMaxVarint64Length];
      memcpy(cache_key, table->rep_->cache_key_prefix,
             table->rep_->cache_key_prefix_size);
      char* end = EncodeVarint64(cache_key + table->rep_->cache_key_prefix_size,
                                 handle.offset());
      Slice key(cache_key, static_cast<size_t>(end - cache_key));

      cache_handle = block_cache->Lookup(key);
      if (cache_handle != nullptr) {
        block = reinterpret_cast<Block*>(block_cache->Value(cache_handle));
        RecordTick(statistics, BLOCK_CACHE_HIT);
      } else if (no_io) {
        return nullptr;
      } else {
        // 读文件
        s = ReadBlock(table->rep_->file.get(), options, handle, &block, didIO);
        if (s.ok()) {
          if (block->isCachable() && options.fill_cache) {
            cache_handle = block_cache->Insert(
                key, block, block->size(), &DeleteCachedBlock);
          }
        }
        RecordTick(statistics, BLOCK_CACHE_MISS);
      }
    } else if (no_io) {
      return nullptr;
    } else {
      s = ReadBlock(table->rep_->file.get(), options, handle, &block, didIO);
    }
  }

  Iterator* iter;
  if (block != nullptr) {
    iter = block->NewIterator(table->rep_->options.comparator);
    if (cache_handle == nullptr) {
      iter->RegisterCleanup(&DeleteBlock, block, nullptr);           // ①
    } else {
      iter->RegisterCleanup(&ReleaseBlock, block_cache, cache_handle); // ②
    }
  } else {
    iter = NewErrorIterator(s);
  }
  return iter;
}
```

**问题**：整个函数是一大坨过程式代码，cache lookup → miss 则读文件/insert cache → 创建 iterator → 最后手写 if-else 决定清理方式。每个调用方都要记住用哪个 cleanup 函数。

#### 2. 早期 RocksDB：更多分支，同样的手写模式

`block_based_table_reader.cc:1964-2080`（`f0bf3bf34^`），`NewDataBlockIterator`。逻辑膨胀了很多（compression dict、persistent cache、dummy cache entry for memory tracking...），但 cleanup 仍然是散落的手写 RegisterCleanup：

```cpp
if (block.cache_handle != nullptr) {
  iter->RegisterCleanup(&ReleaseCachedEntry, block_cache,
                        block.cache_handle);                         // ③ 来自 cache
} else {
  if (!ro.fill_cache && rep->cache_key_prefix_size != 0) {
    // 插入一个 dummy cache entry 来追踪内存使用
    // ... 构造 unique key ...
    s = block_cache->Insert(unique_key, nullptr, ... , &cache_handle);
    if (s.ok()) {
      if (cache_handle != nullptr) {
        iter->RegisterCleanup(&ForceReleaseCachedEntry, block_cache,
                              cache_handle);                         // ④ dummy entry
      }
    }
  }
  iter->RegisterCleanup(&DeleteHeldResource<Block>, block.value, nullptr); // ⑤ owned
}
```

同一个函数内部有 ③④⑤ 三种不同的 cleanup 注册，分别对应「来自 cache」「来自 dummy cache entry」「自己 owns block」。调用方得同时操心清理逻辑和业务逻辑。

#### 3. 现代 RocksDB：CachableEntry::TransferTo

`block_based_table_reader_impl.h:166-220`：

```cpp
template <typename TBlockIter>
TBlockIter* BlockBasedTable::NewDataBlockIterator(const ReadOptions& ro,
                                                  CachableEntry<Block>& block,
                                                  TBlockIter* input_iter,
                                                  Status s) const {
  TBlockIter* iter = input_iter != nullptr ? input_iter : new TBlockIter;
  if (!s.ok()) {
    iter->Invalidate(s);
    return iter;
  }

  // ... init iterator, handle dummy cache entry ...

  block.TransferTo(iter);    // 👈 一行替代之前的 ③④⑤
  return iter;
}
```

`CachableEntry::TransferTo` 内部（`cachable_entry.h:126-137`）：

```cpp
void TransferTo(Cleanable* cleanable) {
  if (cleanable) {
    if (cache_handle_ != nullptr) {
      cleanable->RegisterCleanup(&ReleaseCacheHandle, cache_, cache_handle_);
    } else if (own_value_) {
      cleanable->RegisterCleanup(&DeleteValue, value_, nullptr);
    }
  }
  ResetFields();  // 所有权转交，自己不再负责
}
```

而调用方的视角也更清爽（`binary_search_index_reader.cc:56-73`）：

```cpp
  auto it = index_block.GetValue()->NewIndexIterator(...);
  assert(it != nullptr);
  index_block.TransferTo(it);   // 一行，不需要知道 block 是 cache 的还是 owned
  return it;
```

#### 演进总结

| | 写法 | 调用方需要知道什么？ |
|---|---|---|
| LevelDB | `if (cache_handle) RegisterCleanup(ReleaseBlock, ...) else RegisterCleanup(DeleteBlock, ...)` | cache handle 存在与否、该用哪个 cleanup 函数 |
| 早期 RocksDB | 同上 + 多了 dummy cache entry、persistent cache 等分支 | 三种清理方式，加上 compression dict 的独立 release |
| 现代 | `block.TransferTo(iter)` | **什么都不需要知道** |

演变路径：每次资源类型增加（persistent cache、dummy entry、row cache...），调用方就得在每个地方加新的 if-else 分支。`TransferTo` 把这些决策归一到 `CachableEntry` 内部——资源持有者自己知道该怎么释放自己，调用方只管"转移所有权"。

## 文件索引

| 文件 | 说明 |
|------|------|
| `StringView.hpp` | 基础视图类型 |
| `Cleanable.hpp` | 清理回调链表 |
| `PinnableView.hpp` | Pin/Copy 双模式 |
| `PinnableView_demo.cpp` | 模拟 RocksDB Get 流程 |
| `PinnableView.md` | 本文档 |

## 编译运行

```bash
g++ -std=c++17 -O2 -Wall -Wextra -o demo PinnableView_demo.cpp && ./demo
```

## 输出关键点

```
--- Get user:1 ---
[cache]  pin   block:0  ref=1
  val.data: 0x600003c1d240
  block:    0x600003c1d240
  same?     YES (zero-copy!)        ← Pin 模式，地址相同
  (PinnableView going out of scope...)
[cache]  unpin block:0  ref=0
[cache]  evict block:0              ← 清理回调执行，block 被淘汰

--- PinSelf fallback ---
  val.data: 0x60000321c2d0
  src.data: 0x600003c1d2a0
  same?     no — copied into internal buffer  ← Self 模式，地址不同
```

## 核心原则

> **PinnableView 的寿命不能超过它指向的数据。** 在 Pin 模式下，Cleanable 链自动保证这一点；在 Self 模式下，数据在内部 buffer 中，天然安全。
