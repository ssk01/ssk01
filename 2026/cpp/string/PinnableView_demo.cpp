#include "PinnableView.hpp"
#include "Cleanable.hpp"
#include "StringView.hpp"

#include <iostream>
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>
#include <cassert>

static int s_alloc_count = 0;
static int s_free_count = 0;

struct Block {
    std::string data;
    int refcount = 0;
    int id;

    Block(int i, std::string d) : data(std::move(d)), id(i) {
        ++s_alloc_count;
        std::cout << "[block:" << id << "] alloc  \"" << data << "\"\n";
    }
    ~Block() {
        ++s_free_count;
        std::cout << "[block:" << id << "] free   \"" << data << "\"\n";
    }
};

class BlockCache {
public:
    ~BlockCache() { blocks_.clear(); }

    Block* Insert(int id, std::string data) {
        auto block = std::make_unique<Block>(id, std::move(data));
        Block* ptr = block.get();
        blocks_[id] = std::move(block);
        return ptr;
    }

    const char* Pin(int id) {
        auto it = blocks_.find(id);
        if (it == blocks_.end()) return nullptr;
        it->second->refcount++;
        std::cout << "[cache]  pin   block:" << id << "  ref=" << it->second->refcount << "\n";
        return it->second->data.data();
    }

    void Unpin(int id) {
        auto it = blocks_.find(id);
        if (it == blocks_.end()) return;
        it->second->refcount--;
        std::cout << "[cache]  unpin block:" << id << "  ref=" << it->second->refcount << "\n";
        if (it->second->refcount <= 0) {
            std::cout << "[cache]  evict block:" << id << "\n";
            blocks_.erase(it);
        }
    }

    size_t size() const { return blocks_.size(); }

private:
    std::unordered_map<int, std::unique_ptr<Block>> blocks_;
};

class BlockHolder : public Cleanable {
public:
    BlockHolder(BlockCache* cache, int block_id)
        : cache_(cache), block_id_(block_id) {}

    ~BlockHolder() {
        if (cache_) {
            cache_->Unpin(block_id_);
        }
    }

    BlockHolder(const BlockHolder&) = delete;
    BlockHolder& operator=(const BlockHolder&) = delete;
    BlockHolder(BlockHolder&&) = delete;
    BlockHolder& operator=(BlockHolder&&) = delete;

private:
    BlockCache* cache_;
    int block_id_;
};

struct KVPair {
    int block_id;
    size_t offset;
    size_t length;
};

struct DB {
    BlockCache cache;
    std::unordered_map<std::string, KVPair> index;

    void Put(const std::string& key, const std::string& value) {
        int bid = static_cast<int>(cache.size());
        Block* block = cache.Insert(bid, value);
        index[key] = {bid, 0, block->data.size()};
    }

    bool Get(const std::string& key, PinnableView* result) {
        auto it = index.find(key);
        if (it == index.end()) return false;

        const KVPair& kv = it->second;
        const char* data = cache.Pin(kv.block_id);
        if (data) {
            StringView sv(data + kv.offset, kv.length);
            result->PinSlice(sv, Cleanable::CleanupFunction{
                [](void* arg1, void* /*arg2*/) {
                    auto* h = static_cast<BlockHolder*>(arg1);
                    delete h;
                }
            }, new BlockHolder(&cache, kv.block_id), nullptr);
            return true;
        }
        return false;
    }
};

int main() {
    std::cout << "====== PinnableView: RocksDB-style zero-copy demo ======\n\n";

    DB db;
    db.Put("user:1", "Alice_Jones_Engineer_NY");
    db.Put("user:2", "Bob_Smith_Manager_LA");
    std::cout << "\nCache has " << db.cache.size() << " blocks\n\n";

    std::cout << "--- Get user:1 ---\n";
    {
        PinnableView val;
        bool ok = db.Get("user:1", &val);
        assert(ok);

        std::cout << "  IsPinned: " << val.IsPinned() << "\n";
        std::cout << "  value:    \"" << val << "\"\n";

        const char* block_data = nullptr;
        for (const auto& [k, kv] : db.index) {
            if (kv.block_id == 0) { block_data = db.cache.Pin(kv.block_id); db.cache.Unpin(kv.block_id); break; }
        }
        std::cout << "  val.data: " << static_cast<const void*>(val.data()) << "\n";
        std::cout << "  block:    " << static_cast<const void*>(block_data) << "\n";
        std::cout << "  same?     " << (val.data() == block_data ? "YES (zero-copy!)" : "no") << "\n";

        std::cout << "\n  (PinnableView going out of scope...)\n";
    }
    std::cout << "  Cache has " << db.cache.size() << " blocks\n\n";

    std::cout << "--- Get user:2, then Reset early ---\n";
    PinnableView val2;
    db.Get("user:2", &val2);
    std::cout << "  IsPinned: " << val2.IsPinned() << "\n";
    std::cout << "  value:    \"" << val2 << "\"\n";
    std::cout << "  Cache has " << db.cache.size() << " blocks (still pinned)\n";

    std::cout << "  Calling Reset()...\n";
    val2.Reset();
    std::cout << "  Cache has " << db.cache.size() << " blocks (unpinned, may be evicted)\n\n";

    std::cout << "--- PinSelf fallback (no block cache) ---\n";
    {
        PinnableView val3;
        std::string standalone = "standalone_value_no_cache";
        val3.PinSelf(StringView(standalone));
        std::cout << "  IsPinned: " << val3.IsPinned() << "\n";
        std::cout << "  value:    \"" << val3 << "\"\n";
        std::cout << "  val.data: " << static_cast<const void*>(val3.data()) << "\n";
        std::cout << "  src.data: " << static_cast<const void*>(standalone.data()) << "\n";
        std::cout << "  same?     " << (val3.data() == standalone.data()
                           ? "no — copied into internal buffer" : "no — copied into internal buffer")
                  << "\n";
    }

    std::cout << "\n--- Merged value: pin-then-remove_prefix ---\n";
    {
        db.Put("user:3", "...prefix:actual_data");
        PinnableView val4;
        db.Get("user:3", &val4);
        std::cout << "  raw:     \"" << val4 << "\"\n";
        val4.remove_prefix(9);
        std::cout << "  trimmed: \"" << val4 << "\"  (zero-copy, pinned mode)\n";
        std::cout << "  IsPinned: " << val4.IsPinned() << "\n";
    }

    std::cout << "\n--- Manual PinSlice with raw cleanup fn ---\n";
    {
        PinnableView val5;
        char* heap_buf = new char[6]{'h', 'e', 'l', 'l', 'o', '\0'};
        std::cout << "  heap_buf addr: " << static_cast<const void*>(heap_buf) << "\n";

        val5.PinSlice(
            StringView(heap_buf, 5),
            [](void* arg1, void*) { std::cout << "  (cleanup: delete[] heap_buf)\n"; delete[] static_cast<char*>(arg1); },
            heap_buf, nullptr);

        std::cout << "  val.data addr: " << static_cast<const void*>(val5.data()) << "\n";
        std::cout << "  same? " << (val5.data() == heap_buf ? "YES" : "no") << "\n";
        std::cout << "  value: \"" << val5 << "\"\n";
        std::cout << "  (PinnableView going out of scope...)\n";
    }

    std::cout << "\n====== Summary ======\n";
    std::cout << "Block allocs:   " << s_alloc_count << "\n";
    std::cout << "Block frees:    " << s_free_count << "\n";
    std::cout << "Remaining:      " << (s_alloc_count - s_free_count) << "\n";
}
