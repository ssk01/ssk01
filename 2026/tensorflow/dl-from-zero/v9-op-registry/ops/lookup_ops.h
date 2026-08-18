#pragma once
#include <unordered_map>
#include <vector>
#include <mutex>
#include <stdexcept>
#include "../framework/tensor.h"

namespace lf {

// HashTable: 哈希表资源 (对应 TF LookupTable)
// 核心: 特征 ID 映射 (user_id → embedding_index)
//
// 用法:
//   HashTable table;
//   table.init(keys, values);  // 初始化
//   Tensor indices = table.lookup(ids);  // 批量查找
template <typename KeyType = int64_t, typename ValueType = int32_t>
class HashTable {
public:
    HashTable() = default;

    // 初始化查找表
    void init(const std::vector<KeyType>& keys, const std::vector<ValueType>& values) {
        if (keys.size() != values.size()) {
            throw std::runtime_error("keys and values size mismatch");
        }

        std::lock_guard<std::mutex> lock(mu_);
        table_.clear();
        for (size_t i = 0; i < keys.size(); ++i) {
            table_[keys[i]] = values[i];
        }
    }

    // 查找单个键
    ValueType lookup(KeyType key, ValueType default_value = -1) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = table_.find(key);
        return (it != table_.end()) ? it->second : default_value;
    }

    // 批量查找
    std::vector<ValueType> lookup_batch(const std::vector<KeyType>& keys,
                                       ValueType default_value = -1) const {
        std::lock_guard<std::mutex> lock(mu_);
        std::vector<ValueType> results;
        results.reserve(keys.size());

        for (KeyType key : keys) {
            auto it = table_.find(key);
            results.push_back((it != table_.end()) ? it->second : default_value);
        }

        return results;
    }

    // 插入/更新
    void insert(KeyType key, ValueType value) {
        std::lock_guard<std::mutex> lock(mu_);
        table_[key] = value;
    }

    // 删除
    void erase(KeyType key) {
        std::lock_guard<std::mutex> lock(mu_);
        table_.erase(key);
    }

    // 大小
    size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return table_.size();
    }

    // 清空
    void clear() {
        std::lock_guard<std::mutex> lock(mu_);
        table_.clear();
    }

private:
    mutable std::mutex mu_;
    std::unordered_map<KeyType, ValueType> table_;
};

// 简化版: Tensor 接口的 HashTable (int64 -> int32)
class TensorHashTable {
public:
    TensorHashTable() = default;

    // 从 Tensor 初始化
    void init(const Tensor& keys, const Tensor& values) {
        std::vector<int64_t> k_vec;
        std::vector<int32_t> v_vec;

        for (float f : keys.data) {
            k_vec.push_back(static_cast<int64_t>(f));
        }
        for (float f : values.data) {
            v_vec.push_back(static_cast<int32_t>(f));
        }

        table_.init(k_vec, v_vec);
    }

    // 批量查找，返回 Tensor
    Tensor lookup(const Tensor& keys_tensor, int32_t default_value = -1) const {
        std::vector<int64_t> keys;
        for (float f : keys_tensor.data) {
            keys.push_back(static_cast<int64_t>(f));
        }

        auto results = table_.lookup_batch(keys, default_value);

        Tensor output(keys_tensor.shape);
        for (size_t i = 0; i < results.size(); ++i) {
            output.data[i] = static_cast<float>(results[i]);
        }

        return output;
    }

    size_t size() const { return table_.size(); }

private:
    HashTable<int64_t, int32_t> table_;
};

}  // namespace lf
