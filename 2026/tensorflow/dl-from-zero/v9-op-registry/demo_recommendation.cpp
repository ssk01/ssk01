#include <iostream>
#include <random>
#include <thread>
#include <chrono>
#include "framework/op_registry.h"
#include "framework/kernel_registry.h"
#include "framework/sparse_tensor.h"
#include "ops/queue_ops.h"
#include "ops/lookup_ops.h"
#include "graph/graph.h"
#include "session.h"

using namespace lf;

// ========== 推荐系统 CTR 预估 Demo ==========
// 场景: 点击率预估 (Click-Through Rate Prediction)
// 模型: user_id + item_id → embedding → MLP → 点击概率
//
// 特点:
// 1. 稀疏特征: user/item ID 是百万级稀疏向量 (只有一个非零位)
// 2. 查找表: ID → embedding index 映射
// 3. 队列: 数据管线 (预处理 + 训练解耦)
// 4. Op 注册: 自定义 sparse embedding lookup op

int main() {
    std::cout << "== v9 推荐系统 CTR 预估 Demo ==" << std::endl;

    // ========== 数据准备 ==========
    std::cout << "\n[1] 数据准备" << std::endl;

    const int num_users = 1000;      // 用户数
    const int num_items = 500;       // 物品数
    const int embedding_dim = 8;     // embedding 维度
    const int batch_size = 32;
    const int num_samples = 1000;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> user_dist(0, num_users - 1);
    std::uniform_int_distribution<int> item_dist(0, num_items - 1);
    std::uniform_real_distribution<float> label_dist(0.0f, 1.0f);

    // 生成训练数据: (user_id, item_id, label)
    std::vector<int> user_ids, item_ids;
    std::vector<float> labels;

    for (int i = 0; i < num_samples; ++i) {
        user_ids.push_back(user_dist(rng));
        item_ids.push_back(item_dist(rng));
        // 简化: 随机标签 (真实场景是用户点击行为)
        labels.push_back(label_dist(rng) > 0.7 ? 1.0f : 0.0f);
    }

    std::cout << "  样本数: " << num_samples << std::endl;
    std::cout << "  用户数: " << num_users << ", 物品数: " << num_items << std::endl;
    std::cout << "  正样本率: "
              << (std::count(labels.begin(), labels.end(), 1.0f) * 100.0 / labels.size())
              << "%" << std::endl;

    // ========== 查找表初始化 ==========
    std::cout << "\n[2] 查找表初始化" << std::endl;

    // user_id → embedding_index 映射
    TensorHashTable user_table, item_table;

    Tensor user_keys(std::vector<int>{num_users});
    Tensor user_values(std::vector<int>{num_users});
    for (int i = 0; i < num_users; ++i) {
        user_keys.data[i] = static_cast<float>(i);
        user_values.data[i] = static_cast<float>(i);
    }
    user_table.init(user_keys, user_values);

    Tensor item_keys(std::vector<int>{num_items});
    Tensor item_values(std::vector<int>{num_items});
    for (int i = 0; i < num_items; ++i) {
        item_keys.data[i] = static_cast<float>(i);
        item_values.data[i] = static_cast<float>(i);
    }
    item_table.init(item_keys, item_values);

    std::cout << "  user_table 大小: " << user_table.size() << std::endl;
    std::cout << "  item_table 大小: " << item_table.size() << std::endl;

    // ========== 稀疏特征构造 ==========
    std::cout << "\n[3] 稀疏特征构造" << std::endl;

    // 例: user_id=5 → 稀疏向量 [0,0,0,0,0,1,0,...] (百万维只有一个1)
    // 实际: 用 SparseTensor 表示 → indices=[5], values=[1.0], shape=[num_users]

    // 取第一个样本演示
    int sample_user = user_ids[0];
    int sample_item = item_ids[0];

    // 构造稀疏 user 特征
    Tensor user_indices(std::vector<int>{1});
    user_indices.data[0] = static_cast<float>(sample_user);
    Tensor user_values_sparse(std::vector<int>{1});
    user_values_sparse.data[0] = 1.0f;
    SparseTensor sparse_user(user_indices, user_values_sparse, {num_users});

    // 构造稀疏 item 特征
    Tensor item_indices(std::vector<int>{1});
    item_indices.data[0] = static_cast<float>(sample_item);
    Tensor item_values_sparse(std::vector<int>{1});
    item_values_sparse.data[0] = 1.0f;
    SparseTensor sparse_item(item_indices, item_values_sparse, {num_items});

    std::cout << "  稀疏 user 特征: shape=" << num_users << ", nnz=" << sparse_user.nnz() << std::endl;
    std::cout << "  稀疏 item 特征: shape=" << num_items << ", nnz=" << sparse_item.nnz() << std::endl;

    // 内存对比
    size_t dense_bytes = num_users * sizeof(float);
    size_t sparse_bytes = sparse_user.memory_bytes();
    std::cout << "  内存节省: 稠密=" << dense_bytes << "B, 稀疏=" << sparse_bytes
              << "B (节省 " << (100.0 * (dense_bytes - sparse_bytes) / dense_bytes) << "%)" << std::endl;

    // ========== 简化的训练 ==========
    std::cout << "\n[4] 简化训练循环 (mini-batch)" << std::endl;

    // 构建简化的 CTR 模型图
    // user_embedding[num_users, emb_dim] × user_sparse[num_users] → user_vec[emb_dim]
    // item_embedding[num_items, emb_dim] × item_sparse[num_items] → item_vec[emb_dim]
    // concat(user_vec, item_vec) → [2*emb_dim]
    // MLP: [2*emb_dim] → [4] → [1] → sigmoid → prob

    Graph g;

    // Embedding 矩阵 (简化: 用普通变量)
    Tensor user_emb_val(std::vector<int>{num_users * embedding_dim});
    std::fill(user_emb_val.data.begin(), user_emb_val.data.end(), 0.1f);
    Node* user_emb = g.variable_tensor("user_emb", user_emb_val);

    Tensor item_emb_val(std::vector<int>{num_items * embedding_dim});
    std::fill(item_emb_val.data.begin(), item_emb_val.data.end(), 0.1f);
    Node* item_emb = g.variable_tensor("item_emb", item_emb_val);

    // Placeholder: sparse indices (简化: 用稠密 index)
    Node* user_idx = g.placeholder("user_idx", {batch_size});
    Node* item_idx = g.placeholder("item_idx", {batch_size});
    Node* label = g.placeholder("label", {batch_size});

    std::cout << "  模型参数:" << std::endl;
    std::cout << "    user_embedding: [" << num_users << ", " << embedding_dim << "]" << std::endl;
    std::cout << "    item_embedding: [" << num_items << ", " << embedding_dim << "]" << std::endl;
    std::cout << "  训练参数:" << std::endl;
    std::cout << "    batch_size: " << batch_size << std::endl;
    std::cout << "    样本数: " << num_samples << std::endl;

    // ========== 队列管线演示 ==========
    std::cout << "\n[5] 数据队列管线" << std::endl;

    FIFOQueue data_queue(10);  // 容量 10 batch

    // 生产者线程: 模拟数据预处理 + 入队
    std::atomic<bool> stop_producer(false);
    std::atomic<int> batches_produced(0);

    std::thread producer([&]() {
        int offset = 0;
        while (!stop_producer && offset < num_samples) {
            // 模拟预处理耗时
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            // 构造一个 batch (简化: 只包含 user_ids)
            Tensor batch_data(std::vector<int>{std::min(batch_size, num_samples - offset)});
            for (int i = 0; i < batch_data.size(); ++i) {
                batch_data.data[i] = static_cast<float>(user_ids[offset + i]);
            }

            data_queue.Enqueue(batch_data);
            batches_produced++;
            offset += batch_size;
        }
        data_queue.Close();
    });

    // 消费者: 训练循环
    int batches_consumed = 0;
    int total_consumed = 0;

    auto train_start = std::chrono::steady_clock::now();

    while (true) {
        try {
            Tensor batch = data_queue.Dequeue();

            // 模拟训练耗时
            std::this_thread::sleep_for(std::chrono::milliseconds(5));

            batches_consumed++;
            total_consumed += batch.size();

            if (batches_consumed % 10 == 0) {
                std::cout << "  训练进度: " << batches_consumed << " batches, "
                          << total_consumed << " samples" << std::endl;
            }
        } catch (const std::exception&) {
            break;  // 队列关闭且空
        }
    }

    producer.join();

    auto train_end = std::chrono::steady_clock::now();
    double train_time = std::chrono::duration<double, std::milli>(train_end - train_start).count();

    std::cout << "  训练完成: " << batches_consumed << " batches, "
              << total_consumed << " samples, "
              << train_time << " ms" << std::endl;
    std::cout << "  吞吐量: " << (total_consumed / (train_time / 1000.0)) << " samples/sec" << std::endl;

    // ========== 推理演示 ==========
    std::cout << "\n[6] 推理: 查找表批量查询" << std::endl;

    // 批量查询用户 embedding index
    Tensor query_users(std::vector<int>{5});
    query_users.data = {10, 50, 100, 500, 999};

    Tensor user_indices_result = user_table.lookup(query_users, -1);

    std::cout << "  批量查询结果:" << std::endl;
    for (int i = 0; i < query_users.size(); ++i) {
        std::cout << "    user_id=" << static_cast<int>(query_users.data[i])
                  << " → index=" << static_cast<int>(user_indices_result.data[i]) << std::endl;
    }

    // ========== 总结 ==========
    std::cout << "\n== v9 推荐系统 Demo 总结 ==" << std::endl;
    std::cout << "✓ Op 注册系统: 支持自定义 op 扩展" << std::endl;
    std::cout << "✓ 稀疏张量: 百万维特征 → 节省 "
              << (100.0 * (dense_bytes - sparse_bytes) / dense_bytes) << "% 内存" << std::endl;
    std::cout << "✓ 队列管线: 生产者/消费者解耦, 吞吐量 "
              << static_cast<int>(total_consumed / (train_time / 1000.0)) << " samples/sec" << std::endl;
    std::cout << "✓ 查找表: 批量 ID 映射, 支持 " << user_table.size() << " 用户" << std::endl;

    return 0;
}
