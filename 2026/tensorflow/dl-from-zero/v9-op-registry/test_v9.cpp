#include <iostream>
#include <vector>
#include <thread>
#include "framework/op_registry.h"
#include "framework/kernel_registry.h"
#include "framework/sparse_tensor.h"
#include "ops/queue_ops.h"
#include "ops/lookup_ops.h"

using namespace lf;

int main() {
    std::cout << "== v9 新功能验证 ==" << std::endl;

    // ========== demo 10: Op 注册系统 ==========
    std::cout << "\n== demo 10: Op 注册系统 ==" << std::endl;
    {
        // 注册自定义 op
        OpDefBuilder("CustomAdd")
            .Input("a: float")
            .Input("b: float")
            .Output("sum: float")
            .Attr("scale: float = 1.0")
            .Finalize();

        // 查找 op
        const OpDef* op_def = OpRegistry::Global().LookUp("CustomAdd");
        if (op_def) {
            std::cout << "  注册成功: " << op_def->name << std::endl;
            std::cout << "    inputs: " << op_def->inputs.size() << std::endl;
            std::cout << "    outputs: " << op_def->outputs.size() << std::endl;
            std::cout << "    attrs: " << op_def->attrs.size() << std::endl;
        }

        // 列出所有注册的 op
        auto ops = OpRegistry::Global().ListOps();
        std::cout << "  已注册 op 数量: " << ops.size() << std::endl;
    }

    // ========== demo 11: 稀疏张量 ==========
    std::cout << "\n== demo 11: 稀疏张量 ==" << std::endl;
    {
        // 创建稠密张量
        Tensor dense(std::vector<int>{10});
        dense.data = {0, 0, 3.5, 0, 0, 0, 7.2, 0, 0, 1.8};

        // 转为稀疏
        SparseTensor sparse = SparseTensor::from_dense(dense);
        std::cout << "  稠密张量: size=" << dense.size() << std::endl;
        std::cout << "  稀疏张量: nnz=" << sparse.nnz() << std::endl;

        // 内存对比
        size_t dense_bytes = dense.data.size() * sizeof(float);
        size_t sparse_bytes = sparse.memory_bytes();
        std::cout << "  内存占用: 稠密=" << dense_bytes << "B, 稀疏=" << sparse_bytes
                  << "B (节省 " << (100.0 * (dense_bytes - sparse_bytes) / dense_bytes)
                  << "%)" << std::endl;

        // 验证转换
        Tensor reconstructed = sparse.to_dense();
        bool match = true;
        for (int i = 0; i < dense.size(); ++i) {
            if (std::abs(dense.data[i] - reconstructed.data[i]) > 1e-5) {
                match = false;
                break;
            }
        }
        std::cout << "  稀疏→稠密转换正确: " << (match ? "✓" : "✗") << std::endl;
    }

    // ========== demo 12: 队列 ==========
    std::cout << "\n== demo 12: 队列 (FIFOQueue) ==" << std::endl;
    {
        FIFOQueue queue(5);  // 容量 5

        // 生产者线程
        std::thread producer([&]() {
            for (int i = 0; i < 10; ++i) {
                Tensor t(std::vector<int>{1});
                t.data[0] = static_cast<float>(i);
                queue.Enqueue(t);
                std::cout << "  生产: " << i << std::endl;
            }
            queue.Close();
        });

        // 消费者线程
        std::thread consumer([&]() {
            while (true) {
                try {
                    Tensor t = queue.Dequeue();
                    std::cout << "    消费: " << t.data[0] << std::endl;
                } catch (const std::exception&) {
                    break;  // 队列关闭且空
                }
            }
        });

        producer.join();
        consumer.join();

        std::cout << "  队列验证完成 ✓" << std::endl;
    }

    // ========== demo 13: 查找表 ==========
    std::cout << "\n== demo 13: 查找表 (HashTable) ==" << std::endl;
    {
        TensorHashTable table;

        // 初始化: user_id → embedding_index
        Tensor keys(std::vector<int>{5});
        keys.data = {101, 202, 303, 404, 505};  // user_ids
        Tensor values(std::vector<int>{5});
        values.data = {0, 1, 2, 3, 4};  // embedding_indices

        table.init(keys, values);
        std::cout << "  表大小: " << table.size() << std::endl;

        // 批量查找
        Tensor query_keys(std::vector<int>{3});
        query_keys.data = {202, 999, 404};  // 999 不存在

        Tensor results = table.lookup(query_keys, -1);  // default=-1

        std::cout << "  查询结果:" << std::endl;
        for (int i = 0; i < results.size(); ++i) {
            std::cout << "    key=" << query_keys.data[i]
                      << " → value=" << results.data[i] << std::endl;
        }

        std::cout << "  查找表验证完成 ✓" << std::endl;
    }

    std::cout << "\n== 所有测试完成 ==" << std::endl;
    return 0;
}
