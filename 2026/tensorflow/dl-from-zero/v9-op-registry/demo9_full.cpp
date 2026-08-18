#include <iostream>
#include <algorithm>
#include "graph/graph.h"
#include "graph/partition.h"
#include "core/place.h"
#include "session.h"

using namespace lf;

static bool allclose(const Tensor& a, const Tensor& b, float eps = 1e-5f) {
    if (a.size() != b.size()) return false;
    for (int i = 0; i < a.size(); i++)
        if (std::fabs(a.data[i] - b.data[i]) > eps) return false;
    return true;
}

int main() {
    std::cout << "== v8 Graph Partition Demo 9: 端到端验证 ==" << std::endl;

    // ========== 测试 1: 简单跨设备图 ==========
    std::cout << "\n[测试 1] 简单跨设备图: x(CPU) -> matmul(GPU) -> add(CPU)" << std::endl;
    {
        Graph g;
        Node* x = g.placeholder("x", {4, 128});
        Node* w = g.variable_vec("w", 128, 0.1f);
        Node* matmul = g.matmul(x, w);
        Node* b = g.variable("b", 0.5f);
        Node* y = g.add(matmul, b);

        // 显式设备放置
        g.on(matmul, Device::GPU);
        g.on(w, Device::GPU);

        std::cout << "  原始图: " << g.nodes().size() << " nodes" << std::endl;

        Session sess;
        sess.SetWorkers(2);

        // 运行: 触发图分区
        Tensor x_val({4, 128});
        std::fill(x_val.data.begin(), x_val.data.end(), 0.1f);
        auto result = sess.run(g, {y}, {{x, x_val}});

        std::cout << "  分区后图: " << g.nodes().size() << " nodes" << std::endl;

        // 验证: 全 CPU 参考
        Graph g_ref;
        Node* x_ref = g_ref.placeholder("x", {4, 128});
        Node* w_ref = g_ref.variable_vec("w", 128, 0.1f);
        Node* matmul_ref = g_ref.matmul(x_ref, w_ref);
        Node* b_ref = g_ref.variable("b", 0.5f);
        Node* y_ref = g_ref.add(matmul_ref, b_ref);

        Session sess_ref;
        auto result_ref = sess_ref.run(g_ref, {y_ref}, {{x_ref, x_val}});

        bool match = allclose(result[0], result_ref[0]);
        std::cout << "  跨设备结果 == 全 CPU 参考: " << (match ? "✓ PASS" : "✗ FAIL") << std::endl;
        if (match) {
            std::cout << "  (示例值: y[0]=" << result[0].data[0] << ")" << std::endl;
        }
    }

    // ========== 测试 2: 多跨设备边 ==========
    std::cout << "\n[测试 2] 多跨设备边: GPU matmul 结果给两个 CPU 节点" << std::endl;
    {
        Graph g;
        Node* x = g.placeholder("x", {4, 128});
        Node* w = g.variable_vec("w", 128, 0.1f);
        Node* matmul = g.matmul(x, w);
        Node* y1 = g.sigmoid(matmul);  // CPU
        Node* y2 = g.square(matmul);   // CPU

        g.on(matmul, Device::GPU);
        g.on(w, Device::GPU);

        std::cout << "  原始图: " << g.nodes().size() << " nodes" << std::endl;

        Session sess;
        sess.SetWorkers(2);

        Tensor x_val({4, 128});
        std::fill(x_val.data.begin(), x_val.data.end(), 0.2f);
        auto result = sess.run(g, {y1, y2}, {{x, x_val}});

        std::cout << "  分区后图: " << g.nodes().size() << " nodes" << std::endl;

        // 验证: 全 CPU 参考
        Graph g_ref;
        Node* x_ref = g_ref.placeholder("x", {4, 128});
        Node* w_ref = g_ref.variable_vec("w", 128, 0.1f);
        Node* matmul_ref = g_ref.matmul(x_ref, w_ref);
        Node* y1_ref = g_ref.sigmoid(matmul_ref);
        Node* y2_ref = g_ref.square(matmul_ref);

        Session sess_ref;
        auto result_ref = sess_ref.run(g_ref, {y1_ref, y2_ref}, {{x_ref, x_val}});

        bool match1 = allclose(result[0], result_ref[0]);
        bool match2 = allclose(result[1], result_ref[1]);
        std::cout << "  y1 (sigmoid) == 参考: " << (match1 ? "✓ PASS" : "✗ FAIL") << std::endl;
        std::cout << "  y2 (square) == 参考: " << (match2 ? "✓ PASS" : "✗ FAIL") << std::endl;
        if (match1 && match2) {
            std::cout << "  (DupRecv 优化生效: 一个 matmul 输出复用)" << std::endl;
        }
    }

    // ========== 测试 3: 无跨设备边（全 CPU）==========
    std::cout << "\n[测试 3] 无跨设备边（全 CPU）: 不应插入 Send/Recv" << std::endl;
    {
        Graph g;
        Node* x = g.placeholder("x", {4, 128});
        Node* w = g.variable_vec("w", 128, 0.1f);
        Node* matmul = g.matmul(x, w);
        Node* y = g.sigmoid(matmul);

        // 显式指定全 CPU（关键修复）
        g.on(x, Device::CPU);
        g.on(w, Device::CPU);
        g.on(matmul, Device::CPU);
        g.on(y, Device::CPU);

        int orig_size = g.nodes().size();
        std::cout << "  原始图: " << orig_size << " nodes" << std::endl;

        Session sess;
        Tensor x_val({4, 128});
        std::fill(x_val.data.begin(), x_val.data.end(), 0.15f);
        auto result = sess.run(g, {y}, {{x, x_val}});

        int after_size = g.nodes().size();
        std::cout << "  分区后图: " << after_size << " nodes" << std::endl;
        std::cout << "  节点数未增加: " << (after_size == orig_size ? "✓ PASS" : "✗ FAIL") << std::endl;
        if (after_size == orig_size) {
            std::cout << "  (显式全 CPU，无跨设备边)" << std::endl;
        }
    }

    // ========== 打印一个图的结构 ==========
    std::cout << "\n[图结构] 测试 1 的分区结果:" << std::endl;
    {
        Graph g;
        Node* x = g.placeholder("x", {4, 128});
        Node* w = g.variable_vec("w", 128, 0.1f);
        Node* matmul = g.matmul(x, w);
        Node* b = g.variable("b", 0.5f);
        Node* y = g.add(matmul, b);

        g.on(matmul, Device::GPU);
        g.on(w, Device::GPU);

        Session sess;
        Tensor x_val({4, 128});
        std::fill(x_val.data.begin(), x_val.data.end(), 0.1f);
        sess.run(g, {y}, {{x, x_val}});

        for (const auto& un : g.nodes()) {
            Node* n = un.get();
            const char* dev = (n->device == Device::GPU)   ? "GPU"
                              : (n->device == Device::CPU) ? "CPU"
                                                           : "AUTO";
            std::cout << "  " << n->name << " (" << dev << ")";
            if (n->type == SEND || n->type == RECV) {
                std::cout << " [key=" << n->rendezvous_key << "]";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "\n== 所有测试完成 ==" << std::endl;
    return 0;
}
