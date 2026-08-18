#include <iostream>
#include <algorithm>
#include "graph/graph.h"
#include "graph/gradients.h"
#include "session.h"

using namespace lf;

int main() {
    std::cout << "== 测试跨设备训练（Send/Recv 梯度验证）==" << std::endl;

    // 生成简单的线性回归数据: y = 2*x + 1
    const int N = 100;
    std::vector<float> x_data(N), y_data(N);
    for (int i = 0; i < N; i++) {
        x_data[i] = static_cast<float>(i) / N - 0.5f;
        y_data[i] = 2.0f * x_data[i] + 1.0f;
    }

    // ========== 测试 1: 全 CPU 训练（基准）==========
    std::cout << "\n[测试 1] 全 CPU 训练（基准）" << std::endl;
    {
        Graph g;
        Node* x = g.placeholder("x", {N});
        Node* y = g.placeholder("y", {N});
        Node* a = g.variable("a", 0.0f);  // 应该收敛到 2.0
        Node* b = g.variable("b", 0.0f);  // 应该收敛到 1.0

        Node* y_pred = g.add(g.mul(x, a), b);
        Node* diff = g.sub(y_pred, y);
        Node* loss = g.mean(g.square(diff));

        // 构建梯度
        auto grads = build_gradients(g, loss);
        Node* train_a = g.sgd_step(a, grads[a], 0.5f);
        Node* train_b = g.sgd_step(b, grads[b], 0.5f);

        Session sess;
        std::unordered_map<const Node*, Tensor> feeds = {
            {x, Tensor(x_data, {N})}, {y, Tensor(y_data, {N})}};

        // 初始化变量（第一次 run 会初始化）
        sess.run(g, {train_a, train_b}, feeds);

        std::cout << "  训练前: a=" << sess.var_value(a).data[0]
                  << " b=" << sess.var_value(b).data[0] << std::endl;

        // 训练 200 步
        for (int epoch = 0; epoch < 200; epoch++) {
            sess.run(g, {train_a, train_b}, feeds);
            if (epoch % 50 == 0 || epoch == 199) {
                auto lo = sess.run(g, {loss}, feeds);
                std::cout << "  epoch " << epoch << ": loss=" << lo[0].data[0] << std::endl;
            }
        }

        float final_a = sess.var_value(a).data[0];
        float final_b = sess.var_value(b).data[0];
        std::cout << "  训练后: a=" << final_a << " b=" << final_b << std::endl;

        bool converged = std::fabs(final_a - 2.0f) < 0.1f && std::fabs(final_b - 1.0f) < 0.1f;
        std::cout << "  收敛检查 (a≈2, b≈1): " << (converged ? "✓ PASS" : "✗ FAIL") << std::endl;
    }

    // ========== 测试 2: 跨设备训练（变量在 GPU）==========
    std::cout << "\n[测试 2] 跨设备训练（变量 a 在 GPU）" << std::endl;
    {
        Graph g;
        Node* x = g.placeholder("x", {N});      // CPU
        Node* y = g.placeholder("y", {N});      // CPU
        Node* a = g.variable("a", 0.0f);        // GPU (显式)
        Node* b = g.variable("b", 0.0f);        // CPU (默认)

        // 显式设备放置
        g.on(a, Device::GPU);

        Node* mul = g.mul(x, a);                // x*a: 跨设备 (x在CPU, a在GPU)
        Node* y_pred = g.add(mul, b);           // mul+b: 跨设备 (mul在GPU, b在CPU)
        Node* diff = g.sub(y_pred, y);
        Node* loss = g.mean(g.square(diff));

        std::cout << "  原始图: " << g.nodes().size() << " nodes" << std::endl;

        // 构建梯度
        auto grads = build_gradients(g, loss);
        Node* train_a = g.sgd_step(a, grads[a], 0.5f);
        Node* train_b = g.sgd_step(b, grads[b], 0.5f);

        std::cout << "  梯度图: " << g.nodes().size() << " nodes" << std::endl;

        Session sess;
        sess.SetWorkers(2);  // 启用多设备
        std::unordered_map<const Node*, Tensor> feeds = {
            {x, Tensor(x_data, {N})}, {y, Tensor(y_data, {N})}};

        // 初始化变量（第一次 run 会初始化）
        sess.run(g, {train_a, train_b}, feeds);

        std::cout << "  训练初始: a=" << sess.var_value(a).data[0]
                  << " b=" << sess.var_value(b).data[0] << std::endl;

        std::cout << "  分区后图: " << g.nodes().size() << " nodes" << std::endl;

        // 训练 200 步
        for (int epoch = 0; epoch < 200; epoch++) {
            sess.run(g, {train_a, train_b}, feeds);
            if (epoch % 50 == 0 || epoch == 199) {
                auto lo = sess.run(g, {loss}, feeds);
                std::cout << "  epoch " << epoch << ": loss=" << lo[0].data[0] << std::endl;
            }
        }

        float final_a = sess.var_value(a).data[0];
        float final_b = sess.var_value(b).data[0];
        std::cout << "  训练后: a=" << final_a << " b=" << final_b << std::endl;

        bool converged = std::fabs(final_a - 2.0f) < 0.1f && std::fabs(final_b - 1.0f) < 0.1f;
        std::cout << "  收敛检查 (a≈2, b≈1): " << (converged ? "✓ PASS" : "✗ FAIL") << std::endl;

        if (!converged) {
            std::cout << "  ⚠️  跨设备训练失败 - 可能是 Send/Recv 梯度缺失" << std::endl;
        }
    }

    // ========== 测试 3: 打印分区后的图结构 ==========
    std::cout << "\n[测试 3] 打印跨设备图结构（检查 Send/Recv）" << std::endl;
    {
        Graph g;
        Node* x = g.placeholder("x", {N});
        Node* y = g.placeholder("y", {N});
        Node* a = g.variable("a", 0.0f);
        Node* b = g.variable("b", 0.0f);

        g.on(a, Device::GPU);

        Node* mul = g.mul(x, a);
        Node* y_pred = g.add(mul, b);
        Node* diff = g.sub(y_pred, y);
        Node* loss = g.mean(g.square(diff));

        auto grads = build_gradients(g, loss);
        Node* train_a = g.sgd_step(a, grads[a], 0.5f);
        Node* train_b = g.sgd_step(b, grads[b], 0.5f);

        Session sess;
        sess.SetWorkers(2);
        std::unordered_map<const Node*, Tensor> feeds = {
            {x, Tensor(x_data, {N})}, {y, Tensor(y_data, {N})}};

        // 触发分区
        sess.run(g, {train_a, train_b}, feeds);

        std::cout << "\n  图节点列表:" << std::endl;
        int send_count = 0, recv_count = 0;
        for (const auto& un : g.nodes()) {
            Node* n = un.get();
            const char* dev = (n->device == Device::GPU)   ? "GPU"
                              : (n->device == Device::CPU) ? "CPU"
                                                           : "AUTO";
            std::cout << "    " << n->name << " (" << dev << ")";
            if (n->type == SEND) {
                std::cout << " [SEND key=" << n->rendezvous_key << "]";
                send_count++;
            } else if (n->type == RECV) {
                std::cout << " [RECV key=" << n->rendezvous_key << "]";
                recv_count++;
            }
            std::cout << std::endl;
        }

        std::cout << "\n  统计: " << send_count << " Send 节点, "
                  << recv_count << " Recv 节点" << std::endl;
    }

    std::cout << "\n== 测试完成 ==" << std::endl;
    return 0;
}
