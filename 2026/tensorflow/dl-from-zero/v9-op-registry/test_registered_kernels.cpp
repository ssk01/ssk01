#include <iostream>
#include <cmath>
#include "graph/graph.h"
#include "graph/gradients.h"
#include "kernels/registered_kernels.h"
#include "session.h"

using namespace lf;

int main() {
    std::cout << "== 测试 Op 注册系统（新 vs 旧执行引擎）==" << std::endl;

    // 生成测试数据
    const int N = 100;
    std::vector<float> x_data(N), y_data(N);
    for (int i = 0; i < N; i++) {
        x_data[i] = static_cast<float>(i) / N - 0.5f;
        y_data[i] = 2.0f * x_data[i] + 1.0f;
    }

    // ========== 测试 1: 基本算子验证 ==========
    std::cout << "\n[测试 1] 基本算子（ADD/MUL/MATMUL）" << std::endl;
    {
        Graph g;
        Node* a = g.constant("a", Tensor(2.0f));
        Node* b = g.constant("b", Tensor(3.0f));
        Node* c = g.add(a, b);      // 2 + 3 = 5
        Node* d = g.mul(c, a);      // 5 * 2 = 10

        RunState st;
        st.resize(g.id_count());

        // 使用注册系统执行
        forward_registered(a, st);
        forward_registered(b, st);
        forward_registered(c, st);
        forward_registered(d, st);

        float result = st.out(d).data[0];
        bool pass = std::fabs(result - 10.0f) < 1e-5f;

        std::cout << "  结果: " << result << " (期望 10.0)" << std::endl;
        std::cout << "  状态: " << (pass ? "✓ PASS" : "✗ FAIL") << std::endl;
    }

    // ========== 测试 2: 线性回归训练（完整流程）==========
    std::cout << "\n[测试 2] 线性回归训练（y = 2x + 1）" << std::endl;
    {
        Graph g;
        Node* x = g.placeholder("x", {N});
        Node* y = g.placeholder("y", {N});
        Node* a = g.variable("a", 0.0f);
        Node* b = g.variable("b", 0.0f);

        Node* y_pred = g.add(g.mul(x, a), b);
        Node* diff = g.sub(y_pred, y);
        Node* loss = g.mean(g.square(diff));

        auto grads = build_gradients(g, loss);
        Node* train_a = g.sgd_step(a, grads[a], 0.5f);
        Node* train_b = g.sgd_step(b, grads[b], 0.5f);

        std::cout << "  图节点数: " << g.nodes().size() << std::endl;
        std::cout << "  检查所有 op 是否注册..." << std::endl;

        // 检查所有节点是否都有注册的 kernel
        static const char* type_names[] = {
            "PLACEHOLDER", "VARIABLE", "CONST", "ADD", "MUL", "SUB", "SQUARE",
            "MEAN", "MATMUL", "SIGMOID", "LOG", "RECIP", "REDUCE_SUM", "SGD_STEP",
            "MEAN_GRAD", "MATMUL_GRAD_A", "MATMUL_GRAD_B", "SEND", "RECV"
        };

        bool all_registered = true;
        for (const auto& un : g.nodes()) {
            Node* n = un.get();
            if (n->type == PLACEHOLDER || n->type == SGD_STEP) continue;

            const char* op_name = type_names[n->type];
            const KernelDef* kernel = KernelRegistry::Global().LookUp(op_name, Device::CPU);

            if (!kernel) {
                std::cout << "  ✗ 缺少 kernel: " << op_name << std::endl;
                all_registered = false;
            }
        }

        if (all_registered) {
            std::cout << "  ✓ 所有 op 都已注册" << std::endl;
        }

        // 使用注册系统执行训练（手动模拟 Session.run）
        std::unordered_map<const Node*, Tensor> vars;
        vars[a] = Tensor(0.0f);
        vars[b] = Tensor(0.0f);

        RunState st;
        st.resize(g.id_count());
        st.vars = &vars;

        // 准备 feed
        st.out(x) = Tensor(x_data, {N});
        st.out(y) = Tensor(y_data, {N});

        // 训练 100 步
        for (int epoch = 0; epoch < 100; epoch++) {
            // 前向传播（使用注册系统）
            for (const auto& un : g.nodes()) {
                Node* n = un.get();
                if (n->type == PLACEHOLDER) continue;
                if (n->type == SGD_STEP) continue;

                forward_registered(n, st);
            }

            // 应用梯度（SGD_STEP）
            if (grads.count(a)) {
                float grad_a = st.out(grads[a]).data[0];
                vars[a].data[0] -= 0.5f * grad_a;
            }
            if (grads.count(b)) {
                float grad_b = st.out(grads[b]).data[0];
                vars[b].data[0] -= 0.5f * grad_b;
            }

            if (epoch % 25 == 0 || epoch == 99) {
                float loss_val = st.out(loss).data[0];
                std::cout << "  epoch " << epoch << ": loss=" << loss_val << std::endl;
            }
        }

        float final_a = vars[a].data[0];
        float final_b = vars[b].data[0];
        std::cout << "  训练结果: a=" << final_a << " b=" << final_b << std::endl;

        bool converged = std::fabs(final_a - 2.0f) < 0.1f && std::fabs(final_b - 1.0f) < 0.1f;
        std::cout << "  收敛检查: " << (converged ? "✓ PASS" : "✗ FAIL") << std::endl;
    }

    // ========== 测试 3: 列出所有注册的 kernel ==========
    std::cout << "\n[测试 3] 已注册的 kernel 列表" << std::endl;
    {
        static const char* type_names[] = {
            "PLACEHOLDER", "VARIABLE", "CONST", "ADD", "MUL", "SUB", "SQUARE",
            "MEAN", "MATMUL", "SIGMOID", "LOG", "RECIP", "REDUCE_SUM", "SGD_STEP",
            "MEAN_GRAD", "MATMUL_GRAD_A", "MATMUL_GRAD_B", "SEND", "RECV"
        };

        int count = 0;
        for (const char* op_name : type_names) {
            if (strcmp(op_name, "PLACEHOLDER") == 0 || strcmp(op_name, "SGD_STEP") == 0) {
                continue;
            }

            auto devices = KernelRegistry::Global().GetSupportedDevices(op_name);
            if (!devices.empty()) {
                std::cout << "  " << op_name << ": ";
                for (Device dev : devices) {
                    std::cout << (dev == Device::CPU ? "CPU" : "GPU") << " ";
                }
                std::cout << std::endl;
                count++;
            }
        }

        std::cout << "  总计: " << count << " 个 op 已注册" << std::endl;
    }

    std::cout << "\n== 测试完成 ==" << std::endl;
    return 0;
}
