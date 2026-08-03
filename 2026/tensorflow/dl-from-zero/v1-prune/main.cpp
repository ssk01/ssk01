#include <iostream>
#include <random>
#include <vector>
#include <numeric>
#include <unordered_map>
#include "framework/tensor.h"
#include "graph/graph.h"
#include "graph/prune.h"
#include "kernels/kernels.h"
#include "session.h"

using namespace lf;

int main() {
    const int N = 100;
    const float true_a = 2.0f;
    const float true_b = 3.0f;
    const int epochs = 200;
    const float learning_rate = 0.01f;

    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 0.5f);
    std::uniform_real_distribution<float> x_dist(-5.0f, 5.0f);

    std::vector<float> x_data(N), y_data(N);
    for (int i = 0; i < N; i++) {
        x_data[i] = x_dist(rng);
        y_data[i] = true_a * x_data[i] + true_b + noise(rng);
    }

    Tensor tf_x(x_data, {N});
    Tensor tf_y(y_data, {N});

    // ---- 训练图: 完整计算图(含 loss 子图) ----
    Graph g;
    auto x = g.placeholder("x", {N});
    auto y = g.placeholder("y", {N});
    auto a = g.variable("a", 0.0f);
    auto b = g.variable("b", 0.0f);
    auto y_pred = g.add(g.mul(x, a), b);
    auto diff = g.sub(y_pred, y);
    auto loss = g.mean(g.square(diff));

    std::cout << "training graph: " << g.nodes().size() << " nodes" << std::endl;

    Session sess;
    std::unordered_map<Node*, const Tensor*> train_feeds = {{x, &tf_x}, {y, &tf_y}};

    for (int epoch = 0; epoch < epochs; epoch++) {
        sess.run(g, {loss}, train_feeds, loss);

        float grad_a = a->grad.data[0];
        float grad_b = b->grad.data[0];
        a->output = Tensor(a->output.data[0] - learning_rate * grad_a);
        b->output = Tensor(b->output.data[0] - learning_rate * grad_b);
    }
    std::cout << "trained: a=" << a->output.data[0]
              << "  b=" << b->output.data[0] << std::endl;

    // ---- 图裁剪: 训练图 → 推理图 ----
    // 只保留能算出 y_pred 的节点, loss 子图(diff/square/mean/y)全部剪掉
    std::cout << "\n--- Prune: train graph -> inference graph ---" << std::endl;
    std::cout << "before: " << g.nodes().size() << " nodes" << std::endl;
    prune(g, {y_pred});
    std::cout << "after : " << g.nodes().size() << " nodes" << std::endl;
    for (auto& n : g.nodes()) std::cout << "  keep: " << n->name << std::endl;

    // ---- 推理: 只需要 x,y 已被剪掉 ----
    std::cout << "\n--- Inference on pruned graph ---" << std::endl;
    std::vector<float> test_x = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f, 5.0f};
    for (float tx : test_x) {
        Tensor tx_t(tx);
        std::unordered_map<Node*, const Tensor*> feeds = {{x, &tx_t}};
        sess.run(g, {y_pred}, feeds);
        std::cout << "x=" << tx
                  << "  y_pred=" << y_pred->output.data[0]
                  << "  y_true=" << (true_a * tx + true_b) << std::endl;
    }

    std::cout << "\nFinal parameters: a=" << a->output.data[0]
              << " (true=" << true_a << "), b=" << b->output.data[0]
              << " (true=" << true_b << ")" << std::endl;

    return 0;
}
