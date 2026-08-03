#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>
#include "framework/tensor.h"
#include "graph/gradients.h"
#include "graph/graph.h"
#include "graph/prune.h"
#include "kernels/kernels.h"
#include "session.h"

using namespace lf;

int main() {
    const int N = 100;
    const float true_a = 2.0f, true_b = 3.0f;
    const int epochs = 200;
    const float lr = 0.01f;

    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 0.5f);
    std::uniform_real_distribution<float> x_dist(-5.0f, 5.0f);
    std::vector<float> x_data(N), y_data(N);
    for (int i = 0; i < N; i++) {
        x_data[i] = x_dist(rng);
        y_data[i] = true_a * x_data[i] + true_b + noise(rng);
    }

    // ---- 前向图 ----
    Graph g;
    auto x = g.placeholder("x", {N});
    auto y = g.placeholder("y", {N});
    auto a = g.variable("a", 0.0f);
    auto b = g.variable("b", 0.0f);
    auto y_pred = g.add(g.mul(x, a), b);
    auto diff = g.sub(y_pred, y);
    auto loss = g.mean(g.square(diff));

    // ---- 独立的梯度子图: 梯度变成图里的节点 (对应 TF gradients.py) ----
    auto grads = build_gradients(g, loss);
    auto train_a = g.sgd_step(a, grads[a], lr);   // 优化器消费梯度子图的输出
    auto train_b = g.sgd_step(b, grads[b], lr);
    std::cout << "graph: " << g.nodes().size()
              << " nodes (forward + gradient subgraph + sgd)" << std::endl;

    // ---- 训练: fetch sgd_step 即触发 正向+梯度子图+应用 ----
    Session sess;
    std::unordered_map<const Node*, Tensor> feeds = {
        {x, Tensor(x_data, {N})}, {y, Tensor(y_data, {N})}};
    for (int epoch = 0; epoch < epochs; epoch++) {
        sess.run(g, {train_a, train_b}, feeds);
        if (epoch % 40 == 0 || epoch == epochs - 1) {
            auto lo = sess.run(g, {loss}, feeds);
            std::cout << "epoch " << epoch << ": loss=" << lo[0].data[0] << std::endl;
        }
    }
    std::cout << "trained: a=" << sess.var_value(a).data[0]
              << "  b=" << sess.var_value(b).data[0] << std::endl;

    // ---- prune: 梯度子图 + loss + sgd 不可达, 全被剪掉 (推理图只剩 y_pred 路径) ----
    prune(g, {y_pred});
    std::cout << "pruned : " << g.nodes().size() << " nodes" << std::endl;

    // ---- 并发推理: 同一张图被多个线程同时跑 ----
    const int n_threads = 8, iters = 5000;
    std::vector<std::thread> threads;
    std::vector<double> mean_err(n_threads, 0.0);
    for (int t = 0; t < n_threads; t++) {
        threads.emplace_back([&, t]() {
            float tx = -4.0f + t;
            std::unordered_map<const Node*, Tensor> feed = {{x, Tensor(tx)}};
            double err = 0.0;
            for (int i = 0; i < iters; i++) {
                auto r = sess.run(g, {y_pred}, feed);
                err += std::fabs(r[0].data[0] - (true_a * tx + true_b));
            }
            mean_err[t] = err / iters;
        });
    }
    for (auto& th : threads) th.join();

    std::cout << "\nconcurrent inference: " << n_threads << " threads x " << iters
              << " runs on the same graph:" << std::endl;
    for (int t = 0; t < n_threads; t++) {
        float tx = -4.0f + t;
        std::cout << "  thread " << t << "  x=" << tx
                  << "  mean_abs_err=" << mean_err[t] << std::endl;
    }
    return 0;
}
