#include <atomic>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>
#include "framework/tensor.h"
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

    // ---- 训练图: 含 loss 子图 + sgd_step 优化器节点 (TF: ApplyGradientDescent) ----
    Graph g;
    auto x = g.placeholder("x", {N});
    auto y = g.placeholder("y", {N});
    auto a = g.variable("a", 0.0f);
    auto b = g.variable("b", 0.0f);
    auto y_pred = g.add(g.mul(x, a), b);
    auto diff = g.sub(y_pred, y);
    auto loss = g.mean(g.square(diff));
    g.sgd_step(a, lr);
    g.sgd_step(b, lr);
    std::cout << "training graph: " << g.nodes().size() << " nodes" << std::endl;

    Session sess;
    std::unordered_map<const Node*, Tensor> feeds = {
        {x, Tensor(x_data, {N})}, {y, Tensor(y_data, {N})}};

    for (int epoch = 0; epoch < epochs; epoch++) {
        auto out = sess.run(g, {loss}, feeds, loss);   // forward + backward + 应用 sgd_step
        if (epoch % 40 == 0 || epoch == epochs - 1)
            std::cout << "epoch " << epoch << ": loss=" << out[0].data[0] << std::endl;
    }
    std::cout << "trained: a=" << sess.var_value(a).data[0]
              << "  b=" << sess.var_value(b).data[0] << std::endl;

    // ---- prune: 训练图 -> 推理图 (去掉 loss 子图和 sgd_step) ----
    prune(g, {y_pred});
    std::cout << "pruned : " << g.nodes().size() << " nodes" << std::endl;

    // ---- 并发推理: 同一张图被多个线程同时跑 ----
    // 图无状态, 每轮的值都在各自 run 的 RunState 里, 互不干扰
    const int n_threads = 8, iters = 5000;
    std::vector<std::thread> threads;
    std::vector<double> mean_err(n_threads, 0.0);
    std::vector<double> mean_lat(n_threads, 0.0);
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
