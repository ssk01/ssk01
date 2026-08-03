#include <iostream>
#include <random>
#include <vector>
#include <numeric>
#include <unordered_map>
#include "framework/tensor.h"
#include "graph/graph.h"
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

    Graph g;
    auto x = g.placeholder("x", {N});
    auto y = g.placeholder("y", {N});
    auto a = g.variable("a", 0.0f);
    auto b = g.variable("b", 0.0f);
    auto y_pred = g.add(g.mul(x, a), b);
    auto diff = g.sub(y_pred, y);
    auto loss = g.mean(g.square(diff));

    Session sess;
    std::unordered_map<Node*, const Tensor*> feeds = {{x, &tf_x}, {y, &tf_y}};

    std::cout << "Training y = a*x + b  (true: a=" << true_a
              << ", b=" << true_b << ")" << std::endl;
    std::cout << "Samples: " << N << ", Epochs: " << epochs
              << ", LR: " << learning_rate << std::endl;
    std::cout << "----------------------------------------------" << std::endl;

    for (int epoch = 0; epoch < epochs; epoch++) {
        sess.run(g, {loss}, feeds, loss);

        float grad_a = a->grad.data[0];
        float grad_b = b->grad.data[0];
        a->output = Tensor(a->output.data[0] - learning_rate * grad_a);
        b->output = Tensor(b->output.data[0] - learning_rate * grad_b);

        if (epoch % 20 == 0 || epoch == epochs - 1) {
            sess.run(g, {loss, y_pred}, feeds);
            std::cout << "epoch " << epoch
                      << ": loss=" << loss->output.data[0]
                      << "  a=" << a->output.data[0]
                      << "  b=" << b->output.data[0] << std::endl;
        }
    }

    std::cout << "\n--- Inference ---" << std::endl;
    std::vector<float> test_x = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f, 5.0f};
    float final_a = a->output.data[0];
    float final_b = b->output.data[0];

    for (float tx : test_x) {
        float y_pred_val = final_a * tx + final_b;
        std::cout << "x=" << tx
                  << "  y_pred=" << y_pred_val
                  << "  y_true=" << (true_a * tx + true_b) << std::endl;
    }

    std::cout << "\nFinal parameters: a=" << final_a
              << " (true=" << true_a << "), b=" << final_b
              << " (true=" << true_b << ")" << std::endl;

    return 0;
}
