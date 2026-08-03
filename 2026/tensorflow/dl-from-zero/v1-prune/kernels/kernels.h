#pragma once
#include "../framework/tensor.h"
#include "../graph/graph.h"

namespace lf {

inline void forward(Node* node) {
    switch (node->type) {
    case PLACEHOLDER:
    case VARIABLE:
        break;
    case ADD:
        node->output = tensor_add(node->inputs[0]->output, node->inputs[1]->output);
        break;
    case MUL:
        node->output = tensor_mul(node->inputs[0]->output, node->inputs[1]->output);
        break;
    case SUB:
        node->output = tensor_sub(node->inputs[0]->output, node->inputs[1]->output);
        break;
    case SQUARE:
        node->output = tensor_square(node->inputs[0]->output);
        break;
    case MEAN:
        node->output = tensor_mean(node->inputs[0]->output);
        break;
    case MATMUL:
        node->output = tensor_matmul(node->inputs[0]->output, node->inputs[1]->output);
        break;
    case SIGMOID:
        node->output = tensor_sigmoid(node->inputs[0]->output);
        break;
    case LOG:
        node->output = tensor_log(node->inputs[0]->output);
        break;
    case SGD_STEP: {
        float lr = node->output.data[0];
        const Tensor& var = node->inputs[0]->output;
        const Tensor& grad = node->inputs[1]->output;
        Tensor new_val(var.shape);
        for (int i = 0; i < var.size(); i++)
            new_val.data[i] = var.data[i] - lr * grad.data[i];
        node->inputs[0]->output = new_val;
        break;
    }
    }
}

inline void backward(Node* node) {
    switch (node->type) {
    case PLACEHOLDER:
        break;
    case VARIABLE:
        break;
    case ADD: {
        auto& g = node->grad;
        tensor_add_to(node->inputs[0]->grad, g);
        tensor_add_to(node->inputs[1]->grad, g);
        break;
    }
    case MUL: {
        auto& g = node->grad;
        tensor_add_to(node->inputs[0]->grad,
                      tensor_mul(g, node->inputs[1]->output));
        tensor_add_to(node->inputs[1]->grad,
                      tensor_mul(g, node->inputs[0]->output));
        break;
    }
    case SUB: {
        auto& g = node->grad;
        tensor_add_to(node->inputs[0]->grad, g);
        tensor_add_to(node->inputs[1]->grad, tensor_mul_scalar(g, -1.0f));
        break;
    }
    case SQUARE: {
        auto& g = node->grad;
        const auto& x = node->inputs[0]->output;
        Tensor two_x = tensor_mul_scalar(x, 2.0f);
        tensor_add_to(node->inputs[0]->grad, tensor_mul(g, two_x));
        break;
    }
    case MEAN: {
        float n = static_cast<float>(node->inputs[0]->output.size());
        auto& g = node->grad;
        Tensor broadcasted(node->inputs[0]->output.shape);
        float g_scalar = g.data[0];
        for (int i = 0; i < static_cast<int>(n); i++) broadcasted.data[i] = g_scalar / n;
        tensor_add_to(node->inputs[0]->grad, broadcasted);
        break;
    }
    case MATMUL: {
        auto& g = node->grad;                 // [N]
        const auto& A = node->inputs[0]->output;  // [N, F]
        const auto& B = node->inputs[1]->output;  // [F]
        int N = A.shape[0], F = A.shape[1];
        Tensor gB(std::vector<int>{F});
        for (int f = 0; f < F; f++) {
            float acc = 0.0f;
            for (int n = 0; n < N; n++) acc += g.data[n] * A.data[n * F + f];
            gB.data[f] = acc;
        }
        tensor_add_to(node->inputs[1]->grad, gB);
        Tensor gA(A.shape);
        for (int n = 0; n < N; n++)
            for (int f = 0; f < F; f++) gA.data[n * F + f] = g.data[n] * B.data[f];
        tensor_add_to(node->inputs[0]->grad, gA);
        break;
    }
    case SIGMOID: {
        auto& g = node->grad;
        const auto& p = node->output;         // sigmoid 输出, p(1-p) 复用
        Tensor d(p.shape);
        for (int i = 0; i < p.size(); i++)
            d.data[i] = g.data[i] * p.data[i] * (1.0f - p.data[i]);
        tensor_add_to(node->inputs[0]->grad, d);
        break;
    }
    case LOG: {
        auto& g = node->grad;
        const auto& x = node->inputs[0]->output;
        Tensor d(x.shape);
        for (int i = 0; i < x.size(); i++)
            d.data[i] = g.data[i] / std::max(x.data[i], 1e-7f);
        tensor_add_to(node->inputs[0]->grad, d);
        break;
    }
    case SGD_STEP:
        break;
    }
}

}  // namespace lf
