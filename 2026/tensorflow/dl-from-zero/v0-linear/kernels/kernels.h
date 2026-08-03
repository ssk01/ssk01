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
    case SGD_STEP:
        break;
    }
}

}  // namespace lf
