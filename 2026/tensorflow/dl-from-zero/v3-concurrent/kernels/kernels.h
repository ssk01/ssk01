#pragma once
#include "../framework/tensor.h"
#include "../graph/graph.h"
#include "../runtime.h"

namespace lf {

// 前向: 把每个节点的输出写进本轮 RunState
// 注意这里不再读写 Node 上的任何值 —— 全部通过 RunState
inline void forward(Node* node, RunState& st) {
    switch (node->type) {
    case PLACEHOLDER:
        break;  // 值已由 feed 写入 st.outputs
    case VARIABLE:
        // 从 Session 的持久变量表读当前值 (对应 TF VariableOp 读自己的 Var buffer)
        st.outputs[node] = st.vars->at(node);
        break;
    case ADD:
        st.outputs[node] = tensor_add(st.out(node->inputs[0]), st.out(node->inputs[1]));
        break;
    case MUL:
        st.outputs[node] = tensor_mul(st.out(node->inputs[0]), st.out(node->inputs[1]));
        break;
    case SUB:
        st.outputs[node] = tensor_sub(st.out(node->inputs[0]), st.out(node->inputs[1]));
        break;
    case SQUARE:
        st.outputs[node] = tensor_square(st.out(node->inputs[0]));
        break;
    case MEAN:
        st.outputs[node] = tensor_mean(st.out(node->inputs[0]));
        break;
    case MATMUL:
        st.outputs[node] = tensor_matmul(st.out(node->inputs[0]), st.out(node->inputs[1]));
        break;
    case SIGMOID:
        st.outputs[node] = tensor_sigmoid(st.out(node->inputs[0]));
        break;
    case LOG:
        st.outputs[node] = tensor_log(st.out(node->inputs[0]));
        break;
    case SGD_STEP:
        break;  // 不产生输出; 在 backward 之后由 Session 统一应用梯度
    }
}

// 反向: 把本节点的梯度按链式法则累加到各输入的梯度 (st.grads)
inline void backward(Node* node, RunState& st) {
    switch (node->type) {
    case PLACEHOLDER:
    case VARIABLE:
    case SGD_STEP:
        break;  // 源节点: 无输入可回传, 梯度已由消费者累加进 st.grads
    case ADD: {
        auto& g = st.grad(node);
        tensor_add_to(st.grad(node->inputs[0]), g);
        tensor_add_to(st.grad(node->inputs[1]), g);
        break;
    }
    case MUL: {
        auto& g = st.grad(node);
        tensor_add_to(st.grad(node->inputs[0]),
                      tensor_mul(g, st.out(node->inputs[1])));
        tensor_add_to(st.grad(node->inputs[1]),
                      tensor_mul(g, st.out(node->inputs[0])));
        break;
    }
    case SUB: {
        auto& g = st.grad(node);
        tensor_add_to(st.grad(node->inputs[0]), g);
        tensor_add_to(st.grad(node->inputs[1]), tensor_mul_scalar(g, -1.0f));
        break;
    }
    case SQUARE: {
        auto& g = st.grad(node);
        const auto& x = st.out(node->inputs[0]);
        tensor_add_to(st.grad(node->inputs[0]), tensor_mul(g, tensor_mul_scalar(x, 2.0f)));
        break;
    }
    case MEAN: {
        float n = static_cast<float>(st.out(node->inputs[0]).size());
        auto& g = st.grad(node);
        Tensor broadcasted(st.out(node->inputs[0]).shape);
        float g_scalar = g.data[0];
        for (int i = 0; i < static_cast<int>(n); i++) broadcasted.data[i] = g_scalar / n;
        tensor_add_to(st.grad(node->inputs[0]), broadcasted);
        break;
    }
    case MATMUL: {
        auto& g = st.grad(node);                       // [N]
        const auto& A = st.out(node->inputs[0]);       // [N, F]
        const auto& B = st.out(node->inputs[1]);       // [F]
        int N = A.shape[0], F = A.shape[1];
        Tensor gB(std::vector<int>{F});
        for (int f = 0; f < F; f++) {
            float acc = 0.0f;
            for (int n = 0; n < N; n++) acc += g.data[n] * A.data[n * F + f];
            gB.data[f] = acc;
        }
        tensor_add_to(st.grad(node->inputs[1]), gB);
        Tensor gA(A.shape);
        for (int n = 0; n < N; n++)
            for (int f = 0; f < F; f++) gA.data[n * F + f] = g.data[n] * B.data[f];
        tensor_add_to(st.grad(node->inputs[0]), gA);
        break;
    }
    case SIGMOID: {
        auto& g = st.grad(node);
        const auto& p = st.out(node);   // sigmoid 输出, p(1-p) 复用
        Tensor d(p.shape);
        for (int i = 0; i < p.size(); i++)
            d.data[i] = g.data[i] * p.data[i] * (1.0f - p.data[i]);
        tensor_add_to(st.grad(node->inputs[0]), d);
        break;
    }
    case LOG: {
        auto& g = st.grad(node);
        const auto& x = st.out(node->inputs[0]);
        Tensor d(x.shape);
        for (int i = 0; i < x.size(); i++)
            d.data[i] = g.data[i] / std::max(x.data[i], 1e-7f);
        tensor_add_to(st.grad(node->inputs[0]), d);
        break;
    }
    }
}

}  // namespace lf
