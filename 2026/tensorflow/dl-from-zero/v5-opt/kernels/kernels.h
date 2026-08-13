#pragma once
#include "../framework/tensor.h"
#include "../graph/graph.h"
#include "../runtime.h"

namespace lf {

// 前向: 把每个节点的输出写进本轮 RunState。
// 注意: 没有独立的 backward 了 —— 梯度计算本身是图里的节点(梯度子图),
//       和正向一样走 forward。对应 TF: gradients.py 把梯度建成子图, 由 Executor 执行。
inline void forward(Node* node, RunState& st) {
    switch (node->type) {
    case PLACEHOLDER:
        break;  // 值已由 feed 写入 st.outputs
    case VARIABLE:
        // 从 Session 的持久变量表读当前值 (对应 TF VariableOp 读自己的 Var buffer)
        st.out(node) = st.vars->at(node);
        break;
    case CONST:
        st.out(node) = node->init;
        break;
    case ADD:
        st.out(node) = tensor_add(st.out(node->inputs[0]), st.out(node->inputs[1]));
        break;
    case MUL:
        st.out(node) = tensor_mul(st.out(node->inputs[0]), st.out(node->inputs[1]));
        break;
    case SUB:
        st.out(node) = tensor_sub(st.out(node->inputs[0]), st.out(node->inputs[1]));
        break;
    case SQUARE:
        st.out(node) = tensor_square(st.out(node->inputs[0]));
        break;
    case MEAN:
        st.out(node) = tensor_mean(st.out(node->inputs[0]));
        break;
    case MATMUL:
        st.out(node) = tensor_matmul(st.out(node->inputs[0]), st.out(node->inputs[1]));
        break;
    case SIGMOID:
        st.out(node) = tensor_sigmoid(st.out(node->inputs[0]));
        break;
    case LOG:
        st.out(node) = tensor_log(st.out(node->inputs[0]));
        break;
    case RECIP:
        st.out(node) = tensor_recip(st.out(node->inputs[0]));
        break;
    case REDUCE_SUM:
        st.out(node) = Tensor(tensor_sum(st.out(node->inputs[0])));
        break;
    case MEAN_GRAD:
        st.out(node) = tensor_mean_grad(st.out(node->inputs[0]), st.out(node->inputs[1]));
        break;
    case MATMUL_GRAD_A:
        st.out(node) = tensor_matmul_grad_a(st.out(node->inputs[0]), st.out(node->inputs[1]));
        break;
    case MATMUL_GRAD_B:
        st.out(node) = tensor_matmul_grad_b(st.out(node->inputs[0]), st.out(node->inputs[1]));
        break;
    case SGD_STEP:
        break;  // 不产生输出; 由 Session 在正向之后应用梯度
    }
}

}  // namespace lf
