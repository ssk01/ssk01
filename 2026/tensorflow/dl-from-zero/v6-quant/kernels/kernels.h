#pragma once
#include "../framework/tensor.h"
#include "../graph/graph.h"
#include "../runtime.h"
#include "quantize.h"

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
    case QUANTIZE: {
        // QuantizeV2 (MIN_COMBINED): 输入量化到 int8 (值以 float 存放)
        Node* in = node->inputs[0];
        st.out(node) = quantize_tensor(st.out(in), node->qmin, node->qmax);
        break;
    }
    case Q_MATMUL: {
        // QuantizedMatMul (ReferenceGemm): int8×int8 → int32 累加
        // zero-point = FloatToQuantizedUnclamped(0, min, max) (quantization_utils.h)
        Node* a = node->inputs[0];
        Node* b = node->inputs[1];
        float off_a = quantized_offset(a->qmin, a->qmax);
        float off_b = quantized_offset(b->qmin, b->qmax);
        st.out(node) = quantized_matmul(st.out(a), st.out(b), off_a, off_b);
        break;
    }
    case DEQUANTIZE: {
        // Dequantize (MIN_COMBINED): int32 累加值 (来自 Q_MATMUL) 或 int8
        // 值 (来自 QUANTIZE) 反量化回 float —— 按输入节点类型分派
        const Tensor& q = st.out(node->inputs[0]);
        Tensor out(q.shape);
        bool is_int32 = node->inputs[0]->type == Q_MATMUL;
        for (int i = 0; i < q.size(); i++)
            out.data[i] = is_int32
                ? dequantize_one_i32(q.data[i], node->qmin, node->qmax)
                : dequantize_one_i8(q.data[i], node->qmin, node->qmax);
        st.out(node) = out;
        break;
    }
    case FAKE_QUANT: {
        // FakeQuantWithMinMaxArgs (2016-10-24): 前向 = 量化-反量化往返, 全程 float
        // 里走完 int8 网格。部署时的 dequantize(quantize(x)) 与它同式 —— 训练时
        // 模型见到的舍入误差就是部署时的误差 (QAT 的核心)。
        Node* in = node->inputs[0];
        const Tensor& x = st.out(in);
        Tensor out(x.shape);
        for (int i = 0; i < x.size(); i++)
            out.data[i] = dequantize_one_i8(
                quantize_one(x.data[i], node->qmin, node->qmax), node->qmin, node->qmax);
        st.out(node) = out;
        break;
    }
    case FAKE_QUANT_GRAD: {
        // STE (straight-through estimator) —— 对应 FakeQuantWithMinMaxVarsGradient:
        // 舍入不可导, 近似为恒等: 落在 [min,max] 内的输入梯度直通, 越界截成 0。
        // 范围从第 3 个输入 (fake_quant 节点) 现场读 → 训练中更新范围即时生效。
        // 梯度可能是 [N] (matmul 对 A 的梯度) 而 x 是 [N,F] → 逐行广播掩码。
        Node* g = node->inputs[0];
        Node* x = node->inputs[1];
        const Node* fq = node->inputs[2];
        const Tensor& gv = st.out(g);
        const Tensor& xv = st.out(x);
        Tensor out(xv.shape);
        auto mask = [&](int i) {
            return (xv.data[i] >= fq->qmin && xv.data[i] <= fq->qmax) ? gv.data[i] : 0.0f;
        };
        if (gv.size() == xv.size()) {
            for (int i = 0; i < xv.size(); i++) out.data[i] = mask(i);
        } else {
            int N = xv.shape[0], F = xv.size() / N;
            for (int n = 0; n < N; n++)
                for (int f = 0; f < F; f++) out.data[n * F + f] = mask(n * F + f);
        }
        st.out(node) = out;
        break;
    }
    }
}

}  // namespace lf
