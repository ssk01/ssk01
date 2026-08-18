#pragma once
#include <functional>
#include "../core/metal_matmul.h"
#include "../framework/tensor.h"
#include "../framework/kernel_registry.h"
#include "../graph/graph.h"
#include "../runtime.h"
#include "sme2_gemm.h"
#include "registered_kernels.h"  // 加载所有 kernel 注册

namespace lf {

// CPU matmul kernel 开关: true → [N,F]×[F] 走 SME2/AMX (ZA), false → 标量
// tensor_matmul。ZA 硬件在并发 streaming 线程间不隔离 (sme2_gemm.h:177),
// 必须全局互斥 → 并发 matmul 变顺序。demo 5 (执行队列并发调度) 因此关掉
// SME2 展示真并行; SME2 速度由 demo 7 单独演示。
inline bool matmul_use_sme2 = true;

// 前向: 把每个节点的输出写进本轮 RunState。
// 注意: 没有独立的 backward 了 —— 梯度计算本身是图里的节点(梯度子图),
//       和正向一样走 forward。对应 TF: gradients.py 把梯度建成子图, 由 Executor 执行。
//
// v9 版本：使用 Op 注册系统替代 switch-case
inline void forward(Node* node, RunState& st) {
    // 特殊节点直接处理
    if (node->type == PLACEHOLDER) {
        return;  // 值已由 feed 写入
    }
    if (node->type == SGD_STEP) {
        return;  // 不产生输出; 由 Session 在正向之后应用梯度
    }

    // 节点类型名映射
    static const char* type_names[] = {
        "PLACEHOLDER", "VARIABLE", "CONST", "ADD", "MUL", "SUB", "SQUARE",
        "MEAN", "MATMUL", "SIGMOID", "LOG", "RECIP", "REDUCE_SUM", "SGD_STEP",
        "MEAN_GRAD", "MATMUL_GRAD_A", "MATMUL_GRAD_B", "SEND", "RECV"
    };

    const char* op_name = type_names[node->type];

    // 从注册表查找 kernel
    const KernelDef* kernel = KernelRegistry::Global().LookUp(op_name, Device::CPU);

    if (!kernel) {
        throw std::runtime_error(std::string("No kernel registered for op: ") + op_name);
    }

    // 构造上下文并执行
    OpKernelContext ctx;
    ctx.node = node;
    ctx.state = &st;

    kernel->kernel_fn(&ctx);
}

// 旧版本（switch-case）保留用于对比
inline void forward_legacy(Node* node, RunState& st) {
    switch (node->type) {
    case PLACEHOLDER:
        break;  // 值已由 feed 写入 st.outputs
    case VARIABLE:
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
    case MATMUL: {
        const Tensor& A = st.out(node->inputs[0]);
        const Tensor& B = st.out(node->inputs[1]);
        if (matmul_use_sme2 && A.shape.size() == 2 && B.shape.size() == 1 &&
            A.shape[1] == static_cast<int>(B.shape[0])) {
            const int N = A.shape[0], F = A.shape[1];
            Tensor out(std::vector<int>{N});
            sme2_gemm(N, 1, F, A.data.data(), B.data.data(), out.data.data());
            st.out(node) = std::move(out);
        } else {
            st.out(node) = tensor_matmul(A, B);
        }
        break;
    }
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
        break;
    case SEND:
        if (st.rendezvous) {
            st.rendezvous->Send(node->rendezvous_key, st.out(node->inputs[0]));
        }
        break;
    case RECV:
        if (st.rendezvous) {
            st.out(node) = st.rendezvous->Recv(node->rendezvous_key);
        }
        break;
    }
}

// ---- 设备分派 (v7) ----
// GPU op 的 kernel 入口 —— 对应 commit 1 的 BaseGPUDevice::ComputeAsync:
// kernel 立即返回, 完成时从 Metal 内部线程调 done() 回调, executor 用它传播
// 输出 (GPU op 与 CPU op 重叠执行)。
inline void gpu_compute(Node* node, RunState& st, const std::function<void()>& done) {
    if (node->type != MATMUL) {  // 目前只有 matmul 有 Metal kernel; 防御性走 CPU
        forward(node, st);
        done();
        return;
    }
    const Tensor& A = st.out(node->inputs[0]);
    const Tensor& B = st.out(node->inputs[1]);
    if (A.shape.size() != 2) {   // 形状不符 (非 [N,F]×[F]) → CPU 兜底
        forward(node, st);
        done();
        return;
    }
    const int N = A.shape[0], F = A.shape[1];
    st.out(node) = Tensor(std::vector<int>{N});
    // ComputeAsync: 立即返回, GPU 完成后 done() 从 Metal 线程调 (executor 只
    // 做传播, 不等待 → 与 CPU op 重叠)。x/w/out 指针在 done 前保持有效。
    metal::MatmulAsync(A.data.data(), B.data.data(), st.out(node).data.data(), N, F,
                       done);
}

}  // namespace lf
