#pragma once
#include "../framework/kernel_registry.h"
#include "../framework/tensor.h"
#include "sme2_gemm.h"

namespace lf {

// ============================================================
// 注册所有内置 op 的 CPU kernel
// 对应 TF 的 kernels/ 目录下各个 *_op.cc 文件
// ============================================================

// CPU matmul kernel 开关
extern bool matmul_use_sme2;

namespace {

// 注册 ADD kernel (CPU)
struct RegisterAddCPU {
    RegisterAddCPU() {
        KernelDefBuilder("ADD")
            .DeviceType(Device::CPU)
            .Kernel([](OpKernelContext* ctx) {
                const Tensor& a = ctx->input(0);
                const Tensor& b = ctx->input(1);
                ctx->set_output(tensor_add(a, b));
            })
            .Finalize();
    }
} __register_add_cpu;

// 注册 MUL kernel (CPU)
struct RegisterMulCPU {
    RegisterMulCPU() {
        KernelDefBuilder("MUL")
            .DeviceType(Device::CPU)
            .Kernel([](OpKernelContext* ctx) {
                const Tensor& a = ctx->input(0);
                const Tensor& b = ctx->input(1);
                ctx->set_output(tensor_mul(a, b));
            })
            .Finalize();
    }
} __register_mul_cpu;

// 注册 SUB kernel (CPU)
struct RegisterSubCPU {
    RegisterSubCPU() {
        KernelDefBuilder("SUB")
            .DeviceType(Device::CPU)
            .Kernel([](OpKernelContext* ctx) {
                const Tensor& a = ctx->input(0);
                const Tensor& b = ctx->input(1);
                ctx->set_output(tensor_sub(a, b));
            })
            .Finalize();
    }
} __register_sub_cpu;

// 注册 SQUARE kernel (CPU)
struct RegisterSquareCPU {
    RegisterSquareCPU() {
        KernelDefBuilder("SQUARE")
            .DeviceType(Device::CPU)
            .Kernel([](OpKernelContext* ctx) {
                const Tensor& a = ctx->input(0);
                ctx->set_output(tensor_square(a));
            })
            .Finalize();
    }
} __register_square_cpu;

// 注册 MEAN kernel (CPU)
struct RegisterMeanCPU {
    RegisterMeanCPU() {
        KernelDefBuilder("MEAN")
            .DeviceType(Device::CPU)
            .Kernel([](OpKernelContext* ctx) {
                const Tensor& a = ctx->input(0);
                ctx->set_output(tensor_mean(a));
            })
            .Finalize();
    }
} __register_mean_cpu;

// 注册 MATMUL kernel (CPU)
struct RegisterMatmulCPU {
    RegisterMatmulCPU() {
        KernelDefBuilder("MATMUL")
            .DeviceType(Device::CPU)
            .Kernel([](OpKernelContext* ctx) {
                const Tensor& A = ctx->input(0);
                const Tensor& B = ctx->input(1);

                if (matmul_use_sme2 && A.shape.size() == 2 && B.shape.size() == 1 &&
                    A.shape[1] == static_cast<int>(B.shape[0])) {
                    const int N = A.shape[0], F = A.shape[1];
                    Tensor out(std::vector<int>{N});
                    sme2_gemm(N, 1, F, A.data.data(), B.data.data(), out.data.data());
                    ctx->set_output(std::move(out));
                } else {
                    ctx->set_output(tensor_matmul(A, B));
                }
            })
            .Finalize();
    }
} __register_matmul_cpu;

// 注册 SIGMOID kernel (CPU)
struct RegisterSigmoidCPU {
    RegisterSigmoidCPU() {
        KernelDefBuilder("SIGMOID")
            .DeviceType(Device::CPU)
            .Kernel([](OpKernelContext* ctx) {
                const Tensor& a = ctx->input(0);
                ctx->set_output(tensor_sigmoid(a));
            })
            .Finalize();
    }
} __register_sigmoid_cpu;

// 注册 LOG kernel (CPU)
struct RegisterLogCPU {
    RegisterLogCPU() {
        KernelDefBuilder("LOG")
            .DeviceType(Device::CPU)
            .Kernel([](OpKernelContext* ctx) {
                const Tensor& a = ctx->input(0);
                ctx->set_output(tensor_log(a));
            })
            .Finalize();
    }
} __register_log_cpu;

// 注册 RECIP kernel (CPU)
struct RegisterRecipCPU {
    RegisterRecipCPU() {
        KernelDefBuilder("RECIP")
            .DeviceType(Device::CPU)
            .Kernel([](OpKernelContext* ctx) {
                const Tensor& a = ctx->input(0);
                ctx->set_output(tensor_recip(a));
            })
            .Finalize();
    }
} __register_recip_cpu;

// 注册 REDUCE_SUM kernel (CPU)
struct RegisterReduceSumCPU {
    RegisterReduceSumCPU() {
        KernelDefBuilder("REDUCE_SUM")
            .DeviceType(Device::CPU)
            .Kernel([](OpKernelContext* ctx) {
                const Tensor& a = ctx->input(0);
                ctx->set_output(Tensor(tensor_sum(a)));
            })
            .Finalize();
    }
} __register_reduce_sum_cpu;

// 注册 MEAN_GRAD kernel (CPU)
struct RegisterMeanGradCPU {
    RegisterMeanGradCPU() {
        KernelDefBuilder("MEAN_GRAD")
            .DeviceType(Device::CPU)
            .Kernel([](OpKernelContext* ctx) {
                const Tensor& grad = ctx->input(0);
                const Tensor& x = ctx->input(1);
                ctx->set_output(tensor_mean_grad(grad, x));
            })
            .Finalize();
    }
} __register_mean_grad_cpu;

// 注册 MATMUL_GRAD_A kernel (CPU)
struct RegisterMatmulGradACPU {
    RegisterMatmulGradACPU() {
        KernelDefBuilder("MATMUL_GRAD_A")
            .DeviceType(Device::CPU)
            .Kernel([](OpKernelContext* ctx) {
                const Tensor& grad = ctx->input(0);
                const Tensor& B = ctx->input(1);
                ctx->set_output(tensor_matmul_grad_a(grad, B));
            })
            .Finalize();
    }
} __register_matmul_grad_a_cpu;

// 注册 MATMUL_GRAD_B kernel (CPU)
struct RegisterMatmulGradBCPU {
    RegisterMatmulGradBCPU() {
        KernelDefBuilder("MATMUL_GRAD_B")
            .DeviceType(Device::CPU)
            .Kernel([](OpKernelContext* ctx) {
                const Tensor& grad = ctx->input(0);
                const Tensor& A = ctx->input(1);
                ctx->set_output(tensor_matmul_grad_b(grad, A));
            })
            .Finalize();
    }
} __register_matmul_grad_b_cpu;

// 注册 CONST kernel (CPU)
struct RegisterConstCPU {
    RegisterConstCPU() {
        KernelDefBuilder("CONST")
            .DeviceType(Device::CPU)
            .Kernel([](OpKernelContext* ctx) {
                ctx->set_output(ctx->node->init);
            })
            .Finalize();
    }
} __register_const_cpu;

// 注册 VARIABLE kernel (CPU)
struct RegisterVariableCPU {
    RegisterVariableCPU() {
        KernelDefBuilder("VARIABLE")
            .DeviceType(Device::CPU)
            .Kernel([](OpKernelContext* ctx) {
                ctx->set_output(ctx->state->vars->at(ctx->node));
            })
            .Finalize();
    }
} __register_variable_cpu;

// 注册 SEND kernel (CPU)
struct RegisterSendCPU {
    RegisterSendCPU() {
        KernelDefBuilder("SEND")
            .DeviceType(Device::CPU)
            .Kernel([](OpKernelContext* ctx) {
                if (ctx->state->rendezvous) {
                    ctx->state->rendezvous->Send(ctx->node->rendezvous_key,
                                                ctx->input(0));
                }
            })
            .Finalize();
    }
} __register_send_cpu;

// 注册 RECV kernel (CPU)
struct RegisterRecvCPU {
    RegisterRecvCPU() {
        KernelDefBuilder("RECV")
            .DeviceType(Device::CPU)
            .Kernel([](OpKernelContext* ctx) {
                if (ctx->state->rendezvous) {
                    ctx->set_output(ctx->state->rendezvous->Recv(ctx->node->rendezvous_key));
                }
            })
            .Finalize();
    }
} __register_recv_cpu;

}  // namespace

}  // namespace lf
