#pragma once
#include <cmath>
#include <cstdint>
#include <unistd.h>
#include <sys/syscall.h>
#include <mutex>

// SME2 手写 GEMM —— 对应 commit 1 的 CPU matmul (Eigen)。本机 M4 (FEAT_SME2):
// mopa 外积指令把 A 列向量 ⊗ B 行向量累加进 ZA tile, 打在 AMX 硬件上。
// 编译: 所在单元加 -mcpu=apple-m4 (定义 __ARM_FEATURE_SME2);
// 无 SME2 环境 → cblas_sgemm (Accelerate, 本机实测 ~3 TFLOPS 多线程)。
#if defined(__ARM_FEATURE_SME2) && defined(__ARM_FEATURE_SME) && \
    __has_include(<arm_sme.h>)
#include <arm_sme.h>
#include <arm_sve.h>
#define LF_SME2 1
#else
#define LF_SME2 0
#include <Accelerate/Accelerate.h>
#endif

namespace lf {

// 标量参考实现 (与框架 tensor_matmul 同累加顺序)
inline void gemm_scalar(int M, int N, int K, const float* A, const float* B, float* C) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

#if LF_SME2

// 手写 asm 内核踩坑记录 (Apple clang 17, -O2, -mcpu=apple-m4):
//   * ACLE 内建有误编译: svmopa 双谓词 mopa 只写 ZA 第 0 列 (P2 谓词被丢);
//     svread_ver_za32_f32_m 单条读 -O2 时列号/谓词错乱 (4 条展开才正确);
//     streaming 函数里局部数组 → addvl sp 分配 128B 却按 VL=256 布局 → 栈破坏。
//   * s0-s31 是 z0-z31 的低 32 位: ldr s1 写 z1, 随后 insr z1 整寄存器覆盖
//     → 前 4 条 ldr 只有第 1 条"幸存"。全改 GPR: ldr w11 无别名冲突。
//   * w11-w14 复用: 先读 A (zm 链 insr z1) 再读 B (zn 链 insr z0),
//     循环内只需 4 个整数寄存器。x9=k (A 偏移), x10=k*N (B 偏移)。
//   * ZA 读回: mov w12, #slice + mov z2.s, p0/m, za0v.s[w12, 0] + st1w;
//     slice 寄存器必须是 w12-w15。
//   * !!! M4 的 smstart/smstop 会清零 callee-saved 的 d8-d15 (V8-V15):
//     实测每次 SM 切换 (即使不含 ZA 操作) 后 d8-d15 全为 0, 非首次,
//     每次都清 (疑似 Apple SME 状态保存/恢复路径的副作用)。asm 块必须
//     声明 v8-v15 clobber, 否则调用者跨调用缓存的常量/累加器被静默清零
//     → 除零 inf、best 恒 0 (7c 计时恒 0 的真凶, 曾误判为编译器 bug)。
//     与 z0-z2 clobber 不同: z 寄存器是 streaming 模式内的, 不受影响。
//   * streaming 函数零栈帧: 数据全走寄存器/参数; 输出缓冲由调用者传入
//     (调用者是普通函数, 栈帧正常)。

// N==1 (框架 [N,F]×[F]→[N] 的矩阵向量): zm=A[i..i+3,k] (跨行 gather),
// zn = B[k] 广播 (dup z0.s, w15 一条指令)。每 k 一次 4×1 外积累加。
// 不用 __arm_streaming 属性: 编译器会额外生成 smstart sm 序列 (调用点内联时
// 8 对 smstart/smstop, 多线程下与手写 asm 的模式切换交错 → 偶发污染)。
// 改为完全手写: 每个 asm 块自含 smstart sm/za → ... → smstop za/sm。
void sme2_mv_za(int M, int K, const float* A, const float* B, float* C);
void sme2_mv_za(int M, int K, const float* A, const float* B, float* C) {
    const int64_t K64 = K;
    for (int i = 0; i < M; i += 4) {
        asm volatile(
            "smstart sm\n"
            "smstart za\n"
            "ptrue p0.s\n"
            "zero {za}\n"
            "mov x9, #0\n"
            "1:\n"
            "  ldr w11, [%[a0], x9, lsl #2]\n"
            "  ldr w12, [%[a1], x9, lsl #2]\n"
            "  ldr w13, [%[a2], x9, lsl #2]\n"
            "  ldr w14, [%[a3], x9, lsl #2]\n"
            "  ldr w15, [%[b], x9, lsl #2]\n"
            "  insr z1.s, w14\n"
            "  insr z1.s, w13\n"
            "  insr z1.s, w12\n"
            "  insr z1.s, w11\n"      // zm = {A[i][k],A[i+1][k],A[i+2][k],A[i+3][k]}
            "  dup z0.s, w15\n"       // zn = {B[k],B[k],B[k],B[k]}
            "  fmopa za0.s, p0/m, p0/m, z1.s, z0.s\n"
            "  add x9, x9, #1\n"
            "  cmp x9, %[k]\n"
            "  b.ne 1b\n"
            "mov w12, #0\n"
            "mov z2.s, p0/m, za0v.s[w12, 0]\n"
            "str q2, [%[c]]\n"   // !!! st1w {z2.s} 在 M4 上实际存整个 4×4 ZA tile
                                 // (64 字节列主序: 列 0 正确 + 列 1..3 幽灵位移点积)
                                 // → 每次窗口越界写 48 字节, 破坏相邻缓冲。
                                 // str q2 只写 z2 的 16 字节 = 列 0 (正确结果)。
            "smstop za\n"
            "smstop sm\n"
            : : [a0] "r"(A + (i + 0) * K), [a1] "r"(A + (i + 1) * K),
                [a2] "r"(A + (i + 2) * K), [a3] "r"(A + (i + 3) * K),
                [b] "r"(B), [c] "r"(C + i), [k] "r"(K64)
            : "x9", "w11", "w12", "w13", "w14", "w15",
              "z0", "z1", "z2", "p0",
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",  // smstart/smstop 清零
              "memory");
    }
}

// N%4==0 的完整 GEMM: 4×4 f32 ZA tile, 每 k 一次 4×4 外积累加 (16 MAC/指令)。
// zn = B[k*N + j..j+3] 连续行段 (b0..b3 = B+j+c, 偏移 x10 = k*N)。
void sme2_gemm_za(int M, int N, int K, const float* A, const float* B, float* C,
                  float* tmp);
void sme2_gemm_za(int M, int N, int K, const float* A, const float* B, float* C,
                  float* tmp) {
    const int64_t K64 = K, N64 = N;
    for (int i = 0; i < M; i += 4) {
        for (int j = 0; j < N; j += 4) {
            asm volatile(
                "smstart sm\n"
                "smstart za\n"
                "ptrue p0.s\n"
                "zero {za}\n"
                "mov x9, #0\n"        // k (A 元素偏移)
                "mov x10, #0\n"       // k*N (B 元素偏移)
                "1:\n"
                "  ldr w11, [%[a0], x9, lsl #2]\n"
                "  ldr w12, [%[a1], x9, lsl #2]\n"
                "  ldr w13, [%[a2], x9, lsl #2]\n"
                "  ldr w14, [%[a3], x9, lsl #2]\n"
                "  insr z1.s, w14\n"
                "  insr z1.s, w13\n"
                "  insr z1.s, w12\n"
                "  insr z1.s, w11\n"  // zm = A 列 k 的 4 行
                "  ldr w11, [%[b0], x10, lsl #2]\n"
                "  ldr w12, [%[b1], x10, lsl #2]\n"
                "  ldr w13, [%[b2], x10, lsl #2]\n"
                "  ldr w14, [%[b3], x10, lsl #2]\n"
                "  insr z0.s, w14\n"
                "  insr z0.s, w13\n"
                "  insr z0.s, w12\n"
                "  insr z0.s, w11\n"  // zn = B 行 k 的 4 列
                "  fmopa za0.s, p0/m, p0/m, z1.s, z0.s\n"
                "  add x9, x9, #1\n"
                "  add x10, x10, %[n]\n"
                "  cmp x9, %[k]\n"
                "  b.ne 1b\n"
                "mov w12, #0\n"
                "mov z2.s, p0/m, za0v.s[w12, 0]\n"
                "str q2, [%[t0]]\n"   // !!! st1w {z2.s} 在 M4 上存整个 64B tile →
                "mov w12, #1\n"       // 4 次 st1w 会把 tmp[16] 之后 192B 栈写穿。
                "mov z2.s, p0/m, za0v.s[w12, 0]\n"
                "str q2, [%[t1]]\n"   // str q2 只写 16B = 单列。
                "mov w12, #2\n"
                "mov z2.s, p0/m, za0v.s[w12, 0]\n"
                "str q2, [%[t2]]\n"
                "mov w12, #3\n"
                "mov z2.s, p0/m, za0v.s[w12, 0]\n"
                "str q2, [%[t3]]\n"
                "smstop za\n"
                "smstop sm\n"
                :
                : [a0] "r"(A + (i + 0) * K), [a1] "r"(A + (i + 1) * K),
                  [a2] "r"(A + (i + 2) * K), [a3] "r"(A + (i + 3) * K),
                  [b0] "r"(B + j + 0), [b1] "r"(B + j + 1),
                  [b2] "r"(B + j + 2), [b3] "r"(B + j + 3),
                  [n] "r"(N64), [k] "r"(K64),
                  [t0] "r"(tmp + 0), [t1] "r"(tmp + 4),
                  [t2] "r"(tmp + 8), [t3] "r"(tmp + 12)
                : "x9", "x10", "w11", "w12", "w13", "w14",
                  "z0", "z1", "z2", "p0",
                  "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",  // smstart/smstop 清零
                  "memory");
            for (int s = 0; s < 4; s++)
                for (int c = 0; c < 4; c++)
                    C[(i + s) * N + j + c] = tmp[c * 4 + s];
        }
    }
}

// 对外入口: 形状不支持 → 标量; 否则 M 对齐 4 的部分走 ZA, M 尾部回标量
// (尾部循环留在非 streaming 区, 避免跨模式调用; tmp 在普通函数栈上)。
//
// 并发: M4 上 ZA 在多个 streaming 线程间不隔离 —— 实测两个线程同时
// smstart za + mopa 时结果互相覆盖 (每次错同一位置, 确定性串扰; 与
// Darwin 是否保存 ZA 无关, 是并发同时 active 的物理寄存器竞争)。
// 串行化 ZA 段: 正确性优先, 并发 matmul 变顺序 (demo 5 打印诚实加速比)。
inline void sme2_gemm(int M, int N, int K, const float* A, const float* B, float* C) {
    if (N != 1 && N % 4 != 0) {  // 不支持形状 → 标量
        gemm_scalar(M, N, K, A, B, C);
        return;
    }
    static std::mutex za_mutex;
    const int M4 = M / 4 * 4;
    {
        std::lock_guard<std::mutex> lk(za_mutex);
        if (N == 1) {
            sme2_mv_za(M4, K, A, B, C);
        } else {
            float tmp[16];
            sme2_gemm_za(M4, N, K, A, B, C, tmp);
        }
    }
    for (int i = M4; i < M; i++)  // M 尾部 (M%4): 标量
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

#else

// fallback: cblas_sgemm (Accelerate)
inline void sme2_gemm(int M, int N, int K, const float* A, const float* B, float* C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K,
                B, N, 0.0f, C, N);
}

#endif

}  // namespace lf
