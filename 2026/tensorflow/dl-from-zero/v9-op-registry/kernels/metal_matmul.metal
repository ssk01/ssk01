#include <metal_stdlib>
using namespace metal;

// [N,F] × [F] → [N] 矩阵向量乘 (framework 的 matmul 形态)。
// naive 版: 每输出行一个 thread, 内层 f 循环与 CPU scalar kernel 同序累加 →
// 数值在浮点容差内一致 (编译期可能有 FMA 收缩差异)。
// 诚实标注: 不是 tiled 优化版 —— 目标是演示 GPU 执行路径 + async 重叠,
// 不是性能 (性能级 GPU GEMM 见 README 讨论)。
kernel void matmul_nv(const device float* x [[buffer(0)]],   // [N,F] row-major
                      const device float* w [[buffer(1)]],   // [F]
                      device float* out [[buffer(2)]],       // [N]
                      const constant int* F [[buffer(3)]],
                      uint n [[thread_position_in_grid]]) {
    float acc = 0.0;
    for (int f = 0; f < *F; f++) acc += x[n * (*F) + f] * w[f];
    out[n] = acc;
}
