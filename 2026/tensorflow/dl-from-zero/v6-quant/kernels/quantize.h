#pragma once
#include <cmath>
#include <cstdint>
#include <utility>
#include "../framework/tensor.h"

namespace lf {

// ============================================================
// 量化内核 —— 对应 TF 首个量化 commit ca4e053aa52
// (2016-04-22, "Add quantization support through new ops and tools")
//
// 机制: min/max 属性式量化 (范围由用户/校准提供), 不是 2017 年
// integer-arithmetic-only 论文的 affine scale/zero-point 方案 ——
// 但 QuantizedMatMul 的 offset 就是 zero-point 的雏形。
//
// 全部公式逐条对应 2016 源码:
//   - QuantizeV2        quantize_op.cc       (MIN_COMBINED 模式)
//   - QuantizedMatMul   quantized_matmul_op.cc / reference_gemm.h
//   - Dequantize        dequantize_op.cc     (MIN_COMBINED 模式)
//   - FloatToQuantized  quantization_utils.h
// ============================================================

// min/max 修正 (quantize_op.cc): 保证 max > min, 避免除零/分辨率退化。
// TF 原文: max_range = max(input_max, input_min + epsilon),
//          epsilon = max(1.0, |min|, |max|) / 100.0
inline void nudge_range(float& min, float& max) {
    float epsilon = std::max(1.0f, std::max(std::fabs(min), std::fabs(max))) / 100.0f;
    max = std::max(max, min + epsilon);
}

// MIN_COMBINED 模式 int8 量化 (quantize_op.cc):
//   q = round((clamp(x, min, max) - min) * 255/(max-min)) - 128
// 返回值用 float 容器装整数 —— 类型系统简化 (TF 用 qint8 真实类型,
// 值域与算术完全一致)
inline float quantize_one(float x, float min, float max) {
    float scale = 255.0f / (max - min);
    float clamped = std::max(min, std::min(x, max));
    return std::round((clamped - min) * scale) - 128.0f;
}

inline Tensor quantize_tensor(const Tensor& x, float min, float max) {
    Tensor out(x.shape);
    for (int i = 0; i < x.size(); i++) out.data[i] = quantize_one(x.data[i], min, max);
    return out;
}

// FloatToQuantizedUnclamped (quantization_utils.h) 在 x=0 处的值:
//   q = round(0 * range_scale) - round(min * range_scale) - 128
// 这就是 QuantizedMatMul 的 zero-point: 数值 0 对应的 int8 档位。
inline float quantized_offset(float min, float max) {
    return -std::round(min * 255.0f / (max - min)) - 128.0f;
}

// QuantizationRangeForMultiplication (quantization_utils.h):
// 两个 int8 输入做 matmul, 输出 (int32 累加值) 能覆盖的浮点范围 =
// 每个输入"一个 level"的乘积 × int32 全范围。
// 对反量化来说这是**精确**范围: dequant(c) = c * level_a * level_b
inline std::pair<float, float> quantized_matmul_range(float amin, float amax,
                                                      float bmin, float bmax) {
    float level_a = (amax - amin) / 255.0f;
    float level_b = (bmax - bmin) / 255.0f;
    return {level_a * level_b * -2147483648.0f,
            level_a * level_b * 2147483647.0f};
}

// int8 反量化 (dequantize_op.cc, MIN_COMBINED):
//   x = (q + 128) * (max-min)/255 + min
inline float dequantize_one_i8(float q, float min, float max) {
    return (q + 128.0f) * (max - min) / 255.0f + min;
}

// int32 反量化 (dequantize_op.cc, MIN_COMBINED, half_range = 2^31):
//   x = (c + 2^31) * (max-min)/(2^32-1) + min
// 配合 quantized_matmul_range, 数值上 ≈ c * level_a * level_b
inline float dequantize_one_i32(float c, float min, float max) {
    return (c + 2147483648.0f) * ((max - min) / 4294967295.0f) + min;
}

// int8 × int8 → int32 的矩阵乘 (reference_gemm.h):
//   c[n] = Σ_k (a[n][k] - offset_a) * (b[k] - offset_b), int32 累加
// 对应 ReferenceGemm —— 2016-04-22 的 gemmlowp 快路径被 "if (false && ...)"
// 禁用, 当时实际执行的就是这个参考实现 (正确性优先, 加速是后来的事)。
inline Tensor quantized_matmul(const Tensor& a, const Tensor& b,
                               float offset_a, float offset_b) {
    int N = a.shape[0], F = a.shape[1];
    Tensor out(std::vector<int>{N});
    for (int n = 0; n < N; n++) {
        int32_t acc = 0;
        for (int f = 0; f < F; f++) {
            int32_t av = static_cast<int32_t>(std::lround(a.data[n * F + f]));
            int32_t bv = static_cast<int32_t>(std::lround(b.data[f]));
            acc += (av - static_cast<int32_t>(std::lround(offset_a))) *
                   (bv - static_cast<int32_t>(std::lround(offset_b)));
        }
        out.data[n] = static_cast<float>(acc);  // int32 值 (float 容器)
    }
    return out;
}

}  // namespace lf
