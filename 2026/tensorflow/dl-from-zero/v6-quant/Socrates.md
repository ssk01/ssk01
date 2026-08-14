# Socrates.md - 问答记录

### Q: TF 首个量化 commit 的实际机制是什么? (2016-04-22 ca4e053aa52)
min/max 属性式量化: QuantizeV2 按 MIN_COMBINED 公式 q = round((clamp(x)-min)*255/(max-min))-128 把输入压到 int8; QuantizedMatMul 用 ReferenceGemm 做 int32 累加 (c = Σ(a-offset_a)(b-offset_b), offset = FloatToQuantizedUnclamped(0) 即 zero-point 雏形); 输出范围 = QuantizationRangeForMultiplication (level_a*level_b*int32 全范围, 对反量化是精确的: dequant(c) = c*level_a*level_b); Dequantize 反量化回 float。两个冷门事实: ① gemmlowp 快路径当时被 if(false&&...) 禁用, 实际执行的是朴素参考实现; ② 这是 2016 的 min/max 方案, 不是 2017 integer-arithmetic-only 论文的 affine scale/zero-point 方案。
(2026-08-13)

<!-- 以下继续记录 -->
