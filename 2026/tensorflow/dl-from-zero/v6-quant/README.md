# v6-quant — 量化/低精度推理 (dl-from-zero 系列第 6 版)

dl-from-zero 系列第 6 迭代: **min/max 量化到 int8 的推理** —— 对应 TF 首个量化 commit `ca4e053aa52`(2016-04-22, 提交说明原文: *"Add quantization support through new ops and tools"*) + 论文 v2 §5 的 quantization 段落。

## 核心概念(meta 问题)

**v0-v5 的推理全在 fp32 上做——fp32 不是免费的。** 推理场景(移动端/高吞吐数据中心)的瓶颈是算力和带宽: fp32 每个数 4 字节、需要 fp32 算术单元; 把权重和激活压到 int8, 数据带宽减 4x, 并且能用整数乘加单元(后来演化为 gemmlowp / int8 GEMM 的标准形态)。

**为什么量化能几乎无损?** 神经网络的权重/激活分布是"有用的精度很少": 这里 logistic 回归的权重 ~[-1,1]、激活 ~[-4,4], 需要的是动态范围, 不是 fp32 的 2^24 个浮点值——256 个 int8 档位足够承载大部分信息, 舍入误差在累加后被平均掉。

TF 的答案(2016-04-22): 加一组**量化 ops** + 量化公式:

```
QuantizeV2:      (x, min, max) → int8    q = round((clamp(x,min,max)-min) * 255/(max-min)) - 128
QuantizedMatMul: int8 × int8 → int32      c = Σ (a - offset_a) * (b - offset_b),  int32 累加
Dequantize:      int32/int8 → float       x = (c + 2^31) * (max-min)/(2^32-1) + min
```

图层面: 推理图里把 `matmul(x, w)` 替换成 `Dequantize(QuantizedMatMul(Quantize(x), Quantize(w)))` 三件套。

**诚实标注**(本 demo 的关键):
1. 论文 v1(2015-11)没有量化; v2(2016-05)的 §5 段落才写: *"we have implemented support for quantization, which enables faster inference on mobile devices as well as high-throughput datacenter inference, and use the gemmlowp low-precision matrix multiplication library to accelerate quantized computation."*
2. 首个量化 commit 里的 gemmlowp 快路径被 `if (false && ...)` 禁用——**当时实际执行的是 ReferenceGemm**(朴素 int32 累加)。正确性优先, 加速是后来的事。
3. 2016 的量化是 **min/max 属性式**(范围由用户/校准提供), 不是 2017 年 integer-arithmetic-only 论文的 affine scale/zero-point 方案; 但 QuantizedMatMul 的 offset 就是 zero-point 的雏形(数值 0 对应的 int8 档位)。
4. 我们的容器简化: int8/int32 值用 float Tensor 装整数, 不引入真实类型系统——值域与算术和 TF 完全一致, 但 TF 有 qint8/qint32 真实类型 + kernel 注册。

## 量化数学(kernels/quantize.h, 逐条对应 2016 源码)

| 公式 | 对应 TF 源码 |
|---|---|
| `q = round((clamp(x,min,max) - min) * 255/(max-min)) - 128` | `quantize_op.cc` QuantizeV2, MIN_COMBINED 模式 |
| `offset = -round(min * 255/(max-min)) - 128`(zero-point) | `quantization_utils.h` `FloatToQuantizedUnclamped(0, min, max)` |
| `c = Σ (a - offset_a) * (b - offset_b)`, int32 累加 | `reference_gemm.h` ReferenceGemm |
| 输出范围: `[level_a*level_b*(-2^31), level_a*level_b*(2^31-1)]`, level = (max-min)/255 | `quantization_utils.h` `QuantizationRangeForMultiplication` |
| int8 反量化: `x = (q + 128)*(max-min)/255 + min` | `dequantize_op.cc` Dequantize, MIN_COMBINED |
| int32 反量化: `x = (c + 2^31)*(max-min)/(2^32-1) + min` | 同上, half_range = 2^31 |
| `max = max(max, min + max(1,|min|,|max|)/100)` | `quantize_op.cc` 的 epsilon 修正(保证 max > min, 防除零) |

两个值得注意的数值事实:
- **输出范围是"精确"的, 不是保守放大**: `dequant(c) = c * level_a * level_b`。直觉: c 是"每个输入减掉 zero-point 后"的整数和, 而 `(a_q - offset_a) * level_a ≈ a_float`——反量化正好把乘积尺度还回去。
- **offset 与量化公式的舍入不一致是 2016 原版的真实状态**: 输入量化用 `round((x-min)*s)`, offset 用 `round(x*s) - round(min*s)`, 两处舍入并不完全相等——TF 当时就这样, 我们照抄。

## 量化图变换(graph/quantize.h 的 `quantize_inference`)

TF 的量化不是"换 kernel"就完——每个量化 op 都要带 min/max, **图结构得变**:

```
matmul(x, w)  →  dequantize( q_matmul( quantize(x, [xmin,xmax]),
                                        const_int8(w, [wmin,wmax]) ) )
```

| 步骤 | 我们 | 对应 TF 历史 |
|---|---|---|
| 权重 | 静态值直接烘焙成 int8 常量(范围从值统计), 原变量删除 | 部署 freeze 语义; 2016 用户手动把权重量化成常量 |
| 激活 | 运行时 Quantize 节点, 范围来自**校准**(跑一遍数据统计 min/max) | TF 早期"用户提供 min/max"; 自动校准是 2016-06 `quantize_graph.py` eightbit 模式的事 |
| 输出范围 | `q_matmul` 构造时按两个输入范围算 | `QuantizationRangeForMultiplication` |
| 自动化 | 程序化 pass | 2016 年没有自动工具, 手工拼图; 后来 `quantize_graph.py` |

demo 级差异: 只量化 matmul 一层, bias 保持 fp32(2016 的 `quantized_bias_add` 只给 conv 用); 输出先反量化再进激活函数——单层逻辑回归的自然形态。多层的 int32 → int8 requantize 链(`QuantizeDownAndShrinkRange`)没做, 概念相同。

**与 v5 的联动**: 量化图变换后再次 run, 自动优化(CSE/常量折叠)照常运行——CSE 的哈希/等价判定加入了 `qmin/qmax`, 两个范围不同的 Quantize 节点不会误合并, 相同的会被合并(同一个激活喂两个 matmul 时)。

## 运行

```bash
make && ./build/train
```

## 结果(demo 5, 其余为 v5 回归)

```
训练完成: loss=0.440387                    (F=16 特征 logistic 回归, CTR 预估最小形态)
推理图: 6 nodes                            (x, w, b, matmul, add, sigmoid)
量化图: 8 nodes                            (matmul → Quantize/int8-常量/Q_MatMul/Dequantize, 变量 w 被 freeze 删除)
校准: x ∈ [-4.47, 4.03]  scale=30.01  zero_point=6
权重: w ∈ [-0.60, 0.68]  scale=200.03  zero_point=-8
输入量化往返误差 (平均): 0.0083            (int8 分辨率 = 1 个 level ≈ 0.033)
fp32: AUC=0.880733  acc=0.8045
int8: AUC=0.880486  acc=0.8050             ← int8 几乎不掉点
logits 误差: max=0.226  mean=0.017
```

误差解读: 每个乘积项的输入量化误差 ≤ 0.5 level, F=16 项累加后平均误差 ~0.017; max=0.226 出现在 logit 最大的极端样本(逐项误差同向叠加的最坏情况)。这正是工业界"int8 上线几乎不掉点"的微观机理。

## 与 TF 的对应

| 我们 | TF |
|---|---|
| `kernels/quantize.h` 全部公式 | `tensorflow/contrib/quantization/kernels/`(quantize_op.cc / quantized_matmul_op.cc / reference_gemm.h / dequantize_op.cc / quantization_utils.h), commit ca4e053aa52 |
| `QUANTIZE` 节点 | `QuantizeV2` op(MIN_COMBINED) |
| `Q_MATMUL` 节点(int8×int8 → int32) | `QuantizedMatMul` op(gemmlowp 分支当时被 `if (false && ...)` 禁用, 实际跑 ReferenceGemm) |
| `DEQUANTIZE` 节点(按输入类型分派 int8/int32) | `Dequantize` op |
| offset = zero-point | `FloatToQuantizedUnclamped` |
| `quantize_inference` 图变换 pass | 2016 手工拼图 → 2016-06 `quantize_graph.py` eightbit 模式自动做 |
| 校准提供激活 min/max | TF 早期用户提供范围, 自动校准后来才出现 |
| int8 值用 float 容器装 | 类型系统简化; TF 用 qint8/qint32 真实类型 |

## 今天价值(2026, 为什么这个 feature 现在还有意义)

- **LLM 推理**: GPTQ / AWQ / GGUF 的 int8/int4 全是 min/max(或加权重缩放)量化; AWQ 按激活范围加权保护显著通道——本 demo 的"激活范围校准"思想直接演化为 AWQ。
- **搜广推**: 特征/权重 int8 上线是标配, 推理延迟/内存降 4x。
- **gemmlowp 的血统**: 2016 年被禁用的 gemmlowp 快路径, 今天演化为 int8 GEMM 的事实标准(oneDNN / CUTLASS / 移动端 NPU 都只吃 int8)。
- **硬件**: 数据中心 tensor core 的 int8 峰值是 fp32 的 2-4x; 移动端 DSP/NPU 只支持低精度——量化是"模型上手机/模型上 NPU"的前提。
