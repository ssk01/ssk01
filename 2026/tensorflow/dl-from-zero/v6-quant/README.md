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
5. **demo 5 是"导出/部署时的后训练量化 (PTQ)"**: 训练完的模型才做图变换, 权重冻结成常量、激活范围来自校准——对应 2016 年形态。**量化感知训练 (QAT, 训练图里插 fake-quant 节点让模型学会抵抗舍入) 是后来的事**: FakeQuant ops 2016-10-24 才加入, 2017 年 Jacob et al. 论文定型, 2018 年 `tf.contrib.quantize` 工具化 —— **demo 6 实现了 QAT** (见下文)。工业界常规流程: 先 PTQ, 掉点超阈值再上 QAT (要重训, 成本高)。
6. **量化范围我们存节点字段 (`qmin/qmax`), TF 不是这么存的**: 2016 的 QuantizeV2/QuantizedMatMul 把 min/max 作为**运行时输入张量**(`quantize_op.cc` 里 `ctx->input(1)/(2)`), 范围在图上作为数据流传递, 可由校准 op 动态算出; 通用元数据走 `NodeDef.attr`(`map<string, AttrValue>`), 每种 op 注册时声明自己要哪些 attr, kernel 按名字取——**Node 结构体不随 attr 数量膨胀**。我们平铺两个 float 是 demo 规模下的简化(与 v5 的 `scalar` 同属一类)。
7. **`quantized_matmul` 只有 2 层循环**: 权重是向量, 本 demo 的 matmul 是 matrix-vector(`x[N,F] × w[F]`), 收缩维 f 就是内层循环; TF 的 QuantizedMatMul/ReferenceGemm 是完整 `[M,K]×[K,N]` 矩阵乘 (3 层循环), 多层/多分类时第三层循环才回来。

## 量化数学(kernels/quantize.h, 逐条对应 2016 源码)

> **看不懂 scale/level/zero_point? 一个等价视角**: 直觉版量化 = 先 norm 到 [0,1] 再 ×256。我们的公式只是同一件事:
> - `×255` 不是 `×256`: 255 是间隔数, 端点 x=max → q=127 恰好铺满不越界 (×256 在 max 处溢出要 clamp)
> - `-128` 是平移: 无符号 [0,255] → 带符号 [-128,127] (int8 是带符号类型), 不改变任何信息
> - `level` = (max-min)/255 = "一档多宽" (你的 1/256 宽度); `scale` = 255/(max-min) = 1/level = 你的 256/(max-min)
> - `zero_point`(offset) = 浮点 0 对应的整数档位 = 你版本里 q'(0) 那个数, 唯一作用是 matmul 里居中: `(q - offset)·level ≈ x`, 不居中两个 [0,255] 整数相乘符号量级就乱
> 数字对照 (x=0.5): 你的 q'=round(0.5847×256)=150 ↔ 我们 q=round(0.5847×255)-128=21 (150-128=22, 差 1 是 255/256)



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

### 逐条拆解(7 个公式 = 压入 1 / 算术 3 / 还原 6 + 支撑件 2/4/7)

1. **压入** `q = round((clamp(x,min,max)-min)*s) - 128`, s=255/(max-min): 归一化到 [0,1] → 铺到 256 档 → 平移进 [-128,127](255 档正好铺满 int8)。x=min↔q=-128, x=max↔q=127; clamp 截断不溢出; round 是**唯一误差源** (≤0.5 档)。
2. **zero-point** `offset = -round(min*s) - 128`: 即 q(0) = round(-min*s)-128——数值 0 的整数档位。作用: 公式 3 里居中每个值, 使 `(q-offset)*level ≈ 浮点值`。ReLU 激活 (min=0) → offset=-128; 零中心权重 → offset≈0。
3. **int32 累加** `c = Σ(a-offset_a)(b-offset_b)`: 居中后 |a-offset_a|≤128, 单乘积≤2^14, F 项累加≤F·2^14——F=1024 也才 2^24 ≪ int32 的 ±2^31, **永不溢出**。c 的物理意义: ≈ Σa·b/(level_a·level_b)。
4. **输出范围** `[level_a·level_b·(-2^31), level_a·level_b·(2^31-1)]`: c 每单位对应浮点 level_a·level_b (由 3 的物理意义), int32 全范围×档位乘积 = 输出浮点范围。**精确而非保守**: 与 6 配合代数恒等, 无安全系数。
5. **int8 反量化** `x = (q+128)*level + min`: 1 的严格逆。端点: q=-128↔min, q=127↔max; 往返误差≤0.5 档。
6. **int32 反量化** `x = (c+2^31)*(max-min)/(2^32-1) + min`: 与 5 同构, 网格换成 [0, 2^32-1]。端点: c=-2^31↔min, c=2^31-1↔max。精确性: 用 4 的范围代入 (min = -level_a·level_b·2^31, max-min = level_a·level_b·(2^32-1)) 得 `x = (c+2^31)·level_a·level_b - level_a·level_b·2^31 = c·level_a·level_b`, **代数恒等零额外误差**。用平移形式而非 `x=c·level_a·level_b` 是为保持 MIN_COMBINED 家族统一形态 (Dequantize 不区分输入来源, 只按范围通用处理)。
7. **防除零** `max = max(max, min + max(1,|min|,|max|)/100)`: min==max (常量张量) 时 s 除零 → NaN。ε 取 max(1,|min|,|max|)/100: `1` 保证 min=max=0 不死锁; 取最大量级使修正相对 1%; 只动 max 是单侧修正 (范围只变大), TF 源码原样 (max_range = max(input_max, input_min + epsilon))。

**串起来**: 误差全部集中在公式 1 的单次舍入——int32 累加永不溢出 (3) + 反量化代数精确 (6) = 量化唯一的信息损失是 0.5 档的舍入, 这就是"int8 几乎无损"的完整论证。

### 数值走一遍(F=2 手算, 范围用 demo 真实定标)

定标: x ∈ [-4.47, 4.03] → scale=30.01, level=0.0333, zero_point=**6**; w ∈ [-0.60, 0.68] → scale=200.03, level=0.0050, zero_point=**-8**。样本 x=[0.5, -1.2], w=[0.3, -0.4], bias=0.1 (保持 fp32):

```
① 压入:  q(x₁)=round((0.5+4.47)·30.01)-128=149-128=21   q(x₂)=98-128=-30
         q(w₁)=round(0.90·200.03)-128=180-128=52         q(w₂)=40-128=-88
② 居中:  a'=[21-6, -30-6]=[15, -36]                      b'=[52+8, -88+8]=[60, -80]
         验证: 15×0.0333=0.500≈0.5, 60×0.0050=0.300≈0.3   ← "减 offset × level = 浮点值"
③ 累加:  c = 15·60 + (-36)·(-80) = 900 + 2880 = 3780
④ 反量化: x = c·level_a·level_b = 3780/(30.01·200.03) = 0.630
⑤ 对照:  fp32 = 0.5·0.3 + (-1.2)(-0.4) = 0.630  误差 0.0003 (舍入恰好很准)
```

- **误差来源看得见**: 最坏情况是值落在档位正中——x=2.01 → q=194 → 重构 1.9945, 误差 0.0155 ≈ 半档 0.0166; demo 的 F=16 就是 16 个半档误差的累加 (平均相消 0.017, 极端同向 0.226)
- **zero-point 数值验证**: q(0) 对 x = round(4.47·30.01)-128 = 134-128 = **6** ✓; 对 w = round(0.60·200.03)-128 = 120-128 = **-8** ✓——w 的 offset≈0 说明 0 在网格中间 (零中心权重), x 的 offset=6 说明 0 在网格偏下 (ReLU 激活形态)

## 量化图变换(graph/quantize.h 的 `quantize_inference`)

TF 的量化不是"换 kernel"就完——每个量化 op 都要带 min/max, **图结构得变**:

```
matmul(x, w)  →  dequantize( q_matmul( quantize(x, [xmin,xmax]),
                                        const_int8(w, [wmin,wmax]) ) )
```

| 步骤 | 我们 | 对应 TF 历史 |
|---|---|---|
| 权重 | 静态值直接冻结成 int8 常量(范围从值统计), 原变量删除 | 部署 freeze 语义; 2016 用户手动把权重量化成常量 |
| 激活 | 运行时 Quantize 节点, 范围来自**校准**(跑一遍数据统计 min/max) | TF 早期"用户提供 min/max"; 自动校准是 2016-06 `quantize_graph.py` eightbit 模式的事 |
| 输出范围 | `q_matmul` 构造时按两个输入范围算 | `QuantizationRangeForMultiplication` |
| 自动化 | 程序化 pass | 2016 年没有自动工具, 手工拼图; 后来 `quantize_graph.py` |

demo 级差异: 只量化 matmul 一层, bias 保持 fp32(2016 的 `quantized_bias_add` 只给 conv 用); 输出先反量化再进激活函数——单层逻辑回归的自然形态。多层的 int32 → int8 requantize 链(`QuantizeDownAndShrinkRange`)没做, 概念相同。

**与 v5 的联动**: 量化图变换后再次 run, 自动优化(CSE/常量折叠)照常运行——CSE 的哈希/等价判定加入了 `qmin/qmax`, 两个范围不同的 Quantize 节点不会误合并, 相同的会被合并(同一个激活喂两个 matmul 时)。

## QAT —— 量化感知训练(demo 6, 对应 FakeQuant ops)

**PTQ 的问题**: 量化误差是训练完才"空降"的——模型从没见过自己的 int8 形态, 只能祈祷误差在阈值附近不害人。单层误差不累积没事 (demo 5 掉 0.0002), 深层网络误差逐层放大, 才需要让模型**在训练时就看见自己的部署形态**: 训练图里插 fake-quant 节点, 前向走量化-反量化往返, 梯度用 STE 直通——权重被优化得"天生抗量化"。

### 机制(三个部件)

```
① fake quant 前向 (FakeQuantWithMinMaxArgs):
   out = dequantize( quantize(x) )         ← 全程 float, 但中间值走 int8 网格
   与部署时 dequantize(quantize(x)) 同式 → 训练见到的误差 = 部署误差

② STE 梯度 (straight-through estimator, FakeQuantWithMinMaxVarsGradient):
   ∂L/∂x = grad · (x ∈ [min,max] ? 1 : 0)
   舍入不可导 → 近似为恒等; [min,max] 外截断 (clamp 也不可导, 截掉梯度)

③ 范围更新 (对应 TF 对 min/max 的 EMA, decay≈0.999; 我们 running min/max 简化):
   每轮按观测扩展 x/w 的范围, 只改节点字段, 不触发图重编译;
   FAKE_QUANT_GRAD 从 fake_quant 节点**现场读范围** (对应 TF 的 min/max 张量输入)
```

训练图: `logit = matmul(fake_quant(x), fake_quant(w)) + b`, bias 保持 fp32 (与 demo 5 一致)。fake_quant 的梯度有两条腿: 激活侧梯度是 [N] 对 [N,F] 的逐行掩码广播, 权重侧是 [F] 等尺寸逐元素——对应 matmul 两个输入的梯度形状。

### 导出(QAT 图直接喂 quantize_inference)

```
variable → fake_quant → matmul   →   variable → fake_quant → const_int8(w, 训练范围) → Q_MatMul
```

- 权重链 `variable → fake_quant → matmul` 被烘焙成 int8 常量, 范围**沿用训练冻结值** (`quantize_inference` 新增 `wrange` 参数)——模型是按这个范围训出来的, 导出别换范围
- 激活侧 quantize 吃 **fake_quant 的输出**, 而往返后的值就在 int8 网格上 → 二次量化是**恒等** (quantize(roundtrip(x)) == quantize(x), fp32 精度内精确成立): 不引入额外误差, 图变换无需特殊处理
- 推理图 8 nodes (多两个 fake_quant), 量化图 9 nodes (fake_quant 留在执行路径上当 quantize 的输入)

### demo 6 结果(与 demo 5 同一份数据, seed 7)

```
== demo 6: QAT 量化感知训练 (fake quant + STE) ==
  训练完成: loss=0.440516                (demo 5 PTQ: 0.440387 —— 训练带量化噪声, 略高)
  训练范围: x ∈ [-4.47008, 4.02585]  w ∈ [-0.598151, 0.677068]
  推理图: 8 nodes
  量化图: 9 nodes
  fp32(真实权重): AUC=0.880709  acc=0.805
  QAT fake-quant: AUC=0.880631  acc=0.804
  QAT int8 导出:  AUC=0.880576  acc=0.8045
  fake-quant vs int8 logits 差: max=0.0424267  mean=0.011342
  对照 demo 5 (PTQ): AUC 0.880733 → 0.880486 (掉 0.000246933)
  本 demo  (QAT): AUC 0.880709 → 0.880576 (掉 0.0001332)
```

解读(诚实版):
- **QAT 把量化掉点砍半**: 0.000133 vs PTQ 的 0.000247; 部署 AUC 0.880576 vs PTQ 部署 0.880486。单层模型误差不累积, 这 ~0.0001 的差就是 QAT 的全部收益——**数值上无关痛痒, 机制上完整成立** (深层网络误差逐层放大, 收益成比例放大, 这才是 QAT 的战场)
- **代价**: fake quant 噪声也让训练略难——QAT 模型的纯 fp32 能力 0.880709 比 PTQ 训练的 0.880733 低一点点; 赚的是"量化后"的账
- **fake-quant vs int8 导出 logits 差 max=0.042 / mean=0.011**: 训练模拟 ≠ 部署的缝来自 2016 MIN_COMBINED 的**双重舍入**——量化/反量化用 min/max (`(q+128)·level+min`), matmul 的 offset 用 `round(min·s)` 单独舍入, 两个整数网格不完全重合 (0.5 档 × 另一侧的值, 逐项 ~0.02 的随机游走)。2017 年 affine 方案 (scale/zero_point 单一定义) 没有这个缝, 训练模拟和部署逐位一致
- **范围演化**: x 全批喂入 → 首轮观测即定死; w 从 0 长到 w_star, 范围 [0, 0.01] → [-0.60, 0.68] 真实逐轮扩展——这就是"范围跟着权重一起训"的机制

### 与 TF 的对应

| 我们 | TF |
|---|---|
| `FAKE_QUANT` (前向往返) | `FakeQuantWithMinMaxArgs` (2016-10-24 加入; attrs 版范围写死) |
| `FAKE_QUANT_GRAD` (STE 掩码) | `FakeQuantWithMinMaxVarsGradient` (min/max 是**运行时张量输入**, 我们简化成从 fake_quant 节点现场读) |
| 训练中更新范围 (running min/max) | TF 对 FakeQuantWithMinMaxVars 的 min/max 做 ExponentialMovingAverage (decay≈0.999) |
| 训练图插 fake quant (手写) | 2018 年 `tf.contrib.quantize` (quantize_graph 的 QAT 模式, 自动插到 conv/matmul 前后) |
| 导出沿用训练范围 (`wrange` 参数) | tf.contrib.quantize 导出时把 FakeQuant 的 min/max 变量冻结成常量 |
| STE 梯度建在梯度子图里 | gradients.py 给 FakeQuant 注册的梯度函数 |

## 运行

```bash
make && ./build/train
```

## 结果(demo 5 的 PTQ; demo 6 的 QAT 结果见上文 QAT 一节)

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

误差解读(详细机理, 数字全部验证过):
- **误差唯一来源 = 量化往返**: int32 累加和反量化精确 (dequant(c) = c·level_x·level_w), logit 误差全部来自 x/w 量化到 int8 时的每项舍入
- **单项误差界 ≤ 0.5 level** (round 舍入界); 舍入误差相位近似均匀 → 平均 |误差| ≈ **0.25 level**——实测往返误差 0.0083 ≈ 0.25 × 0.033, 精确吻合
- **每项乘积误差** ≈ |x|·0.5·level_w + |w|·0.5·level_x, 用 |x|≤4 (6σ)、|w|≤0.68 得每项 ≤ 0.021
- **max=0.226 (最坏)**: 16 项误差同向叠加 (理论上界 0.34), 出现在极端样本——每项误差正比于 |x_f|, 大 logit 来自大 x 分量, 所以 logit 最大的样本吃最重的误差
- **mean=0.017 (平均)**: 误差符号对称、项间近似独立 → 16 项累加是随机游走, 按 √F 增长而非 F 线性, 平均值只有最坏情况的 ~1/13
- **为什么不掉点**: AUC 是排序指标, 平均扰动 0.017 远小于 logit 分布 (±3 量级), 只影响 logit 极接近的样本对; 而误差最大的极端样本 logit 最大, 离阈值/邻居最远, 排序最稳。量化误差打在"最不容易错的地方"——这正是工业界"int8 上线几乎不掉点"的微观机理

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
| `FAKE_QUANT` + STE 梯度 (demo 6) | `FakeQuantWithMinMaxArgs/Vars` ops (2016-10-24) + `tf.contrib.quantize` (2018), 见上文 QAT 一节 |

## 今天价值(2026, 为什么这个 feature 现在还有意义)

- **LLM 推理**: GPTQ / AWQ / GGUF 的 int8/int4 全是 min/max(或加权重缩放)量化; AWQ 按激活范围加权保护显著通道——本 demo 的"激活范围校准"思想直接演化为 AWQ。
- **搜广推**: 特征/权重 int8 上线是标配, 推理延迟/内存降 4x。
- **gemmlowp 的血统**: 2016 年被禁用的 gemmlowp 快路径, 今天演化为 int8 GEMM 的事实标准(oneDNN / CUTLASS / 移动端 NPU 都只吃 int8)。
- **硬件**: 数据中心 tensor core 的 int8 峰值是 fp32 的 2-4x; 移动端 DSP/NPU 只支持低精度——量化是"模型上手机/模型上 NPU"的前提。
