# 从 MatMul 代码生成到 Primitive Cache，再接入 TF demo

日期：2026-10-09

## 总体路线

分四阶段推进，每阶段用实测回答一个问题，通过后再进入下一阶段：

1. 找到一个常见 kernel，证明针对执行描述生成的代码有执行收益，优先 MatMul。
2. 测清生成成本，证明重复请求值得缓存，以及容量不足如何造成重复生成。
3. 自己实现一个最小 primitive 与缓存 demo，将已经验证的 kernel 包装起来。
4. 将这个 primitive 接入现有 TF demo 的 MatMul 执行路径。

最终形成一条可解释的链路：执行描述 → 生成专用代码 → primitive 持有代码和配置 → 缓存复用 primitive → TF 算子调用。

当前阶段一、二已用 LIBXSMM 官方项目的 SME 路径完成大于 100 的本机对照，详见 [实测与复现](experiments/jit-matmul/README.md)。阶段三、四已实现并通过验证，见 [primitive 与 TF 图执行](v10-primitive-cache/README.md)。业务背景不写入实验与文档。

## 阶段一：生成的 kernel 为什么值得用

### Case 与比较对象

优先二维 MatMul：`A[M,K] × B[K,N] → C[M,N]`。主 case 要求 M、N、K 均大于 100；先选 128、256、512，小矩阵乘向量结果不作为阶段一验收。先控制为单线程、确定布局、相同数据类型及计算语义，再选择少量固定 shape。BF16 是优先探索方向，但必须有可运行的目标实现；FP32 实验单独标注，不替代 BF16 结论。

“自带 kernel”明确分为以下比较对象：

| 对象 | 用途 |
| --- | --- |
| TF demo 当前 MatMul | 判断接入后是否有实际价值 |
| 预编译的通用优化 kernel | 判断固定 shape 专用化是否有额外收益 |
| 按 shape 等描述生成的 kernel | 待验证的执行路径 |

只赢过朴素三重循环，不能证明 JIT 本身优越。比较现成库时，需确认基线实际实现、数据类型、布局及线程数；只知道库名不够。

先借助成熟代码生成实现寻找有效 case，并检查生成代码。若需要严格解释收益来源，再在相同计算策略下比较通用参数版本和固定参数专用版本，尽量保持 SIMD、分块、融合和并行配置一致。不同算法或实现的性能差异，只能说明整体实现更快，不能全部归因于 JIT。

第一轮不承诺自己写机器码生成器。找到有效 case 后，再决定复用生成器，还是为了教学实现范围很小的生成器。使用编译器生成专用动态库也必须明确标注其机制，不能冒充 oneDNN 的直接机器码生成。

### 测量与交付

- 创建、代码生成、内存分配与输入准备放到执行计时之外；计时前预热。
- 分别记录 kernel 执行时间和必要的数据转换/打包成本，避免隐藏额外工作。
- 使用相同输入对照参考结果，明确累加类型和误差容限。
- 保存实际实现名称、代码生成证据、环境与可复现命令。
- 至少五次独立重复，交替运行顺序，报告波动与中位数。

验收：找到至少一组具有可重复执行收益的 shape，并能用代码或配置解释收益来自哪些专用化。生成代码不保证总能赢；未找到有效 case 时，继续定位，不进入集成阶段。

## 阶段二：生成成本是否值得缓存

沿用阶段一的同一个 kernel。将准备过程拆开测量：

| 阶段 | 主要工作 |
| --- | --- |
| 描述与选择 | 检查参数、选择实现、确定布局及分块 |
| 生成 | 生成计算及所需辅助 kernel 的机器码 |
| 代码就绪 | 分配/设置可执行内存、完成后端要求的代码发布操作 |
| 其他初始化 | 配置、索引表及资源准备 |
| 执行 | 使用本次输入、权重和输出运行 kernel |

按实际后端插桩或 profile；无法分离的项目合并报告，不虚构精确占比。尤其不能把整个 primitive 创建时间直接称为 JIT 时间。

先测单 shape 的冷创建、已保留 kernel 的重复执行与缓存命中路径，估计专用化回本次数：

`回本次数 ≈ 额外准备成本 /（通用 kernel 每次执行时间 − 专用 kernel 每次执行时间）`

这个估计只有执行差为正才有意义，并须计入两条路径不同的数据准备成本。它回答“值不值得生成”；缓存实验另外回答“已生成的代码值不值得保留”。

随后对照固定 shape、混合 shape 小容量、相同混合序列足够容量、相同混合序列关闭缓存。预热完整工作集后再测；性能运行关闭日志，诊断运行统计 hit/miss 与真实生成次数。记录准备、执行、完整请求时间及代码占用。

验收：重复生成确实占据可辨别的请求开销；缓存减少真实生成，并带来可重复的总耗时收益。说明生成成本占比与适用 shape 范围，不要求人为制造毫秒级开销。若生成很便宜或执行远大于准备，就如实判定这个 case 不足以展示显著缓存收益。

## 阶段三：实现自己的最小 primitive demo

自己实现对象生命周期和缓存组织；kernel 生成后端可先复用阶段一已验证的实现。这与重新实现 oneDNN 全套 JIT 是两项独立工作。

建议最小接口：

```text
MatmulDesc                 # shape、dtype、布局及支持的执行属性
compile(desc) -> Primitive # 生成代码并建立执行配置
Primitive.execute(A,B,C)   # 每次传入当前数据
PrimitiveCache.get(desc)  # 命中返回已有对象，miss 才 compile
```

Primitive 持有生成代码及其所有权、入口、执行配置和必要的只读辅助表。缓存不保存输入、权重数值或计算结果。需要权重打包时，单独定义其生命周期和成本，不能在修改权重后复用失效数据。

第一版只支持一个算子、一个目标平台和单请求线程；实现容量可调的 LRU、容量 0 禁用、命中/未命中/生成/淘汰计数。key 覆盖所有影响代码与执行语义的描述；固定不支持的属性明确拒绝。淘汰后仍在执行的对象由共享所有权保持存活。

避免重复缓存掩盖实验：若后端还存在自己的缓存，需明确控制并诊断其状态。应用层 miss 但后端命中，不能计作真实代码生成。

验收：容量充足时预热后不再生成；容量不足时可复现淘汰和再生成；相同描述修改输入或权重仍得到正确新输出；缓存开关不改变数值结果。沿用阶段二的测量口径。

## 阶段四：接入 TF demo

在现有 op registry 与 MatMul kernel 调用路径中增加适配，先审查当前接口再确定新版本目录。框架根据 Tensor 元信息构造描述，从共享缓存取得 primitive，然后传入当前 Tensor 数据执行。缓存生命周期跨越单次算子调用。

保留原 MatMul 作为基线；显式选择后端和容量。不支持的描述明确报错或报告 fallback，不能悄悄回退后仍按 JIT 结果统计。

交付一个固定 shape 场景和一个混合 shape 场景，报告框架完整请求、primitive 准备、kernel 执行及缓存统计。先验证前向 MatMul；自动求导、训练与多请求并发另行评估。

验收：在 TF demo 内重现前面验证过的数值正确性、真实代码复用和性能变化，并解释框架开销是否稀释了 kernel 级收益。

## 已有实验处于哪里

- [oneDNN MatMul 实验](experiments/onednn-primitive-cache/README.md)：本机当前 FP32 路径实际使用参考 GEMM，BF16 两种输出模式无法创建。它尚未通过阶段一。
- [Arm JIT 实验](experiments/onednn-primitive-cache/ARM-JIT.md)：Shuffle/Reorder 已验证真实代码生成和缓存减少重复生成，但没有完成阶段一的优化通用 kernel 对照，也未测清生成在创建中的占比。作为诊断辅助，不替代 MatMul 主线。
- Metal 实验保留为辅助材料，不用其 pipeline 创建时间代表 oneDNN JIT 成本。

## 官方依据与复现路线（2026-10-09 更新）

先复现成熟项目的案例，暂停扩展自写 NEON 生成器。`experiments/jit-matvec` 是未纳入主线的小矩阵探索，虽通过数值检查，但不满足新尺寸要求。

### 为什么使用运行时代码生成

oneDNN 的 [MatMul 文档](https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html) 明确区分创建时已知 shape 与执行时才提供 shape：前者给实现更多优化信息，后者便于复用通用 primitive，但限制布局选择等优化。[profiling 文档](https://uxlfoundation.github.io/oneDNN/dev_guide_profilers.html) 说明 JIT 依据 primitive 参数与机器指令集生成代码。

由此应将目标表述为：为运行时确定的尺寸、步长、dtype、硬件和执行属性提供专用实现。JIT 是实现专用化的方法，不意味着同样优化的预编译机器码天然更慢；静态版本覆盖范围与生成/存储成本也是设计取舍。

### 现成案例分别能证明什么

| 来源 | 适合作为什么依据 | 不能直接证明什么 |
| --- | --- | --- |
| [oneDNN MatMul performance example](https://uxlfoundation.github.io/oneDNN/page_matmul_perf_cpp.html) | 完整 MatMul 基准；准备、reorder、创建后预热，再测执行；本地 v3.9.1 源码支持指定尺寸，自动搜索从 128 开始 | 没有相同策略的非 JIT 基线，不能单独归因 JIT 收益 |
| [oneDNN ukernel 基本概念](https://uxlfoundation.github.io/oneDNN/dev_guide_ukernel_basic_concepts.html) 与 [BRGeMM 示例](https://uxlfoundation.github.io/oneDNN/page_cpu_brgemm_example_cpp.html) | 分离配置、生成与执行；官方建议将耗时的代码生成移出主计算循环 | 示例本身不是大矩阵 JIT 对照性能证明 |
| [Intel oneMKL JIT GEMM 说明](https://www.intel.com/content/www/us/en/developer/articles/technical/onemkl-improved-small-matrix-performance-using-just-in-time-jit-code.html) | 直接展示标准 GEMM 与 JIT 的对照、显式生成 API 及成本摊销方法 | 主打小矩阵，不能直接套用为大矩阵必然收益；文中所述旧版本甚至对三个维度均至少 128 的情况不生成 |
| [Hello SME! 原作者论文，第 IV 节](https://arxiv.org/html/2409.18779v1#S4) | M4 上 LIBXSMM SME 代码生成与 Accelerate 对照；K=512，M=N 从 1 到 512，包含目标尺寸 | FP32、原论文平台/版本的结果，不是本机 BF16 结果；整体 kernel 优化差异不能全归因于生成时机 |

论文中的生成器按矩阵元信息固化实现，结合寄存器分块及数据搬运策略。当前选它作为本机复现路线，优先测试 M=N=128、256、512，K=512；使用相同布局、alpha/beta 和线程条件对照 Accelerate。验证完整输出，分别记录创建、首次执行、热执行、必要的转换与总成本。

阶段二在同一后端上使用真实生成接口测生命周期，再做混合描述的容量对照。尤其排查后端内置缓存，不能把外层对象创建误当作真实生成。

当前状态：已复现阶段一、二，详见 [LIBXSMM 实测](experiments/jit-matmul/README.md)。144×512 乘 512×144 的五次执行倍率中位数为 1.126×；混合尺寸缓存容量 2→8 时，160 次 miss→0，总时间中位数 15.533→2.703 ms。纯代码生成约 3.618 μs，完整冷创建约 49.75 μs，两者不可混称。阶段三、四由 GPT-6.1 Sol 子 agent 实现并验证，见 [v10 实现、测试与端到端实测](v10-primitive-cache/README.md)。二维前向已完成；BF16、训练与并发不在已完成范围。
