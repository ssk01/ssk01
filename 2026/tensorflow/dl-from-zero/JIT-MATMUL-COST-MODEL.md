# oneDNN MatMul 代码生成与成本模型

这份笔记解释 MatMul 如何从操作描述变成执行方案和机器码，primitive cache 为什么有价值，以及怎样估算计算和多级缓存搬运的成本。阅读项目实现先看[阅读导览](READING-GUIDE.md)；这里集中回答设计原理和性能归因。

核心结论：**优化方案决定怎样算，JIT 把方案落实成专用代码，缓存避免重复准备。三者有关联，但不能把更好的分块策略所带来的收益全部归给 JIT。** 相同机器码在相同条件下执行，不会因为是运行时生成而天然更快。

## 1. 范围与证据口径

| 内容 | 证据性质 | 范围 |
| --- | --- | --- |
| 当前 v10 demo | 仓库实现 | 教学框架接 LIBXSMM，自己实现 primitive 生命周期与 LRU；不是接入 oneDNN |
| LIBXSMM 与 Accelerate 对照 | 本机实测和反汇编诊断 | M4 Pro、FP32、单线程；不能外推 BF16/AMX |
| oneDNN MatMul 配置与 JIT | 源码分析 | oneDNN v3.9.1，x86/AMX BRGEMM 路径；本机 ARM 未执行该路径 |
| 本文数值成本案例 | 明确假设下的手算 | 不是 benchmark，也不是运行 oneDNN 评分函数得到的预测 |

oneDNN 源码基准提交为 `80a3a8e745d2f0186e674b0af9332fd6e074c94f`。不同版本、ISA、数据类型和布局可能选择不同实现。

## 2. 从执行方案到机器码

```text
操作描述：M/N/K、类型、布局、转置、ISA、线程数、post-ops
    ↓
配置选择：分块、线程划分、遍历顺序、数据整理、尾块策略
    ↓
代码生成：加载、乘加、循环、预取、存储、后处理
    ↓
指令编码、跳转修正、可执行内存等平台处理
    ↓
保存函数入口与代码所有权，供多次执行复用
```

### 2.1 固定规则为什么仍能动态生成

规则固定不等于输出程序固定。比如生成器中有：

```cpp
// 简化示意：emit 发射指令，不是在这里做矩阵乘法。
emit_fmopa(tile0);
if (m_block > 16) emit_fmopa(tile1);
if (n_block > 16) emit_fmopa(tile2);
if (m_block > 16 && n_block > 16) emit_fmopa(tile3);
```

32×32 块生成四条指令，16×16 块只生成第一条。这些 if 在创建代码时执行，生成后的计算循环不用再执行这些判断。K、步长、类型和 post-ops 也会影响输出代码。

这叫专用化，不等于在线自动调优。生成器可以使用人工规则，也可以先用成本模型选方案；“运行时生成”不保证“方案一定最优”。

### 2.2 编译发生在哪里

构建库时，C++ 编译器先编译生成器本身。创建 primitive 时，生成器运行，再输出专用机器码。

LIBXSMM 的 AArch64 编码器直接填指令位。例如 FP32 FMOPA 基础编码是 `0x80800000`，把 tile、向量寄存器和谓词寄存器编号写入指定 bit 后，得到实际指令。某组参数生成 `0x80800440`，反汇编为：

```asm
fmopa za0.s, p1/m, p0/m, z2.s, z0.s
```

oneDNN 的 x86 手工 JIT 路径使用 Xbyak。生成器调用 `tdpbf16ps(...)` 等接口时，编码器把指令写入缓冲区；它不是当场执行乘法，也不需要生成 C++ 字符串后调用 g++。

这是输入语言很窄的专用编译过程：计算描述 → 配置 → 指令 → 机器码，不一定经过通用 IR 和 LLVM 式优化 pass。

机器码与 ELF `.text` 中的指令属于同一性质；macOS 对应 Mach-O 的 `__TEXT,__text`。JIT 可以直接创建内存中的代码，不必先生成完整目标文件。

### 2.3 不是消除所有函数调用

完整 MatMul 仍可能由外层驱动遍历工作块，调用 packing、主 kernel、尾块 kernel 和归并函数。一个 primitive 可以包含多个生成函数。

非 JIT 的预编译 kernel 也在直接执行机器码，并不天然属于解释执行。少几次间接调用可能有收益，但 JIT 还可能消除已知条件、固定访存偏移、专门处理尾块并融合后处理。相同优化也可以提前编译；JIT 的价值是实际参数出现后再按需生成，不必预先准备所有组合。

## 3. 本机实验真正证明了什么

LIBXSMM 是可生成专用计算内核的库；Accelerate 是 Apple 的数值计算框架，这里用其 `cblas_sgemm`。CBLAS 是接口，不能把它等同于朴素三重循环。

对 `A[144,512] × B[512,144]`，两边的主体都实际使用 SME，主循环都有两次向量加载和四条 FMOPA。差别之一是外层面板及边缘组合：

| 输出区域 | Accelerate 命中的路径 | LIBXSMM |
| --- | --- | --- |
| 128×128 主体 | 16 个 32×32 块 | 16 个 32×32 块 |
| 16×128 底部 | 4 个 16×32 块 | 2 个 16×64 块 |
| 128×16 右侧 | 2 个 64×16 块 | 2 个 64×16 块 |
| 16×16 右下角 | 1 个边缘块 | 1 个边缘块 |
| 总计算块数 | 23 | 21 |

LIBXSMM 该路径中 N/64、M/32 是生成器的固定规则；Apple 命中的外层路径按 32 列推进。两边都没有在这次执行中试跑多个面板宽度来挑最快值。

进一步保留 Apple 自己的 packing 和计算函数，改变外层面板宽度，得到：

| 路径 | 每次耗时，中位数 |
| --- | ---: |
| 正常 cblas_sgemm | 16.18 μs |
| Apple 内部函数，32 列面板 | 14.99 μs |
| 同样的 Apple 内部函数，64 列面板 | 13.91 μs |
| LIBXSMM 完整生成内核 | 13.37 μs |

5 个独立进程，每进程 31 轮，每条路径每轮 300 次，随机排列路径顺序。改变 A/B 后输出仍与 CBLAS 一致。32→64 的耗时下降在五次运行中为 6.4%～7.4%。这次实验独立于早期实验中的 13.55/15.22 μs，不混用计时结果。

**修改 Apple 内核的调用组织，无需生成新代码，就追回了大部分内部路径差距。** 因而这个 case 证明了专用实现和分块策略的收益，不能证明 JIT 本身必然优于预编译实现。

直接调用内部函数同时改变了入口、调度和临时缓冲管理，不能把与正常 CBLAS 的差值全部叫作参数检查成本；剩余差距也未逐项归因。私有地址调用仅用于该系统版本的诊断，不是项目支持的后端接口。

本轮的[原始计时与环境记录](experiments/jit-matmul/accelerate-panel-analysis.json)保留了五次结果。主线实验的复现方法仍见[实验 README](experiments/jit-matmul/README.md)。

## 4. oneDNN 的 MatMul 有哪些优化决策与生成代码

oneDNN 官方的 [Basic Concepts](https://uxlfoundation.github.io/oneDNN/dev_guide_basic_concepts.html)说明：primitive 保存操作参数，提前计算如 cache blocking 的配置，并可生成专用代码；重复执行摊薄准备成本。[MatMul 与 SGEMM 教程](https://uxlfoundation.github.io/oneDNN/v3.7/page_cpu_sgemm_and_matmul_cpp.html)则区分创建时确定 shape 与执行时才确定 shape，体现专用性与通用性的取舍。

在检查的 AMX MatMul 路径中：

1. 配置阶段可枚举 batch/M/K/N 方向的线程划分，确定分块并比较评分。
2. 根据完整块/尾块、首次计算/继续累加、归约批次等组合，建立所需 BRGEMM 描述。
3. 为实际需要的描述生成内核，并按需生成 A/B 整理和辅助函数。

BRGEMM 可理解为 `C = Σ AᵢBᵢ`，用于连续累加多对输入块。生成逻辑包括：

| 优化位置 | 具体内容 |
| --- | --- |
| 乘加 | BF16 路径发射 TDPBF16PS，安排 tile 使用及数据复用 |
| 加载存储 | 预取、适用的加载形式、交错存储 |
| 累加策略 | 清零或读取旧累加结果；按 alpha/beta 和类型选择路径 |
| 尾部 | M/N/K 尾块、掩码、边缘读取处理 |
| 数据准备 | 按需复制、转置、转换或整理输入布局 |
| 后处理 | 按属性生成 bias、scales、post-ops 和输出转换 |

融合不意味着全部在 AMX tile 内完成；结果可能经过工作区进入向量寄存器做后处理。所有条目都受实现和配置支持范围约束。

源码入口见文末。这里的方案选择比本机 LIBXSMM 示例中的固定面板规则丰富，但不能推断任何 shape 都会使用同一套模型。

## 5. AMX macro-blocking 成本模型怎样打分

以下是 `matmul_amx_blocking_params_macro_t::calculate_blocking_scores()` 的逻辑。它是候选方案的执行成本估计，不是创建/JIT 耗时估计。

### 5.1 遍历方向与流量

每线程工作按输出条带 strip 建模。horizontal 路径假设 B 在条带间从 L2 复用，每条带加载新的 A；vertical 路径交换角色。模型区分第一条的装载和后续稳态条带，并考虑共享输入的线程数。

### 5.2 带宽与缓存

基本形式为 `字节数 / 有效带宽`。带宽表来自源码注释所述的 EMR 机器，两个 NUMA 域、每域 32 核，不是运行时现场测量。

| 参数 | 模型值，按 bytes/cycle 使用 |
| --- | ---: |
| L1 load/store hit | 106.41 |
| L1 load miss | 106.41 / 2.28 |
| L1 store miss | 106.41 / 2.85 |
| LLC | 6.0 |

共享读取带宽表按共享核数给出值：1→2.28、2→0.90、4→2.39、8→2.98、16→3.31、32→4.06，中间用线性插值。它是经验曲线，不能理解成硬件峰值或核数越多必然越快。

L1 流量按复用关系推算，区分命中、未命中、输出 cache line 写入和累加临时结果访问，不是逐条 cache line 模拟，也不读取当前硬件 miss 计数。

### 5.3 AMX 乘加

```text
TMUL 次数 ≈ 条带计算量 / (tile_M × tile_K × tile_N)
单条 TMUL 周期估计 = m_tmul × 64 × 16 / 1024
计算周期 = 次数 × 单条周期
```

该模型假设减小 tile 的 K/N 不会等比例缩短指令成本；因此尾块利用率低时，少量有效计算仍可能支付较大的指令成本。这是模型假设，不是所有硬件的通用延迟保证。

### 5.4 后处理

`postops_estimator_t` 先生成一条 cache line 对应的后处理代码，但不执行它：

```text
预计向量指令数 = max(代码字节数 − 8, 0) / 7
后处理周期 = 指令数 × 处理的 cache line 数 / 1.2
```

代码大小用于近似指令数量，1.2 是 AVX IPC 经验参数；这不是精确的依赖或端口调度分析。

### 5.5 合成成本与 K 归并

```text
稳态条带周期 = max(DRAM, LLC, L1, TMUL, AVX 后处理的周期估计)
GEMM 周期 = 首条周期 + (条带数 − 1) × 稳态条带周期
```

取最大值近似各阶段重叠后的吞吐瓶颈，不等于真实时间线。

沿 K 分线程时，每线程产生部分 C，还要读取、求和、写最终结果。模型计入这些访问成本，并加上测量得到的 25000 周期固定开销；临时结果和 batch 等条件不满足时，直接判候选无效。

```text
预测总周期 = (GEMM 周期 + 归并周期) × 每线程 batch 数
评分 = 模型中的理想峰值周期 / 预测总周期
```

分数越高越好。另一套 micro-blocking 策略用缓存利用、数据复用和线程均衡的加权分数，不能与本节的周期模型混为一谈。

## 6. 五类权衡怎样影响候选排序

| 情形 | 可获收益 | 代价或限制 |
| --- | --- | --- |
| 数据搬运重 | 改善分块、遍历和线程共享关系，减少重复加载 | 更大块可能挤出缓存；共享不等于零成本 |
| tile 利用率低 | 更规整的块减少低利用率边缘计算 | 必须兼顾线程均衡和可用任务数 |
| 后处理重 | 改善输出分配、尾部和计算/输出组织 | AMX 本体加速未必改善总吞吐 |
| M/N 并行度不足 | 沿 K 切分，让更多核心参与 | 增加部分 C、归并和同步 |
| K 归并昂贵 | 放弃 K 划分，减少额外读写 | 可用计算并行度可能降低 |

例如 `128×16384` 乘 `16384×128`，输出任务可能不多而归约很长，沿 K 划分值得作为候选；若输出本身很大，M/N 已有足够并行度，多份部分 C 又很贵，K 划分就可能不划算。非线性后处理通常需要等部分结果相加完成后执行。

## 7. 教学手算与瓶颈判定

本节明确假设：BF16 输入、FP32 输出、beta=0；有效计算吞吐 P=16 TFLOP/s，DRAM 带宽 B=100 GB/s。所有时间都是手算，且忽略部分真实开销。GB 使用十进制，KiB/MiB 使用二进制。

```text
计算量 F = 2MNK
最低流量 Q = 2MK + 2KN + 4MN
计算时间 = F/P；搬运时间 = Q/B
平衡计算强度 P/B = 160 FLOPs/字节
```

| 操作 | M/N/K | 计算时间估计 | 最低 DRAM 时间估计 |
| --- | --- | ---: | ---: |
| 较小 GEMM | 128/128/512 | 1.05 μs | 3.28 μs |
| 大方阵 GEMM | 4096/4096/4096 | 8.59 ms | 1.34 ms |
| GEMV | 4096/1/4096 | 2.10 μs | 335.79 μs |
| 向量点积 | 1/1/4096 | 0.000512 μs | 0.164 μs |

点积和低 N 的 GEMV 不应期待达到整机矩阵乘法吞吐，表中的计算时间只是乐观参照。点积还要考虑水平归约、依赖链和固定开销。

第一行的计算强度为 51.2 FLOPs/字节。若 P 不变，临界带宽为 `16 TFLOP/s ÷ 51.2 = 312.5 GB/s`：低于它搬运更慢，高于它计算更慢。但若实际 kernel 只有 2 TFLOP/s，计算就需要 8.39 μs，判断再次改变。

**瓶颈由 shape、数据驻留层级和有效计算效率共同决定，不能只看 shape 或峰值规格。**

## 8. 大方阵的多级缓存手算

继续算 L=4096，比较 L2 宏块 b=128/512；L1 块 t=64，寄存器块 r=16。采用逐输出块、分 K 累加的理想化嵌套分块模型。

假设同一计算资源范围内的有效带宽：DRAM/LLC 100 GB/s、LLC/L2 200 GB/s、L2/L1 2 TB/s、L1/寄存器 8 TB/s；L1 64 KiB、L2 4 MiB。LLC 足够保住输入，输出不会把输入挤掉。这里不是一台实际机器的规格，也没有混用整机峰值与某单核的测量值。

工作集基本容量 `A块+B块+C块=8b²`：128 为 128 KiB，512 为 2 MiB；L1 的 t=64 块为 32 KiB。暂不计 padding、额外缓冲、冲突和写分配流量。

令 `Cbytes=4L²`：

```text
DRAM 流量 = 8L²
LLC/L2 流量 = 4L³/b + Cbytes
L2/L1 流量 = 4L³/t + Cbytes × (2L/b − 1)
L1/寄存器流量 = 4L³/r + Cbytes × (2L/t − 1)
```

`2L/b−1` 表示 K 宏块间的部分结果访问：第一次不读旧 C，后续读部分和再写回。实际实现改变循环顺序或驻留策略后，需要重算流量，不能直接套公式。

| 成本项 | b=128 | b=512 |
| --- | ---: | ---: |
| DRAM，均 128 MiB | 1.34 ms | 1.34 ms |
| LLC/L2，2.215 / 0.604 GB | 11.07 ms | 3.02 ms |
| L2/L1，8.523 / 5.302 GB | 4.26 ms | 2.65 ms |
| L1/寄存器，均 25.703 GB | 3.21 ms | 3.21 ms |
| 计算 | 8.59 ms | 8.59 ms |
| 取最大值的吞吐估计 | 11.07 ms | 8.59 ms |

LLC/L2 时间高于 DRAM 时间，是因为反复传输的数据更多：输入只从 DRAM 进入 LLC 一次，仍可多次进入 L2。带宽更高，不代表处理十几倍流量的总时间更短。

### 为什么不继续增大到 1024

b=1024 时，A/B/C 块分别为 2/2/4 MiB，总计 8 MiB，超过假设的 4 MiB L2。原先的驻留前提失效，不能再机械使用更低流量的预测。

512 也未被证明最优：矩形分块、其他块宽、K 独立分块都可能更好。仅看基本容量，4 MiB 对应 b≤约724；若只使用 3/4 容量，则 b≤约627，还要满足 tile 对齐和实际缓冲要求。

而且 512 已经转为计算限制，继续减少搬运不一定降低最大项，却可能损害并行度和缓存驻留。优化目标是总成本，不是块越大越好。

## 9. GEMV 与点积应怎样看

小向量 v 通常容易留在缓存，但仍要从缓存加载到寄存器。GEMV 的主要流量往往来自 A。优化包括连续读取 A、一次加载 v 更新多个输出、多个独立累加器、避免不划算的 packing，以及按带宽饱和点选择线程数。多个请求若能批成 GEMM，可增加 A 的复用，但会引入等待约束。

oneDNN v3.9.1 的 GEMM 后端有 M=1/N=1 转 GEMV 的逻辑，也有 BF16 向量点积指令 `vdpbf16ps` 的 JIT 实现；具体 MatMul primitive 是否选到该路径，还要看分派和支持条件。

GEMV 的 `thread_checker()` 使用经验决策，FP32 AVX2 路径有：

```cpp
if (m * n + 288 * n < 41700) return 1;
```

这些 m/n 是 GEMV 驱动内部维度，不能不看映射就代入原始 MatMul 维度。注释注明阈值来自冷热缓存下的性能测量。其他分支还看每线程工作量、特定线程池下的 L2 容量；非 FP32 路径使用最小划分宽度等规则。

点积可以表示为 `M=N=1` 的 MatMul，但不能因此把 AMX macro-blocking 的周期模型直接套上去。检查到的 GEMV 路径采用上述启发式，而非前面的逐项周期估算；这也不是对 oneDNN 所有版本和点积实现的穷尽性结论。

## 10. primitive cache 与项目结论

primitive 的准备可能包含候选评估、布局与缓冲配置、多个主块/尾块 kernel 生成和辅助代码生成。缓存复用这些准备成果，避免相同描述重新创建。

当前 demo 的缓存保存描述对应的代码所有权和执行入口，不缓存输入、权重数值或输出；执行时仍传入当前数据。权重预打包数据的复用是另一项机制，不能与 primitive cache 混称。

因此项目已展示的内容应表述为：专用实现的性能对照、创建成本、缓存命中与框架接入。要证明 JIT 本身的额外收益，需要在相同算法、分块、布局、融合和线程条件下，对照预编译组合与生成版本；当前实验没有完成这一严格隔离。

## 11. 源码阅读入口

以下 oneDNN 链接固定到本次分析的提交：

- [MatMul 配置与启发式入口](https://github.com/uxlfoundation/oneDNN/blob/80a3a8e745d2f0186e674b0af9332fd6e074c94f/src/cpu/x64/matmul/brgemm_matmul_utils.cpp)：`compute_blocking_heuristic`。
- [AMX 候选搜索与评分](https://github.com/uxlfoundation/oneDNN/blob/80a3a8e745d2f0186e674b0af9332fd6e074c94f/src/cpu/x64/matmul/amx_blocking_heuristics.cpp)：`find_best_blocking`、两套 `calculate_blocking_scores`。
- [带宽表与模型参数](https://github.com/uxlfoundation/oneDNN/blob/80a3a8e745d2f0186e674b0af9332fd6e074c94f/src/cpu/x64/matmul/amx_blocking_heuristics.hpp)：`bw_map_t`。
- [后处理成本估算](https://github.com/uxlfoundation/oneDNN/blob/80a3a8e745d2f0186e674b0af9332fd6e074c94f/src/cpu/x64/matmul/postops_estimator.hpp)：生成但不执行的估算代码。
- [MatMul 创建内核集合](https://github.com/uxlfoundation/oneDNN/blob/80a3a8e745d2f0186e674b0af9332fd6e074c94f/src/cpu/x64/matmul/brgemm_matmul.cpp)：`init`。
- [AMX 指令与后处理生成](https://github.com/uxlfoundation/oneDNN/blob/80a3a8e745d2f0186e674b0af9332fd6e074c94f/src/cpu/x64/brgemm/jit_brgemm_amx_uker.cpp)：`generate`、`tdpbxxd`。
- [GEMV 分派与线程选择](https://github.com/uxlfoundation/oneDNN/blob/80a3a8e745d2f0186e674b0af9332fd6e074c94f/src/cpu/x64/gemm/gemv_driver.cpp)：`jump_to_gemv`、`thread_checker`。
- [LIBXSMM M4 分块规则](https://github.com/libxsmm/libxsmm/blob/3057b239c16994d307589e40e45f40857ad5f5b8/src/generator_gemm_common_aarch64.c)：`libxsmm_generator_gemm_aarch64_setup_blocking_sme`。
- [LIBXSMM AArch64 指令编码](https://github.com/libxsmm/libxsmm/blob/3057b239c16994d307589e40e45f40857ad5f5b8/src/generator_aarch64_instructions.c)：`libxsmm_aarch64_instruction_sme_compute`。
