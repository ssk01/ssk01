# Arm JIT 候选与 M4 Pro 实测

日期：2026-10-09。源码范围固定为 oneDNN 3.9.1，commit `80a3a8e745d2f0186e674b0af9332fd6e074c94f`，不是对所有版本的支持承诺。

## 哪些可用

| 算子 | 此版本的 Arm JIT 路径 | 本机结果 |
| --- | --- | --- |
| MatMul BRGEMM | SVE-256/512 | 本机不满足；此前 FP32 走参考 GEMM，BF16 创建失败 |
| 卷积、Pooling | 多种 SVE 实现，具体要求依算子/类型而异 | 本轮未实测，不作为 M4 候选 |
| Eltwise、Softmax | SVE 路径；128-bit SVE 也需要 SVE | 本轮未实测，不可把 SVE-128 当 NEON |
| BatchNorm FP32 | 包含 ASIMD JIT | 选中 `bnorm_jit:asimd`、创建成功，执行出现 SIGILL；原因未定位 |
| Shuffle FP32 | 包含 ASIMD JIT | `jit:asimd`，创建、执行、逐元素验证通过 |
| Reorder FP32 | 通用 JIT 含 ASIMD 路径 | `jit:uni`，创建、执行、逐元素验证通过 |

ASIMD/NEON 路径不要求 SVE。ACL 后端属于另一套实现，本次未启用，不能由上述结果推断 ACL 性能。

源码入口：[BatchNorm 列表](https://github.com/uxlfoundation/oneDNN/blob/80a3a8e745d2f0186e674b0af9332fd6e074c94f/src/cpu/cpu_batch_normalization_list.cpp)、[Shuffle 列表](https://github.com/uxlfoundation/oneDNN/blob/80a3a8e745d2f0186e674b0af9332fd6e074c94f/src/cpu/cpu_shuffle_list.cpp)、[Eltwise 列表](https://github.com/uxlfoundation/oneDNN/blob/80a3a8e745d2f0186e674b0af9332fd6e074c94f/src/cpu/cpu_eltwise_list.cpp)。

## 真正生成了多少代码

另开独立进程，设置 `ONEDNN_JIT_DUMP=1`，记录生成的机器码文件。该诊断运行不用于计时。

| 算子 | 容量 2：生成文件数/累计字节 | 容量 256：生成文件数/累计字节 |
| --- | --- | --- |
| Shuffle | 168 / 276,864 | 8 / 13,184 |
| Reorder | 168 / 2,236,080 | 8 / 106,480 |

总请求为 8 次预热加 160 次测量。大容量的 8 个文件全部在预热时生成，测量期 160 次命中；小容量则在测量期额外生成 160 份代码。累计字节是所有生成事件的总量，不是同时驻留内存大小，也不表示每份代码内容都不同。

源码也对应这一现象：Shuffle 的 `init()` 先预计算输入偏移表，再创建 JIT kernel 并调用 `create_kernel()`；Reorder 的 `init()` 创建转换 kernel 并生成代码。两者的 primitive cache 命中都跳过这段初始化。

Shuffle 因而还展示了一个额外点：缓存对象可以持有根据布局预计算的偏移表。这是执行元数据，不是输入张量或权重数值。

源码：[Shuffle 初始化和偏移表](https://github.com/uxlfoundation/oneDNN/blob/80a3a8e745d2f0186e674b0af9332fd6e074c94f/src/cpu/aarch64/shuffle/jit_uni_shuffle.cpp)、[Reorder 初始化](https://github.com/uxlfoundation/oneDNN/blob/80a3a8e745d2f0186e674b0af9332fd6e074c94f/src/cpu/aarch64/jit_uni_reorder.cpp)。

## 创建成本实测

5 次独立进程、交替容量顺序、关闭 verbose 和 JIT dump；各列取中位数。每个进程先预热 8 个描述，再回放 20 轮，共 160 次。输入 buffer 预先准备，每次执行后在计时外逐元素验证。

- Shuffle：`[2,32,H,8]`，H=8..15，`nChw8c`，沿 C 维 shuffle，group_size=4。
- Reorder：`[32,H*8]`，H=8..15，连续 ab 转为 ba。

| 算子 | 容量 | primitive 创建 ms | descriptor 创建 ms | execute/wait ms | 请求总 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shuffle | 2 | 1.190 | 0.197 | 0.986 | 2.392 |
| Shuffle | 256 | 0.039 | 0.066 | 0.482 | 0.605 |
| Reorder | 2 | 1.929 | 0.177 | 0.648 | 2.785 |
| Reorder | 256 | 0.034 | 0.027 | 0.145 | 0.218 |

单次 primitive 创建阶段的均值约为 Shuffle 7.44 → 0.24 μs、Reorder 12.05 → 0.22 μs。独立 verbose 运行确认两个算子都是小容量 160 miss、大容量 160 hit，执行次数相同。

这是很小的串行微基准。execute/wait 也出现变化，可能涉及新生成代码的首次执行、缓存状态及计时噪声，尚未细分。因此总请求差值不能全部归因于代码生成；JIT dump 次数和独立创建阶段耗时才是直接证据。原始数据见 [arm-measured-results.json](arm-measured-results.json)。

## 复现

在 README 的 oneDNN 构建基础上增加三个算子。这个版本已有 CMake cache 时，旧 DNNL 前缀与新 ONEDNN 前缀可能同时存在，以下显式设置两者：

```sh
cmake -S /tmp/playgroud-onednn-v3.9.1 -B /tmp/playgroud-onednn-build \
  '-DONEDNN_ENABLE_PRIMITIVE=MATMUL;BATCH_NORMALIZATION;SHUFFLE;REORDER' \
  '-DDNNL_ENABLE_PRIMITIVE=MATMUL;BATCH_NORMALIZATION;SHUFFLE;REORDER'
cmake --build /tmp/playgroud-onednn-build -j 8
cmake --install /tmp/playgroud-onednn-build
cmake --build build -j 4
python3 run_arm_cases.py build/arm_jit_probe
```

`arm_jit_probe reorder 2` 和 `arm_jit_probe shuffle 256` 可单独运行。BatchNorm 探针保留用于记录失败，但自动计时脚本不运行它。

JIT dump 会写入进程工作目录，建议在独立空目录运行绝对路径的程序，分别统计两种容量，避免混入旧文件。

## 对 demo 的建议

优先使用 **Reorder**：转置/布局变换容易理解，输入输出能直接验证，本机已经证明 miss 会重新生成代码。Shuffle 作为第二个例子，补充预计算偏移表的复用。

现在已有本机可运行的真实 oneDNN JIT case，无需为了展示 primitive cache 而先自写代码生成器。若另有学习 JIT 编码和 ABI 的目标，再单独做教学生成器。以上都是 FP32 的 Arm case，不能等同于其他平台的 BF16 MatMul 性能。
