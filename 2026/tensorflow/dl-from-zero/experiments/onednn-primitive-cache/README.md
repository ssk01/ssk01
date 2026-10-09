# oneDNN primitive cache 最小实验

本实验调用真实 oneDNN MatMul，对比固定 shape、混合 shape、小容量、大容量和关闭缓存。程序每次请求都创建 descriptor/primitive，执行完成后销毁外层对象，让 oneDNN 内部缓存决定是否复用准备结果。

后续已找到本机可运行的真实 JIT 算子：Shuffle 与 Reorder，包含缓存对照和机器码导出证据，见 [ARM-JIT.md](ARM-JIT.md)。下文 MatMul 的结论只针对该算子及其构建配置。

## 本机结果

2026-10-09，Apple M4 Pro，macOS arm64，AppleClang 17，oneDNN **3.9.1**，commit `80a3a8e745d2f0186e674b0af9332fd6e074c94f`。CPU runtime=SEQ，GPU=NONE，ACL=OFF，未链接 CBLAS。只构建 MATMUL，primitive cache 开启。选择固定版本是为了可复现，不代表最新版本或其他构建的表现。

FP32，连续布局，K=N=256；混合 M=16/24/32/40/48/56/64/72。预热一个完整工作集，随后测 200 次请求。固定组 M=40，也测 200 次。每种模式运行 5 个独立进程，交替顺序，性能运行关闭 verbose。

| 模式 | 容量 | 诊断 hit/miss | prepare 累计 ms | execute/wait 累计 ms | request 累计 ms |
| --- | ---: | --- | ---: | ---: | ---: |
| fixed | 2 | 200 / 0 | 0.223 | 46.247 | 46.469 |
| mixed-small | 2 | 0 / 200 | 0.997 | 31.369 | 32.376 |
| mixed-large | 256 | 200 / 0 | 0.197 | 31.547 | 31.767 |
| mixed-off | 0 | 0 / 200 | 0.913 | 32.169 | 33.144 |

耗时取 5 次运行的各列中位数，列间不要求严格相加。hit/miss 来自另一次打开 verbose 的诊断进程，每组执行次数都为 200；诊断耗时不用于性能结论。fixed 的计算内容与混合组不同，不能据总时间比较缓存收益。

混合组增大容量后，准备时间约从 4.98 μs/请求降到 0.98 μs/请求；总时间中位数只下降约 1.9%。这个量级不支持“miss 是主要瓶颈”的结论。缓存命中现象已复现，但昂贵 JIT 反复生成没有在这个配置下复现。

所有 shape 在预热时及测量完成后逐元素验证，测量期间每个请求额外抽查三个位置。权重每请求变化。最大绝对误差为 0，三种混合模式 checksum 一致。数据与逐进程结果保存在 [measured-results.json](measured-results.json)。

## 为什么没有明显 JIT 创建成本

verbose 打印 `gemm:jit:f32`，但这个名字在 `gemm_f32_matmul.hpp` 中是固定字符串，不能单凭名字判断当前平台生成了代码。

核对本次锁定版本及构建配置：

1. Arm 的 BRGEMM MatMul 候选需要 SVE，本机选择 `gemm_f32_matmul_t`。
2. 本例无 bias/post-op、输出直接作为累加结果，`has_pp_kernel_` 为 false；primitive `init()` 直接返回成功，不生成后处理 kernel。
3. `extended_sgemm()` 的 x64 JIT 分支仅在 DNNL_X64 下编译；本次也没有 USE_CBLAS，最终调用 `ref_gemm<float>`。

因此本次 miss 主要重复 descriptor/实现选择及轻量对象准备。不能把测到的 prepare 时间称为机器码生成时间。

## BF16 补测

程序增加了 `bf16`（BF16 输入、权重、输出）和 `bf16-f32`（BF16 输入、权重，FP32 输出）两种模式。输入从 FP32 按 round-to-nearest-even 转换到真实的 16 位 BF16 存储，没有将 FP32 buffer 冒充 BF16。

在上述本机版本/构建上，两种模式都在创建 MatMul descriptor 时失败。`ONEDNN_VERBOSE=all` 显示 SVE BRGEMM 候选被 `unsupported isa` 拒绝，BF16 GEMM 和参考实现也被数据类型支持检查拒绝。因此没有 BF16 执行耗时或缓存对照结果；前面的 FP32 结果不能用于推断 BF16 的创建成本。这只描述本次配置，不意味着所有 Arm 或所有 oneDNN 版本都不支持 BF16。

支持相应实现的环境可执行：

```sh
python3 run_cases.py build/cache_case --dtype bf16 --out results/bf16
python3 run_cases.py build/cache_case --dtype bf16-f32 --out results/bf16-f32
# 单独诊断不支持原因
ONEDNN_VERBOSE=all build/cache_case mixed 2 2 256 256 bf16
```

BF16 模式的实际执行和正确性检查尚未在支持该模式的平台验证。FP32 路径在增加类型参数后已重新构建并通过数值验证。

固定版本源码：

- [实现列表](https://github.com/uxlfoundation/oneDNN/blob/80a3a8e745d2f0186e674b0af9332fd6e074c94f/src/cpu/matmul/cpu_matmul_list.cpp)
- [名称与 primitive init](https://github.com/uxlfoundation/oneDNN/blob/80a3a8e745d2f0186e674b0af9332fd6e074c94f/src/cpu/matmul/gemm_f32_matmul.hpp)
- [属性配置](https://github.com/uxlfoundation/oneDNN/blob/80a3a8e745d2f0186e674b0af9332fd6e074c94f/src/cpu/matmul/gemm_f32_matmul.cpp)
- [GEMM 分派](https://github.com/uxlfoundation/oneDNN/blob/80a3a8e745d2f0186e674b0af9332fd6e074c94f/src/cpu/gemm/gemm.cpp)

## 复现

依赖需要 C++17 编译器、CMake、Python 3 和 oneDNN。以下命令将源码、构建产物和库安装到 /tmp，适用于本次环境；如果对应路径已有其他内容，请换成新的目录。

```sh
git clone --depth 1 --branch v3.9.1 https://github.com/uxlfoundation/oneDNN.git /tmp/playgroud-onednn-v3.9.1
cmake -S /tmp/playgroud-onednn-v3.9.1 -B /tmp/playgroud-onednn-build \
  -DCMAKE_BUILD_TYPE=Release -DONEDNN_CPU_RUNTIME=SEQ \
  -DONEDNN_GPU_RUNTIME=NONE -DONEDNN_BUILD_TESTS=OFF \
  -DONEDNN_BUILD_EXAMPLES=OFF -DONEDNN_BUILD_GRAPH=OFF \
  -DONEDNN_ENABLE_PRIMITIVE=MATMUL -DONEDNN_LIBRARY_TYPE=SHARED \
  -DCMAKE_INSTALL_PREFIX=/tmp/playgroud-onednn-install
cmake --build /tmp/playgroud-onednn-build -j 8
cmake --install /tmp/playgroud-onednn-build
```

在本实验目录执行：

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/tmp/playgroud-onednn-install
cmake --build build -j 4
python3 run_cases.py build/cache_case
```

或者单独执行：

```sh
# 参数：模式 容量 回放轮数 K N
ONEDNN_VERBOSE=0 build/cache_case mixed 2 25 256 256
ONEDNN_VERBOSE=0 build/cache_case mixed 256 25 256 256
ONEDNN_VERBOSE=2 build/cache_case mixed 2 25 256 256
```

run_cases.py 会保存每次输出和 JSON 汇总；诊断解析只统计 MEASURE_BEGIN/END 中的事件。已验证 oneDNN 3.9.1 的 verbose v1 格式；事件数量或格式不符合预期时失败退出，避免误报 0 miss。

prepare 包括 descriptor 和 primitive 两段，JSON 分别记录 pd_ms 与 primitive_ms。request 还包括当前对象销毁；输入 buffer、memory 对象预先准备，数值检查在计时区间外。计时只做单请求串行执行，不代表完整 TF Session 或服务延迟。

## 是否自己实现代码生成

如果目标是展示真实 oneDNN 的昂贵 miss，优先把这个程序放到实际支持目标 JIT 实现的机器上，并核对实际实现和创建路径。无需重写 oneDNN。

如果目标是理解代码生成本身，可以另做一个小型教学 JIT：只支持固定 K 的 FP32 dot product，使用汇编生成器产生直线/分块的 NEON 指令，缓存生成对象，并分别测代码生成、执行和复用。先处理好 macOS 可执行内存、ABI、寄存器保存及指令缓存一致性，再增加 shape 变化与淘汰。这个教学任务不承诺性能优于编译器，也不以额外空转制造昂贵 miss。

本次已完成真实 oneDNN 实验，尚未实现教学 JIT，也未接入 TF demo。
