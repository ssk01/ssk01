# 阅读导览：从 TF 学习框架到 primitive cache

这份导览解释本项目的结构，以及提交 `97c85a9` 增加的 JIT / primitive cache 功能。建议先读本文，再沿文件链接看实现；不用先读完所有历史版本或实测 JSON。

继续理解“为什么生成代码、收益来自哪里”，看 [oneDNN MatMul 代码生成与成本模型](JIT-MATMUL-COST-MODEL.md)。其中区分了源码事实、本机实测和教学估算。

## 1. 项目在做什么

`dl-from-zero` 是逐步实现 TensorFlow 架构概念的教学框架，不是 TensorFlow 源码或它的插件。已有版本逐步实现图、Session、梯度、优化和设备执行。本次在这个框架上增加二维 MatMul，以及可跨请求复用的计算准备对象。

| 目录 | 主要概念 | 阅读本次功能时的用途 |
| --- | --- | --- |
| `v0-linear` | Graph、Node、Tensor、Session 的最小实现 | 不熟悉框架时先建立概念 |
| `v1-prune`～`v6-quant` | 图裁剪、分布式、运行状态分离、梯度子图、图优化、量化 | 遇到相关问题再回看 |
| `v7-parallel`、`v8-graph-partition` | 执行队列、多设备、跨设备传输 | 理解框架的调度开销 |
| [v9-op-registry](v9-op-registry/) | 通过 registry 注册 op 和 kernel | v10 直接复用的框架底座 |
| [experiments/jit-matmul](experiments/jit-matmul/README.md) | 单独验证成熟 JIT kernel 的性能和创建成本 | 本次功能的实验依据 |
| [v10-primitive-cache](v10-primitive-cache/README.md) | primitive 对象、LRU、TF 图执行接入 | 本次实现的主线 |

v10 直接 include v9 的代码，并没有复制一套完整框架。这里的“TF demo”始终指本仓库的教学框架。

## 2. 先分清这三层各自负责什么

```text
教学框架：Graph / Session / Executor / KernelRegistry
    负责什么时候执行哪个算子、输入输出如何传递
                    ↓
本次实现：MatmulRuntime / PrimitiveCache / Primitive
    负责按描述找实现、管理代码的生命周期、统计命中和创建
                    ↓
LIBXSMM：GEMM dispatch / SME 代码生成器
    负责选择计算策略、生成和执行机器指令
```

**自己实现的是中间层和框架适配；主线使用 LIBXSMM 已有的生成器。** 没有把 oneDNN 接到 TF demo，也没有重新实现 oneDNN 的完整 primitive 系统。这里借鉴的是它“准备一次、执行多次”的思路。

JIT 的意义是根据运行时确定的尺寸、步长、数据类型和硬件生成专用实现。它本身不是加速指令；同样的优化也可以写进预编译 kernel。具体实现是否更快，仍要与成熟库对照。官方依据和案例出处集中在[规划文档](primitive-cache-plan.md)和[实验说明](experiments/jit-matmul/README.md)。

## 3. 这次具体改了哪些文件

| 文件 | 改动及原因 |
| --- | --- |
| [primitive-cache-plan.md](primitive-cache-plan.md) | 四阶段规划、官方依据、当前完成范围 |
| [experiments/jit-matmul/main.cpp](experiments/jit-matmul/main.cpp) | 对照 LIBXSMM 与 Accelerate，分别测热执行、创建、单独生成及缓存容量 |
| [v10/primitive.h](v10-primitive-cache/primitive.h) | 描述 key、代码所有权、执行对象、LRU、统计；核心实现 |
| [v10/tf_matmul.h](v10-primitive-cache/tf_matmul.h) | 注册 MatMul2D，把 Tensor 转为描述及当前数据指针，提供三个后端 |
| [v10/demo.cpp](v10-primitive-cache/demo.cpp) | 独立调用与真实 Session 调用、正确性和生命周期测试、计时入口 |
| [v10/cpu_only.cpp](v10-primitive-cache/cpu_only.cpp) | 让本次 CPU demo 不依赖 Metal shader 构建产物 |
| [v10/CMakeLists.txt](v10-primitive-cache/CMakeLists.txt) | 构建、可选 LIBXSMM 链接、CTest |
| 两处 `run_*.py` / `measured-results.json` | 多进程重复测量及原始结果；结论可追溯 |

v9 有四处小修改，都是为了正确接入新算子：

- [graph/graph.h](v9-op-registry/graph/graph.h)：追加 `MATMUL2D` 类型和 `Graph.matmul2d()`，选择 CPU。
- [kernels/kernels.h](v9-op-registry/kernels/kernels.h)：把新类型映射到注册名称 `MatMul2D`。
- [core/executor.h](v9-op-registry/core/executor.h)：把新类型归为昂贵计算节点。
- [graph/gradients.h](v9-op-registry/graph/gradients.h)：对尚未实现的二维梯度明确报错。

旧 `MATMUL` 是 `[M,K] × [K] → [M]`，新算子是 `[M,K] × [K,N] → [M,N]`。单独新增类型，是为了让图优化和求导能区分两者，避免公共子表达式消除（CSE）误合并。

提交里还保留了探索材料：`experiments/onednn-primitive-cache` 记录 oneDNN 本机路径验证，`jit-matvec` 是未纳入主线的小矩阵探索，`primitive-cache-case` 是 Metal 辅助实验。**第一遍阅读可以跳过它们**，不要把不同后端、尺寸或 dtype 的数字混在一起。

## 4. 按这个顺序读核心代码

### 第一步：看一次请求如何跑起来

打开 [demo.cpp](v10-primitive-cache/demo.cpp) 的 `run()`。它创建共享 `MatmulRuntime`，调用 `install()`，构建两个 placeholder 和一个 `matmul2d` 节点，然后设置一个 Session worker。

`request` lambda 有两个入口：`primitive` 模式直接调用 `runtime.compute()`；`tf` 模式调用真正的 `session.run()`。两者最终使用同一套 MatMul 实现，差别是后者经过图执行框架。这里的 `primitive` 模式也包含 Tensor 输出分配，并非只测裸函数调用。

### 第二步：看框架怎么进入计算后端

打开 [tf_matmul.h](v10-primitive-cache/tf_matmul.h)：

1. `install()` 注册 `MatMul2D` 及 CPU kernel，lambda 捕获 runtime 的 `shared_ptr`。
2. `MatmulRuntime::compute()` 检查 Tensor 的二维形状、连续布局和数据长度，并分配输出。
3. JIT 分支按 M/N/K、stride 和布局构造 `MatmulDesc`，调用 `cache.get()`。
4. 拿到 primitive 后，传入**本次** A/B/C 指针执行。其他分支使用 Accelerate 或标量循环。

TF Tensor 为行主序，后端通过 `Cᵀ = BᵀAᵀ` 映射到列主序 GEMM：交换 M/N 和 A/B 指针，不搬动数据。阶段一实验用列主序，v10 用行主序适配，所以两部分性能分别测量。

### 第三步：看 cache hit 和 miss 的差别

打开 [primitive.h](v10-primitive-cache/primitive.h)，先看底部 `PrimitiveCache::get()`，再向上读：

```text
get(desc)
  ├─ hit：把条目移到 LRU 头部 → 返回已有 Primitive
  └─ miss：compile_backend(desc) → 创建 Primitive → 插入 LRU
                                      容量不足时移除尾部条目
execute(A, B, C)
  └─ 调用 Primitive 持有的后端函数，用当前数据计算
```

这是便于阅读的 list 实现，查找是线性的，并非生产级 O(1) LRU。容量为 0 时不保留条目。

`MatmulDesc` 是 key，记录尺寸、步长、布局、dtype、转置、alpha/beta；当前只允许 FP32、无转置、alpha=1、beta=0。`Primitive` 保存描述及后端 owner；`Backend` 保存可执行函数指针、代码字节数，并在销毁时释放 kernel。

### 第四步：理解为什么 miss 不一定等于生成

LIBXSMM 自己也有 registry。若外层淘汰对象、后端却保留代码，再次 dispatch 可能只是后端命中，不能据此宣称重新生成了机器码。

因此 `compile_backend()` 既检查 registry 的变化，也检查 kernel metadata，排除 reference fallback。专用 LIBXSMM 构建开启 dereg，最后一个 owner 消失时真正释放代码。一个进程级弱引用池协调多个 cache 和调用者的所有权，防止同一函数被重复释放；弱引用本身不保留代码。

`misses` 表示外层没有该条目，`generated` 表示后端确实新增代码。若调用者仍持有被淘汰对象，新请求可能发生 miss，却复用存活 owner 而不生成。缓存容量限制条目数，不能当作代码内存的硬上限。

最后看 `demo.cpp` 的 `lifecycle()` 和 `main()` 测试分支：修改 A/B、淘汰后继续使用对象、跨 cache 共享、释放后再次生成，都有对应检查。

## 5. 缓存的东西，以及收益发生在哪里

比如 A 为 `144×512`、B 为 `512×128`。第一次查找时，根据描述生成可执行实现；后续尺寸和属性相同，即使 A/B 数值变了，也可以复用该实现。

缓存持有的是 **描述 + 代码所有权 + 执行入口**。代码本体由后端管理，并不只是省掉查找一个函数指针；更重要的是保留这个指针对应的可执行代码，避免销毁后再创建。A、B、C 以及计算结果不在缓存里，也没有另存一份预打包权重。

理解收益时分开看三个问题：

| 问题 | 已有证据 | 结论边界 |
| --- | --- | --- |
| 生成后的实现是否值得用？ | FP32 `144×512 × 512×144`，五进程倍率中位数 1.126×，对照 Accelerate | 证明本机存在有效案例，不证明所有矩阵都快，也不隔离所有算法差异 |
| 准备是否值得复用？ | 该 case 冷创建约 49.75 μs，单独生成约 3.618 μs | 完整创建包含其他工作；两项是独立诊断，不能相减当作精确内部 profile |
| 接入框架还有效吗？ | v10 混合 160 请求，容量 2→8：约 23.185→9.747 ms | 主要省掉重复准备；热缓存与 Accelerate 的 9.762 ms 基本持平 |

混合测试有 8 种描述，顺序重复回放。容量 2 留不住一轮工作集，因此预热后仍反复 miss；容量 8 可以全部保留，测量区间为 160 hit、0 miss、0 generated。容量改变时 kernel 执行时间接近，准备时间明显下降，这是缓存收益的直接证据。

输出中的 `prepare_us` 包含查找、dispatch、诊断和淘汰，不等于纯机器码生成。`execute_us` 记录 kernel 路径；`total_us` 记录完整请求，TF 模式还含 feed 拷贝、分配、调度和等待。框架开销解释了为什么单 kernel 的优势未必转成同等端到端收益。

## 6. 跑一遍最有帮助的对照

以下命令适用于本次验证过的 macOS / Apple M4 环境。从仓库根目录开始。先按[实验构建说明](experiments/jit-matmul/README.md#复现)准备固定版本、开启 dereg 的 LIBXSMM，默认放在 `/tmp/playgroud-libxsmm`。

```sh
cd 2026/tensorflow/dl-from-zero/v10-primitive-cache
cmake -S . -B /tmp/playgroud-v10-build \
  -DLIBXSMM_ROOT=/tmp/playgroud-libxsmm -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/playgroud-v10-build -j4
ctest --test-dir /tmp/playgroud-v10-build --output-on-failure

# 同一混合工作集：对照反复 miss 和充分缓存
VECLIB_MAXIMUM_THREADS=1 /tmp/playgroud-v10-build/primitive_demo tf jit 2 20 mixed
VECLIB_MAXIMUM_THREADS=1 /tmp/playgroud-v10-build/primitive_demo tf jit 8 20 mixed

# 对照成熟库的完整请求耗时
VECLIB_MAXIMUM_THREADS=1 /tmp/playgroud-v10-build/primitive_demo tf accelerate 0 20 mixed
```

先检查 `requests`、`hits`、`misses`、`generated` 和 `max_error`，再比较耗时。单次运行数字会波动；正式比较用 `run_bench.py` 重复测量。把 `tf` 换成 `primitive`，即可观察去掉 Session 路径后还有多少开销。

当前完成的是单请求线程、FP32、二维前向。BF16、二维反向、训练、并发 get/execute/release 尚未实现或验证；JIT 后端不可用时明确报错。继续扩展前，先保证能够解释 `get()`、`compile_backend()`、`execute()` 三段各自做什么，以及为什么外层 miss 和真实生成要分别统计。
