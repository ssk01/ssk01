# Primitive 对象、LRU 与 TF 图执行

阶段三、四的可运行教学实现。复用 [阶段一、二验证过的 LIBXSMM 后端](../experiments/jit-matmul/README.md)，自己实现对象所有权、描述 key、LRU 和统计；不重新实现机器码生成器。仅合成 FP32 数据，单请求线程，M4 Pro 的 `appl_m4` 目标。没有 BF16、训练或并发请求结论。

## 执行链与文件

`Graph.matmul2d → MATMUL2D → KernelRegistry → MatmulRuntime → PrimitiveCache.get → compile_backend → Primitive.execute(当前 A,B,C)`。

- `primitive.h`：描述校验、后端生成与 RAII owner、primitive、容量可调 LRU。
- `tf_matmul.h`：注册二维 op/kernel，选择 scalar、Accelerate 或 JIT，共享 runtime 生命周期。
- `demo.cpp`：独立 primitive 请求、真实 v9 Session 图执行、正确性与生命周期测试、计时。
- `cpu_only.cpp`：此 CPU 演示显式关闭 Metal，不依赖 shader 构建产物。
- `run_bench.py` 与 `measured-results.json`：五个独立进程/配置的实测与中位数。

直接 include v9 源码，未复制头文件或构建产物。v9 仅追加独立 `MATMUL2D` 枚举与 `Graph.matmul2d` API、执行名称映射、昂贵节点标记以及显式拒绝二维梯度；原 `MATMUL` 的 `[M,K]×[K]→[M]` 和 kernel 保留。独立类型使 CSE 不会合并二维与 matvec 算子；常量折叠经相同注册 kernel 求值。v9 不安装二维 kernel 时，执行它会明确报缺失注册。

## 描述、布局与数据

`MatmulDesc` key 包含 M/N/K、lda/ldb/ldc、布局、dtype、转置标志、alpha/beta。仅支持正尺寸、FP32、无转置、alpha=1、beta=0，其他属性拒绝；target 固定为进程初始化的 LIBXSMM target，运行中不得切换目标架构。后端/生成策略固定，因此不需要可变后端字段。stride 是以 float 元素为单位的 leading dimension。

Primitive 可接受列主序或行主序。TF Tensor 只接受连续行主序 `[M,K] × [K,N] → [M,N]`，检查 rank、尺寸和数据长度。利用 `Cᵀ=BᵀAᵀ`，将行主序映射到 LIBXSMM 列主序 shape `(N,M,K)` 并交换 A/B 指针；没有实际转置、数据拷贝或权重预打包。故此处性能必须重测，不能套用前两阶段列主序数字。

缓存保存配置和代码，绝不保存输入、权重或结果；每次 execute 接收当前 A/B/C，调用者保证缓冲区大小匹配描述、输出与输入不重叠。Primitive 构造受 cache factory 控制，Backend owner 禁止复制。

## 两层所有权与真实生成

外层 LRU 用 shared_ptr。淘汰只移除缓存持有的引用，调用者仍持有对象时可以继续执行。进程级弱所有权池保证同描述及归一化后相同后端指针只有一个真正 Backend owner；跨缓存复用不会双重释放。最后一个 owner 销毁才 deregister/release。

`hits/misses` 是外层 LRU 查找统计；`generated` 只在 LIBXSMM registry 实际新增一项、kernel metadata 确认 generated/non-reference 后增加。存活 owner 的弱池命中可能使 miss 不生成，专门有测试覆盖。找不到 owner 却发生 registry 命中时直接报错，避免后端隐藏缓存；因此须使用独立进程、dereg-enabled 专用构建，不能混入其他 LIBXSMM 用户。池定期清除失效弱项，不保留代码。容量限制是 cache 条目数，调用者另持有对象时真实常驻代码可以超过容量。

整个 API 不支持并发 get/execute/release。TF demo 使用一个 Session worker、串行请求。注册 kernel 捕获 runtime shared_ptr，调用者释放它后仍有效；重新 install 会替换该注册，不得在图运行期间更换。没有自动回退：JIT 未编译、dispatch 失败、reference kernel 或不可诊断 registry 均报错。

## 复现与验证

先按阶段一 README 构建固定 SHA `3057b239c16994d307589e40e45f40857ad5f5b8` 的 LIBXSMM：`STATIC=1 BLAS=0 DFLAGS='-DNDEBUG -DLIBXSMM_ENABLE_DEREG' WERROR_CFLAG=`。然后在本目录运行：

```sh
cmake -S . -B /tmp/playgroud-v10-build \
  -DLIBXSMM_ROOT=/tmp/playgroud-libxsmm -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/playgroud-v10-build -j4
ctest --test-dir /tmp/playgroud-v10-build --output-on-failure
VECLIB_MAXIMUM_THREADS=1 /tmp/playgroud-v10-build/primitive_demo tf jit 8 20 mixed
VECLIB_MAXIMUM_THREADS=1 /tmp/playgroud-v10-build/primitive_demo tf accelerate 0 20 fixed
python3 run_bench.py /tmp/playgroud-v10-build/primitive_demo > /tmp/v10-results.json
```

CLI 为 `primitive|tf jit|accelerate|scalar capacity rounds fixed|mixed`。不设置 LIBXSMM_ROOT 可构建 scalar/Accelerate 版本；选择 jit 会返回非零错误，不隐式回退。C++17/macOS Accelerate，CPU-only，无 Metal SDK shader 依赖。

CTest 已通过：完整输出对 Accelerate、修改 A/B 对 double 标量参考、同 shape 更换当前缓冲区不重新生成、淘汰后持有对象仍执行、两个 cache 和等价 row/column 描述共享 owner、最终 registry 归零、释放后再生成、容量 0/2/8 的真实生成计数、固定 shape 热命中、真实 Session 更换权重、旧 matvec 数值、CSE 区分二维与 matvec、常量二维图折叠、runtime 共享生命周期、二维求导显式拒绝、非法 beta 拒绝。无 LIBXSMM 构建及其 CTest 也通过。

## 当前实测

2026-10-09，M4 Pro；五个独立进程/配置，顺序交替，Accelerate 线程上限 1。固定工作负载为 `144×512 × 512×128`，20 次请求；混合为 M=128/144/160/176/192/208/224/240、K=512、N=128，20 轮共160次。所有维度大于100。每项先跑完整工作集预热；所有数值检查在计时外，输出最大对照误差 0。生命周期测试另有独立 double 参考容差 `2e-4*(1+|reference|)`。

| 路径/混合容量 | hit/miss/真实生成 | 准备 ms | kernel ms | 完整请求 ms |
|---|---|---:|---:|---:|
| primitive / 0 | 0/160/160 | 13.027 | 3.050 | 16.653 |
| primitive / 2 | 0/160/160 | 13.368 | 2.891 | 16.390 |
| primitive / 8 | 160/0/0 | 0.0039 | 2.855 | 2.965 |
| TF / 0 | 0/160/160 | 12.889 | 2.862 | 22.834 |
| TF / 2 | 0/160/160 | 13.443 | 2.786 | 23.185 |
| TF / 8 | 160/0/0 | 0.0100 | 2.811 | 9.747 |
| TF / Accelerate | — | 0.0026 | 2.931 | 9.762 |

小容量到容量8：primitive 完整请求降低约81.9%，TF 完整请求降低约58.0%。关闭和小容量各真实生成160次、累计572320字节；容量8测量期间生成0字节。该字节数是累计生成量，不是内存峰值。

固定 shape 的 TF JIT 容量8：20 hit/0 miss/0生成，kernel累计0.287 ms、完整请求1.046 ms；Accelerate kernel累计0.299 ms、完整请求1.065 ms。差异很小，不作为明确端到端加速结论。混合 TF 热缓存 JIT 与 Accelerate 完整请求几乎持平。主要成果是避免重复准备，并非承诺比成熟通用库明显更快。

prepare 计时覆盖查找、dispatch、metadata/registry 诊断和淘汰，不称为纯机器指令生成时间。execute 仅 kernel；primitive complete 包括输出分配、返回和容量0最终 release；TF complete 包含 feed Tensor 构造/拷贝、Session 的执行计划查找、槽位和输出分配、worker调度/等待、注册 kernel、缓存准备、执行及输出返回。checksum 与数值检查在完整请求计时外。各列独立取中位数，不保证逐列相加。缓存命中收益会被这些框架成本稀释，当前数据清楚显示了这一点。
