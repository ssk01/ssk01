# 从现成案例验证 MatMul JIT 与代码缓存

本实验沿用 LIBXSMM 的 SME GEMM 生成器，比较系统 Accelerate SGEMM；没有自行实现矩阵计算或指令生成算法。主矩阵的 M、N、K 均大于 100。数据为 FP32，输出和累加也为 FP32；不作为 BF16 的结果。

## 来源与设计理由

[Hello SME! 原作者论文，第 IV 节](https://arxiv.org/html/2409.18779v1#S4) 在 M4 上比较 LIBXSMM 生成的 SME GEMM 与 Accelerate，覆盖 K=512、M=N 从 1 到 512。作者按矩阵描述固化代码，组合寄存器分块并处理布局。本实验复用项目实现，在当前 M4 Pro/macOS 上重新测量。

[oneDNN MatMul 文档](https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html) 说明创建时已知尺寸和布局有利于优化；执行时才提供尺寸则牺牲部分优化机会。[ukernel 文档](https://uxlfoundation.github.io/oneDNN/dev_guide_ukernel_basic_concepts.html) 将配置、代码生成、执行分开，建议将生成移出计算循环。

JIT 的作用是为运行时才知道的描述生成专用实现。相同优化也可以预编译；这个实验比较的是生成后的 LIBXSMM kernel 与系统 GEMM 的整体实现，不是隔离了所有其他差异的“JIT 时机”实验，也没有证据断言 Accelerate 内部完全不使用运行时代码生成。

## 环境与执行对照

- Apple M4 Pro，具体 macOS 信息保存在 `measured-results.json`。
- LIBXSMM 固定提交 `3057b239c16994d307589e40e45f40857ad5f5b8`，运行时 target 为 `appl_m4`。
- A/B/C 均列主序，无转置，alpha=1、beta=0。相同缓冲区、同样输入，无预先打包的权重。生成 kernel 自己处理所需的数据搬运。
- 单请求线程，Accelerate 设置 `VECLIB_MAXIMUM_THREADS=1`。不宣称操作系统严格绑核。
- 每组五个独立进程；进程内 15 轮，轮换两种实现的先后顺序，每轮预热后批量执行 100 次。表中为进程结果的中位数。
- 验证完整输出，也验证同 shape 修改权重后的输出。当前所有对照最大绝对误差为 0。通过 kernel metadata 确认不是 reference fallback。独立 `verify` 又对 128、129、144、176、256 的方形输出（K=512）逐元素对照 double 标量参考，并改变 A/B 重测；最大绝对误差 0.000031163。

| M=N，K=512 | LIBXSMM 热执行 μs | Accelerate μs | 五次倍率中位数 |
| ---: | ---: | ---: | ---: |
| 112 | 8.701 | 10.292 | 1.168× |
| 128 | 10.462 | 11.364 | 1.080× |
| 144 | 13.555 | 15.218 | 1.126× |
| 160 | 15.654 | 16.530 | 1.057× |
| 176 | 19.231 | 21.461 | 1.113× |
| 192 | 22.073 | 23.142 | 1.051× |
| 256 | 38.487 | 39.264 | 1.022× |
| 384 | 83.075 | 85.967 | 1.035× |
| 512 | 151.375 | 153.441 | 1.014× |

选择 144×512 乘 512×144 为主 case：五次倍率范围 1.115–1.205，每次都更快。大尺寸收益明显变小，512 的约 1% 差异不作为强性能结论。各列分别取中位数，因此列间相除不一定等于倍率中位数。

这证明当前平台存在大于 100 的有效案例，但没有重现原论文所有幅度，也不意味着 JIT 对所有矩阵都更快。

## 创建成本究竟是什么

主 case 的第一次 dispatch 创建时间中位数约 49.75 μs，代码为 4700 字节。它包含查找、生成、内存管理和代码可执行化等工作，不能全称为机器指令生成耗时。

额外的 `generator` 模式直接调用项目的 `libxsmm_generator_gemm_kernel`，把二进制生成到已分配的普通缓冲区，计时不包含分配、发布和执行。预热 10 次后生成 800 次；五进程中位数为每次 **3.618 μs**。生成内容逐字节匹配 dispatch 返回的 4700 字节代码。

这是独立诊断，不是同一次创建的内部 profile，不能拿二者相减就给剩余阶段分配精确比例。它足以说明：这个案例的完整创建成本远不止指令编码。按冷创建约 50 μs 和热执行差约 1.66 μs 粗估，约数十次复用才摊平初次创建；实际回本还受缓存查找和首次执行影响。

## 缓存容量实验

使用相同工作负载：M 依次为 128、144、160、176、192、208、224、240，N=128、K=512。完整预热一轮，再回放 20 轮，共 160 次请求。各进程分别关闭缓存、容量 2、容量 8。数值检查和 checksum 在计时之外。

| 容量 | 测量区间 hit/miss | 准备累计 ms | 执行累计 ms | 完整请求累计 ms |
| ---: | --- | ---: | ---: | ---: |
| 0 | 0 / 160 | 12.460 | 2.956 | 15.823 |
| 2 | 0 / 160 | 12.937 | 2.646 | 15.533 |
| 8 | 160 / 0 | 0.0069 | 2.696 | 2.703 |

五次独立进程的中位数，分别聚合，各列不保证恰好相加。小容量到足够容量的总耗时降低约 **82.6%**，执行时间接近，改善主要发生在准备路径。关闭/小容量均重新产生 463680 字节代码，容量 8 在测量区间生成 0 字节。这里的生成字节为累计量，不是常驻内存量。

LIBXSMM 自带 registry/cache。本实验在单线程专用构建中启用 `LIBXSMM_ENABLE_DEREG`，淘汰时真正释放后端 kernel。每次外层 miss 检查后端 registry 增加一项，清理后检查归零，避免隐藏的后端命中。这不是在声称 LIBXSMM 默认采用此容量 LRU，也不是 oneDNN 的容量实验。

prepare 包含外层查找、dispatch、诊断查询及可能的淘汰；total 还包含执行和关闭缓存时的释放。此阶段只是实验性的代码保留逻辑，尚未完成 primitive 抽象。

## 复现

```sh
git clone https://github.com/libxsmm/libxsmm.git /tmp/playgroud-libxsmm
git -C /tmp/playgroud-libxsmm checkout 3057b239c16994d307589e40e45f40857ad5f5b8
make -C /tmp/playgroud-libxsmm -j8 STATIC=1 BLAS=0 \
  DFLAGS='-DNDEBUG -DLIBXSMM_ENABLE_DEREG' WERROR_CFLAG=

cmake -S . -B /tmp/playgroud-libxsmm-case-build \
  -DLIBXSMM_ROOT=/tmp/playgroud-libxsmm -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/playgroud-libxsmm-case-build -j4
/tmp/playgroud-libxsmm-case-build/jit_matmul verify
python3 run_cases.py /tmp/playgroud-libxsmm-case-build/jit_matmul \
  --out /tmp/playgroud-libxsmm-case-results
/tmp/playgroud-libxsmm-case-build/jit_matmul generator 144 144 512
```

从本目录运行 CMake 和脚本。本提交开启 dereg 的路径有一个仅供断言使用的局部变量：需使用 Release 的 NDEBUG，且取消 warning-as-error 才能通过其 unused-variable 警告；没有修改第三方源码。这个专用构建只用于单线程教学实验，不当作已验证的并发 kernel 释放方案。

最初还直接运行了项目自带 `samples/xgemm/gemm_kernel`，128×512 乘 512×128 的数值检查通过；上表来自本目录的对照驱动，而非复制论文数字。

## 阶段三、四实现

阶段一完成了合适规模的实际库对照，阶段二完成了真实重复创建与容量对照。GPT-6.1 Sol 子 agent 已完成 [primitive demo 与 TF 二维图执行](../../v10-primitive-cache/README.md)，包括数值、生命周期、容量与完整 Session 请求实测。原阶段一、二数据保持不变。实现遵循以下交接要求：

- 沿用已测 LIBXSMM 后端，暂不自写生成器；BF16 另列平台验证，不冒充已完成。
- Primitive 持有描述、生成代码所有权和执行入口，输入/权重/输出由每次执行传入。
- 处理 LIBXSMM registry 共享所有权，禁止两个对象重复释放同一后端指针；缓存淘汰也不能释放仍被调用者持有的 primitive。
- 保留真实生成与缓存命中计数，拒绝 reference fallback，并验证改数据不重编译。
- TF 当前主要是矩阵向量语义；完整二维 MatMul 要明确新增接口或 op，保留既有行为，不能悄悄改变训练语义。
- 完成正确性、生命周期、缓存容量和框架端到端验证；不预先承诺框架加速比例。
