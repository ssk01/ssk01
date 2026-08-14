# Socrates.md - 问答记录

### Q: VARIABLE 是 wx+b 里的 x 吗? SGD_STEP 作为 node type 是不是太抽象了?
两个都不是。① VARIABLE 不是输入 x——x 是 PLACEHOLDER(run 时喂数据, 图不拥有); VARIABLE 是 w/b 这类**可训练状态**: 值存活在 Session 的 vars_ map 里, 被 SGD_STEP 原地改写, run 之间持续存在 (对应 TF Variable 语义)。② SGD_STEP 不抽象, 它就是 TF 的真身: TF 里优化器更新就是真实图 op ApplyGradientDescent(var, grad, lr)。为什么更新必须是图节点而非图外命令式代码——v2 分布式逼出来的 (更新要在 PS 端执行, 做成节点 PS 跑同一张图即可); v4 反向子图顺水推舟 (训练 loop 变成纯一次 run); v5 CSE 必须要能识别它: is_stateful 就是给优化器看的, 有副作用节点不能合并/折叠 (两个无输入的 VARIABLE 哈希相同, 合并会把两个独立参数变成同一个; SGD_STEP 合并/折叠会吞掉更新)。demo 简化: SGD_STEP 执行是 Session 特判的, 没走统一 kernel 分派, 但节点类型层面语义与 TF 一致。
(2026-08-14)

### Q: TF2 里 ApplyGradientDescent 还是 op 吗? 和我们的 sgd_step 差在哪?
还是 op, 就在 tf.keras.optimizers.SGD 源码里: `gen_training_ops.ResourceApplyGradientDescent(var=var.handle, alpha=lr_t, delta=grad)` (master 的 optimizer_v2/gradient_descent.py, momentum 时用 ResourceApplyKerasMomentum)。变体从 TF1 的 ApplyGradientDescent (Ref 可变张量) 换成了 resource 变体 (var.handle 句柄)——正好对应 v3 的故事: 状态从图里张量变成外部句柄; 我们的 VARIABLE 值存 Session 的 vars_ map, 结构上已接近 TF2 resource 设计。与 sgd_step 的差别: ① TF 的 lr 是动态输入 tensor (调度器改的是它), 我们是编译期 static scalar, 换学习率要重建图; ② TF 走真 kernel (training_ops.cc 统一分派), 我们 Session 特判 (简化); ③ TF2 变量是资源句柄 + 显式读写 op。op 没死, 只是换了形态。
(2026-08-14)

### Q: TF1 图优化除了 CSE/常量折叠还有啥?
分两层回答, 用户追问后确认问的是**首个 commit 里的优化** (不是 Grappler):
- **首个 commit (f41959ccb, 2015-11-07) 实际只有**: ① `optimizer_cse.cc` CSE pass——写好了但**没接入任何执行路径** (local_session/executor/客户端全都不引用, 只有单测); ② `subgraph.cc` 的 `RewriteGraphForExecution`——按 fetch/target 只跑图的一部分 (v1-prune 的源头, 唯一真正接线的图变换); ③ **没有常量折叠** (tree 里无此文件)。后续: 常量折叠 2016-01-25 加入 (71184628, 常量子图拷出去用本地 executor 跑); 2016 年底 `common_runtime/graph_optimizer.cc` 的 `GraphOptimizer::Optimize` 先折叠后 CSE, 由 direct_session 调用——"Session run 前自动跑 pass"至此才成型。
- **TF1 后期 Grappler 全家桶** (TF1.9 左右加入, 1.14 起默认开启): constant_folding (已做); arithmetic_optimizer (CSE 住在这里 + 算术化简, CSE 已做); model_pruner (删 Identity/NoOp/未用节点——v1-prune 的 Grappler 形态); dependency_optimizer (删冗余 control dep 缩关键路径); remapper (子图融合 Conv+Bias+ReLU → _FusedConv2D、MatMul+BiasAdd、BN 折叠——推理加速最大功臣, 没做); layout_optimizer (NHWC↔NCHW GPU 卷积); loop_optimizer (循环不变量外提/强度削减——对应控制流候选); memory_optimizer (重算换内存 recompute / GPU↔CPU swap); function_optimizer (函数内联——对应 Function 子图候选); auto_mixed_precision (插 fp16 cast——v6 量化的同族兄弟); 其他: shape_optimizer/pin_to_host/scoped_allocator/debug_stripper/autoparallel。v1-prune 和 v5-CSE 是同一思想家族两端: "删掉不影响输出的节点"从 v1 到 Grappler 都是核心优化。
(2026-08-14)

<!-- 以下继续记录 -->
