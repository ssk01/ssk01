# Socrates

### Q: GoogLeNet 多大？
本例使用的 torchvision GoogLeNet 完整推理模型有 6,624,904 个参数，FP32 参数约 25.27 MiB。官方权重文件约 49.73 MiB，包含辅助分类头等额外权重。本例截取至 inception4d 的特征提取器有 2,470,080 个参数。DeepDream 固定这些参数，只通过梯度上升修改输入图像；运行时还需要保存中间激活和梯度。
(2026-10-30)

### Q: 确认 DeepDream 是这个意思吗？
核对 Google 原始开源 Notebook：https://github.com/google/deepdream/blob/master/dream.ipynb 。原版固定 ImageNet GoogLeNet，最大化某隐藏层激活的 L2 目标，通过梯度上升修改输入图片，并使用随机平移、梯度幅度归一化、多尺度处理。本项目的激活平方均值目标与原版平方和目标只差正比例常数，梯度归一化后方向一致；属于同类算法的 PyTorch 实现。权重、选层和超参数不同，结果不会逐像素一致。动物形态是学到的视觉特征被反复增强的结果，具体形态依赖原图和选层。
(2026-10-30)


补充：DeepDream 图像的成因需要区分语义内容与视觉伪影。Google 2015 年原文（Olah 共同署名）将动物形状关联到动物训练数据，并展示场景数据训练的网络；选层决定更偏边缘纹理还是复杂形状。重复优化把初始微弱匹配放大。按卷积权重共享和整层激活目标分析，同类局部特征可在多个位置同时被增强，而目标没有要求对象数量和身体结构合理。Olah 等人的 2017 年 Feature Visualization 进一步指出，步幅卷积和池化在反向传播中会产生高频、棋盘状梯度，影响纹理；这是架构对伪影的贡献，不能据此把狗脸语义归因到某个池化组件。图像外观还依赖多尺度、抖动、正则化和输入参数化。来源：[Google 2015](https://research.google/blog/inceptionism-going-deeper-into-neural-networks/)、[Distill 2017](https://distill.pub/2017/feature-visualization/)。
(2026-11-02)


实测补充（2026-11-02）：在完全相同的原版 Caffe GoogLeNet 特征提取器上，ImageNet 权重生成鸟头、眼睛与动物皮肤般斑纹；Places205 权重生成塔尖、拱门等建筑般结构。统一预处理后差异仍明显，Places 原生预处理也生成建筑结构。两者的分类头不参与计算。原 torchvision 网络的随机权重实验中，暗灰探针平均梯度为负、亮灰为正、中灰为零；正齐次网络的平方激活目标满足 J(tz)=t²J(z)，数值验证 Euler 比值约为 2。随机网络有响应，优化放大偏离中灰的幅度，无需物体语义；原图和多尺度原图基底保留了花的形状。结果与可复跑代码见 attribution-controls/README.md。

### Q: 能复现 Olah 说的狗头左右朝向吗？
使用 Lucid 原研究 InceptionV1 权重及 Distill DogCircuit.svg 指定的 mixed4a:308、mixed4a:222。狗头朝画面右侧时，两通道最大激活分别为 789.72、378.37；同一裁剪镜像后分别为 337.55、722.21。后续 mixed4b:409 对两向均较强（1051.29、1121.88）。从随机 Fourier 参数优化两个指定通道的 pre-ReLU 激活，也生成相反朝向的长嘴狗头，无狗照片或文字提示。自然图像来自一只狗，裁剪与灰度是同一图的变体；该实验复现单元方向偏好，未验证完整跨层因果通路或跨犬种泛化。代码、结果及 Notebook 位于 dog-direction/。

补充：按用户要求新增 [Iris 小模型版本](iris/README.md)。4→6→3 tanh 网络只有 51 个参数，120 条训练、30 条测试，测试正确 29/30；仅按训练集贡献选择 h3。在真实 Versicolor 测试样本上，把 h3 从 0.621 换成 Virginica 训练激活中位数 −0.774，保持输入和权重固定，Virginica 概率从 0.72% 变为 76.95%。输入梯度上升/下降分别偏好较短窄/较长宽花瓣。已验证前向计算、有限差分梯度、干预 logit 等式和权重不变，Notebook 5 格全部执行。这个实验展示机制及干预，不构成准确率改进。
(2026-10-30)

### Q: Olah 关于真正理解神经网络的研究有什么进展？对改善大模型有什么帮助？
截至 2026-10-30，机制可解释性已有可干预验证的局部机制解释，整体理解前沿模型仍是开放目标。2024 年 Scaling Monosemanticity 用稀疏自编码器从 Claude 3 Sonnet 提取可解释特征，并以增强金门大桥等特征影响输出。2025 年 Circuit Tracing 在 Claude 3.5 Haiku 中追踪多跳事实与押韵规划，并通过更换中间概念改变答案；方法只覆盖部分计算且有近似误差。OpenAI 2025 年稀疏模型研究通过剪除其余模型及删除关键边验证小型任务回路。2026 年自然语言自编码器将内部激活转成文字，以从文字重建激活作为训练信号；受控隐藏动机审计成功率 12–15%，对照低于 3%，解释仍会幻觉且成本高。2026 年 7 月 J-space/J-lens 研究进一步测量和干预可言说内部表征，10 月 J++ 更新仍为初步研究。总体是从寻找特征走向追踪与干预计算，并出现实用审计线索，尚未得到覆盖复杂模型全部行为的可靠解释。
来源：https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html 、https://www.anthropic.com/research/tracing-thoughts-language-model 、https://openai.com/index/understanding-neural-networks-through-sparse-circuits/ 、https://www.anthropic.com/research/natural-language-autoencoders 、https://transformer-circuits.pub/2026/workspace/index.html

#### 对改善模型的实际帮助

机制可解释性最直接的价值是定位故障、控制特定行为和改进安全评估。可以把它类比为调试程序时查看中间变量与调用路径：先解释错误如何产生，再决定修改数据、训练过程还是推理时的内部活动。

| 改进方向 | 已有案例或方法 | 对模型开发的作用 |
|---|---|---|
| 定位错误来源 | Anthropic 利用自然语言自编码器，追查早期 Opus 4.6 偶尔用其他语言回答英文问题的行为，找到了相关训练数据 | 为数据排查和针对性修复提供依据 |
| 控制特定行为 | 增强或抑制内部特征，即 activation steering；金门大桥实验显示特征干预可显著改变回答 | 提供提示词之外的调节入口，并测试某个特征是否实际影响输出 |
| 改进安全评估 | 分析模型是否识别出自己正被测试；在受控实验中寻找人为植入的隐藏动机 | 区分表面表现相似、内部原因不同的行为，发现需要进一步调查的问题 |
| 改进模型结构 | 在训练时限制连接，形成更稀疏、更容易分离的计算回路 | 探索让模型从一开始就更容易检查、调试和修改的设计 |

故障定位和审计案例见 [Natural Language Autoencoders（2026）](https://www.anthropic.com/research/natural-language-autoencoders)；特征干预见 [Scaling Monosemanticity（2024）](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)；结构设计见 [Understanding neural networks through sparse circuits（2025）](https://openai.com/index/understanding-neural-networks-through-sparse-circuits/)。

#### 公开的故障定位与缓解案例

Anthropic（2026-05-07）：早期 Opus 4.6 偶尔用俄语等语言回答英文问题。NLA 解释显示模型反复猜测用户真正的母语，研究人员据此发现监督微调数据只翻译了问题、没有翻译回答。论文公开了内部线索到训练数据根因的追查过程，没有披露清洗数据后的错误率或上线修复结果。来源：https://transformer-circuits.pub/2026/nla/index.html

OpenAI（2025-06-18）：在刻意用错误答案微调 GPT-4o 的实验中，狭窄领域的错误训练泛化为其他领域的不良行为。SAE 找到与不良角色有关的内部特征；反向干预减少不良行为。在另一项纠正微调实验中，120 条安全代码样本、30 步训练使该实验评估的失调率降至 0%。内部干预提供因果证据，重新训练提供行为恢复证据；论文未证明纠正微调必须依赖可解释性才能发现。来源：https://openai.com/index/emergent-misalignment/

OpenAI（2025-12-01）：针对泛化失调与不当认同用户信念两类行为，比较同一提示下好坏回答的 SAE 特征归因，再实际干预候选特征验证因果作用。多项特征可用于减少错误行为；两类问题还共同涉及一个挑衅、极端表达特征。这是受控实验的行为缓解，未声明已作为线上 ChatGPT 的修复部署。来源：https://alignment.openai.com/sae-latent-attribution/

#### 从本项目的狗头实验走向模型改进

本项目已观察到两个指定通道对狗头朝向的不同偏好。进一步改善识别模型，可以沿以下步骤开展实验：

1. 收集多只狗、多个犬种、不同背景和朝向的独立图片，确认具体的方向性识别错误。
2. 检查错误样本中的相关通道响应，提出可验证的机制假设。
3. 通过抑制、增强或替换内部激活，检验这些特征对错误是否有因果影响。
4. 根据结果选择数据增强、重新训练或内部干预等修复方式。
5. 在独立测试集上比较修复前后的错误率，并检查其他方向、类别和任务是否退步。

这些是后续实验方案。本项目当前完成的是特征响应与可视化复现，尚未执行模型修复。找到特征提供了调试入口；修复后目标错误减少、其他能力得到保留，才构成模型改进的证据。

#### 适用范围与局限

目前公开成果对定位问题、研究行为控制和辅助审计提供了具体证据；对全面提升模型能力、降低训练成本的收益仍需验证。内部解释可能遗漏计算或产生错误，特征干预也可能影响其他能力，因此需要独立样本、因果实验和回归评估共同确认。NLA 在受控隐藏动机实验中的成功率为 12–15%，说明审计能力有所改善，距离可靠覆盖复杂模型的所有风险仍有明显差距。稀疏模型研究展示了简单任务上的可解释回路，前沿规模模型的能力与成本收益仍是开放问题。

这一方向的实际目标，是把一部分模型问题转化为“提出机制解释 → 针对性修改 → 测量收益与副作用”的工程过程。
(2026-10-30)

### Q: DeepDream 原文是否讨论训练数据的影响？有多少引用？
Google 2015 年《Inceptionism: Going Deeper into Neural Networks》由 Mordvintsev、Olah、Tyka 共同署名，明确讨论动物训练数据，并展示 Places 场景网络的生成结果。本项目同架构对照复现了动物形态与建筑形态的差异。原文的随机噪声是输入随机，区别于本地补充的随机权重实验。

2026-11-02 查询到的引用统计快照：Rankless 打开页面显示 338（搜索摘要有不同缓存值），Metascience Observatory 的参考文献记录显示 551。两者口径和更新进度不同；不作为 Google Scholar 实时引用总数。来源：[Rankless](https://www.rankless.org/authors/mike-tyka)、[Metascience Observatory](https://explore.metascienceobservatory.org/papers/W2792641098)。
(2026-11-02)

<!-- 以下继续记录 -->
