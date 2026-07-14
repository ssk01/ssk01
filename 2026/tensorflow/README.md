# 双塔模型生产流程 Demo

用一个简单的双塔推荐模型，完整演示「样本构造 → 训练 → 导出 → 验证」的上线链路。

运行一条命令即可：

```bash
python demo_two_tower.py
```

## 模型结构

```
user_feat(3) ──→ Dense(32, relu) ──→ user_emb(16) ──┐
                                                      ├─ concat → Dense(16) → Dense(8) → Dense(1) → sigmoid → score
item_feat(3) ──→ Dense(32, relu) ──→ item_emb(16) ──┘
```

## 四阶段流程

### 阶段 1: 样本构造

```
特征平台 → TFRecord 文件
```

- 用 `tf.train.Example` 序列化每条样本，写入 `samples.tfrecord`
- 每条样本包含 `user_feat`、`item_feat`、`label`
- 关键约束：**训练样本的特征处理逻辑必须和在线 serving 一致**，否则出现 train/serving skew

### 阶段 2: 训练 & checkpoint

```
TFRecord → 训练循环 → checkpoints/
```

- 自定义训练循环 (GradientTape + Adam)，产出 checkpoint
- `tf.train.Checkpoint` 管理模型变量 + optimizer 状态 + global_step

**checkpoint 是什么**：训练过程中周期性保存的变量值快照 (`ckpt-N.data-*` + `.index`)。

```
checkpoint: 变量名 → 张量值 (含 Adam m/v、global_step)
用途:      续训、容错恢复
缺陷:      不自包含图结构，需重新跑构图代码才能 restore
结论:      不适合直接拿去推理
```

### 阶段 3: 导出 saved_model

```
checkpoint → 正向推理图 + 签名 → saved_models/<版本号>/
```

- 把训练好的 layer 拿出来，组装成只含前向算子的 export module
- 用 `tf.function(input_signature=[...])` 定义签名——输入/输出的张量名、dtype、shape
- `tf.saved_model.save` 写出 `saved_model.pb` + `variables/`

```
saved_models/20260714/
├── saved_model.pb          ← 计算图 + SignatureDef
└── variables/
    ├── variables.data-*     ← 权重数值
    └── variables.index
```

**saved_model 是什么**：自包含的推理产物，脱离训练代码即可加载运行。

```
saved_model: 正向图 + 权重 + 签名 (自包含)
用途:        推理交付、上线 serving、迁移到自研平台
优点:        独立运行，不依赖训练代码
```

**signature 是什么**：定义模型对外的「函数接口」——serving 端靠它知道喂什么、取什么。

```
输入: user_feat (float32, batch × 3)
      item_feat (float32, batch × 3)
输出: scores    (float32, batch × 1)
```

### 阶段 4: 正向推理验证

```
saved_model → 加载 → 前向推理 → 对拍打分
```

- 打印 signature 信息（类似 `saved_model_cli show`）
- 同一批输入，比对「训练图前向」和「导出图前向」打分
- diff 应为 0（或 < 1e-5），确认导出没丢算子

## checkpoint vs saved_model 总结

| | checkpoint | saved_model |
|---|---|---|
| 目的 | 训练存档、续训、容错 | 推理交付、上线 serving |
| 内容 | 变量值 (含 optimizer slot、global_step) | 正向图 + 权重 + signature |
| 图结构 | 不自包含，需代码重建 | 自包含，可独立加载 |
| 训练算子 | 保留 (梯度/优化器) | 剔除，只留正向 |
| 加载方 | 训练脚本 `restore` | TF Serving / 自研平台 |

**一句话：checkpoint → 训练态；saved_model → 推理态。**

## 后续流程

1. 拿到 saved_model，用 `saved_model_cli show --dir <path> --all` 核对输入接口
2. 对齐 signature 后做 user 特征压缩、XLA、低精度等优化
3. 每步优化都对拍验证（固定输入 + 基线打分），确保数值一致性
4. 上线前 warmup 预热，避免首请求编译超时

## 执行日志

```
TensorFlow version: 2.21.0
============================================================
双塔模型生产流程 Demo
样本构造 → 训练(checkpoint) → 导出(saved_model) → 验证
============================================================

============================================================
阶段 1: 样本构造 (写 TFRecord)
============================================================
写入 1000 条样本 → samples.tfrecord

============================================================
阶段 2: 训练 (产出 checkpoint)
============================================================
Epoch 1/5 | loss=0.6950 | step=32 | ckpt=checkpoints/ckpt-0
Epoch 2/5 | loss=0.6894 | step=64 | ckpt=checkpoints/ckpt-1
Epoch 3/5 | loss=0.6857 | step=96 | ckpt=checkpoints/ckpt-2
Epoch 4/5 | loss=0.6837 | step=128 | ckpt=checkpoints/ckpt-3
Epoch 5/5 | loss=0.6778 | step=160 | ckpt=checkpoints/ckpt-4

训练完成. 最优 loss=0.6778 (step=160)
checkpoint 目录内容: ['ckpt-4.data-00000-of-00001', 'checkpoint',
  'ckpt-3.data-00000-of-00001', 'ckpt-2.data-00000-of-00001',
  'ckpt-2.index', 'ckpt-4.index', 'ckpt-3.index']
注意: checkpoint 包含 data-* (权重) + index (索引), 以及 optimizer 状态
      checkpoint 不自包含图结构, 不适合直接用于推理!

============================================================
阶段 3: 导出 saved_model
============================================================
saved_model 已导出到: saved_models/20260714

saved_model 目录结构:
20260714/
  fingerprint.pb
  saved_model.pb
  variables/
    variables.data-00000-of-00001
    variables.index
  assets/

============================================================
阶段 4: 加载 & 验证 saved_model
============================================================

--- 4.1 Signature 信息 (模拟 saved_model_cli show) ---
  signature: serving_default
  输入 (用户可见):
    item_feat  dtype=float32  shape=batch × 3
    user_feat  dtype=float32  shape=batch × 3
  输出:
    Identity  dtype=float32  shape=batch × 1

也可用命令行查看: saved_model_cli show --dir saved_models/20260714 --all

--- 4.2 对拍验证: 训练图 vs 导出图 ---

训练图产出: [0.4722  0.4215  0.4410  0.4508  0.4747]
导出图产出: [0.4722  0.4215  0.4410  0.4508  0.4747]

最大差异: 0.0000000000
✓ 对拍通过: 训练图和导出图打分一致!

--- 4.3 输出合理性检查 ---
score 范围: [0.4215, 0.4747]
score 均值: 0.4520
输出在 [0,1] 区间, 合理.

============================================================
总结: checkpoint vs saved_model
============================================================

 checkpoint 目录: checkpoints
   ├── 内容: 模型权重 + optimizer 状态 (Adam m/v) + global_step
   ├── 用途: 续训 / 容错恢复 (只对训练进程有意义)
   └── 缺陷: 不自包含图结构, 需重新跑构图代码才能 restore

 saved_model 目录: saved_models/20260714
   ├── 内容: saved_model.pb (正向图 + signature) + variables/ (权重)
   ├── 用途: 推理交付 / 上线 serving / 迁移到自研平台
   └── 优点: 自包含, 脱离训练代码可独立加载运行

 一句话: checkpoint → 训练态; saved_model → 推理态
 导出过程: 从 checkpoint restore 变量 → 构建正向推理图
          → 绑定 signature → 写成 saved_model
```

## 参考

- [TFRecord格式详解.md](TFRecord格式详解.md) — 文件结构、CRC 校验、Example protobuf 定义、读写范式、与其他格式对比
- [checkpoint详解.md](checkpoint详解.md) — checkpoint 三类文件的内部结构、变量清单、index 映射关系
