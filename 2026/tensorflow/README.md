# 双塔模型生产流程 Demo

用一个简单的双塔推荐模型，完整演示「样本构造 → 训练 → 导出 → 验证」的上线链路。

运行一条命令即可：

```bash
python demo_two_tower.py
```

## 模型结构 (Keras Functional API)

```
user_input(3) ──→ Dense(32) → user_emb(16) ──┐
                                              ├─ concat → Dense(16) → Dense(8) → Dense(1) → sigmoid
item_input(3) ──→ Dense(32) → item_emb(16) ──┘
```

导出方式: `model.export(path)` — Functional API 建模时显式声明了多输入 (`Input(shape=..., name="user_feat")`)，Keras 自动生成正确的 serving_default 签名, 无需 ExportRoot, 无需手动建图.

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
Epoch 1/5 | loss=0.7026 | ckpt=checkpoints/ckpt-0
Epoch 2/5 | loss=0.6902 | ckpt=checkpoints/ckpt-1
Epoch 3/5 | loss=0.6826 | ckpt=checkpoints/ckpt-2
Epoch 4/5 | loss=0.6746 | ckpt=checkpoints/ckpt-3
Epoch 5/5 | loss=0.6628 | ckpt=checkpoints/ckpt-4

训练完成.
checkpoint 目录: ['ckpt-4.data-00000-of-00001', 'checkpoint', ...]
注意: checkpoint 不自包含图结构, 不适合直接用于推理!

============================================================
阶段 3: 导出 saved_model (model.export)
============================================================
Saved artifact at saved_models/20260714
Endpoint 'serve':
  args_0: [TensorSpec(shape=(None,3), name='user_feat'),
           TensorSpec(shape=(None,3), name='item_feat')]
Output: TensorSpec(shape=(None,), dtype=float32)

============================================================
阶段 4: 加载 & 验证 saved_model
============================================================

--- 4.1 Signature ---
  serving_default: user_feat(batch×3) + item_feat(batch×3) → float32(batch×1)

--- 4.2 对拍验证: 训练图 vs 导出图 ---
训练图: [0.5006  0.4726  0.4889  0.4655  0.4637]
导出图: [0.5006  0.4726  0.4889  0.4655  0.4637]
最大差异: 0.0000000000  ✓ 对拍通过

--- 4.3 输出合理性检查 ---
score 范围: [0.4637, 0.5006] 均值: 0.4783  合理

============================================================
总结: checkpoint vs saved_model
============================================================
  checkpoint  → 训练态 (权重 + optimizer slot, 不自包含)
  saved_model → 推理态 (正向图 + 权重 + signature, 自包含)
```

## 参考

- [TFRecord格式详解.md](TFRecord格式详解.md) — 文件结构、CRC 校验、Example protobuf 定义、读写范式
- [checkpoint详解.md](checkpoint详解.md) — checkpoint 三类文件的内部结构、变量清单、对象图
- [saved_model详解.md](saved_model详解.md) — saved_model.pb 内部结构、SignatureDef、导出过程、与 checkpoint 对比
