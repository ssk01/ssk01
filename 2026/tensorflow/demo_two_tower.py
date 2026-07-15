"""
双塔模型生产流程 Demo
======================
按照「样本构造 → 训练(checkpoint) → 导出(saved_model) → 正向推理验证」
四个阶段，用一个简单的双塔推荐模型走完整条链路。

模型结构 (Keras Functional API):
  user_feat(3) → Dense(32) → user_emb(16) ─┐
                                            ├─ concat → Dense(16) → Dense(8) → Dense(1) → sigmoid
  item_feat(3) → Dense(32) → item_emb(16) ─┘

导出方式: model.save() — Functional API 建模时显式声明了多输入,
Keras 自动生成正确的 serving_default 签名, 不需要手动构造图或 ExportRoot.

运行:
    python demo_two_tower.py

对 TensorFlow 版本要求: TF 2.x (已测试 2.21.0)
"""

import os
import shutil
import numpy as np
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")

# ============================================================
# 全局配置
# ============================================================
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
EXPORT_BASE_DIR = os.path.join(WORK_DIR, "saved_models")
TFRECORD_PATH = os.path.join(WORK_DIR, "samples.tfrecord")

USER_DIM = 16       # user embedding 维度
ITEM_DIM = 16       # item embedding 维度
N_SAMPLES = 1000    # 样本数
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
EXPORT_VERSION = 20260714  # export 版本号 (对应文档中的时间戳版本号)


def cleanup():
    """清理上次运行的产物, 确保每次都是一次干净的 Demo"""
    for d in [CHECKPOINT_DIR, EXPORT_BASE_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
    for f in [TFRECORD_PATH]:
        if os.path.exists(f):
            os.remove(f)


# ============================================================
# 阶段 1: 样本构造
# ============================================================
# 文档对应:
#   - "从特征平台按 point-in-time 取特征, 拼上 label, 落成样本文件 (TFRecord)"
#   - "样本里区分 user 特征、item 特征、context 特征"
#
# 这里用合成数据模拟: user 特征 + item 特征 + label
# ============================================================

def build_samples_tfrecord():
    """
    阶段 1: 构造训练样本, 写入 TFRecord.
    
    每个样本包含:
      - user_feat: [age, gender, city_id]  (user 侧连续特征)
      - item_feat: [category, price, brand] (item 侧连续特征)
      - label: 0/1  (是否点击)
    
    注意: 训练样本的特征处理逻辑必须和在线 serving 一致 (文档强调)。
    这里为了简单, 直接用 float, 实际项目可能涉及 hash、分桶、归一化等。
    """
    print("\n" + "=" * 60)
    print("阶段 1: 样本构造 (写 TFRecord)")
    print("=" * 60)

    np.random.seed(42)
    
    with tf.io.TFRecordWriter(TFRECORD_PATH) as writer:
        for i in range(N_SAMPLES):
            user_feat = np.random.randn(3).astype(np.float32)
            item_feat = np.random.randn(3).astype(np.float32)
            
            # 真实推荐系统里 label 来自用户行为日志
            # 这里造一个微弱的 pattern: user[0] * item[0] 越大 → 正样本概率越高
            score = user_feat[0] * item_feat[0]
            prob = 1.0 / (1.0 + np.exp(-score))  # sigmoid
            label = 1 if np.random.random() < prob else 0
            
            example = tf.train.Example(features=tf.train.Features(feature={
                "user_feat": tf.train.Feature(
                    float_list=tf.train.FloatList(value=user_feat)),
                "item_feat": tf.train.Feature(
                    float_list=tf.train.FloatList(value=item_feat)),
                "label": tf.train.Feature(
                    float_list=tf.train.FloatList(value=[float(label)])),
            }))
            writer.write(example.SerializeToString())
    
    print(f"写入 {N_SAMPLES} 条样本 → {TFRECORD_PATH}")
    return TFRECORD_PATH


def parse_tfrecord(serialized):
    """TFRecord 解析: 还原成 (user_feat, item_feat, label)"""
    features = {
        "user_feat": tf.io.FixedLenFeature([3], tf.float32),
        "item_feat": tf.io.FixedLenFeature([3], tf.float32),
        "label":     tf.io.FixedLenFeature([1], tf.float32),
    }
    parsed = tf.io.parse_single_example(serialized, features)
    return parsed["user_feat"], parsed["item_feat"], parsed["label"][0]


# ============================================================
# 阶段 2: 双塔模型定义 & 训练 (产出 checkpoint)
# ============================================================
# 文档对应:
#   - "训练过程中周期性保存的 Variable 值快照, 本质是变量名 → 张量值"
#   - checkpoint 用于续训、容错恢复, 不适合直接拿去推理
# ============================================================

def build_model():
    """
    阶段 2 (前半): 用 Keras Functional API 定义双塔模型。

    为什么用 Functional API 而非 Subclassing:
      - Functional API 建模时显式声明了输入: Input(shape=(3,), name="user_feat")
      - model.save() 能自动生成正确的 serving_default 签名
      - 不需要 ExportRoot、不需要手写 tf.matmul、不需要手动建图
      - 这是生产环境的标准做法
    """
    user_input = tf.keras.Input(shape=(3,), dtype=tf.float32, name="user_feat")
    item_input = tf.keras.Input(shape=(3,), dtype=tf.float32, name="item_feat")

    # User Tower
    u = tf.keras.layers.Dense(32, activation="relu", name="user_dense1")(user_input)
    u = tf.keras.layers.Dense(USER_DIM, name="user_embedding")(u)

    # Item Tower
    i = tf.keras.layers.Dense(32, activation="relu", name="item_dense1")(item_input)
    i = tf.keras.layers.Dense(ITEM_DIM, name="item_embedding")(i)

    # 顶部 MLP: concat → 16 → 8 → 1
    x = tf.keras.layers.Concatenate(name="concat")([u, i])
    x = tf.keras.layers.Dense(16, activation="relu", name="top_dense1")(x)
    x = tf.keras.layers.Dense(8, activation="relu", name="top_dense2")(x)
    logit = tf.keras.layers.Dense(1, name="top_out")(x)
    scores = tf.keras.layers.Lambda(lambda t: tf.sigmoid(tf.squeeze(t, axis=1)), name="scores")(logit)

    model = tf.keras.Model(inputs=[user_input, item_input], outputs=scores, name="two_tower")
    return model


def train_model():
    """
    阶段 2: 训练双塔模型, 产出 checkpoint.

    Functional API 模型: model([user_feat, item_feat]) 或
    model({"user_feat": ..., "item_feat": ...})
    """
    print("\n" + "=" * 60)
    print("阶段 2: 训练 (产出 checkpoint)")
    print("=" * 60)

    raw_dataset = tf.data.TFRecordDataset([TFRECORD_PATH])
    dataset = (raw_dataset
               .map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
               .shuffle(512)
               .batch(BATCH_SIZE)
               .prefetch(tf.data.AUTOTUNE))

    model = build_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

    # 先跑一个 batch 以初始化变量
    for user_feat, item_feat, _ in dataset.take(1):
        _ = model([user_feat, item_feat])

    ckpt_manager = tf.train.CheckpointManager(
        checkpoint, CHECKPOINT_DIR, max_to_keep=3
    )

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n_batches = 0

        for user_feat, item_feat, label in dataset:
            with tf.GradientTape() as tape:
                pred = model([user_feat, item_feat], training=True)
                loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(label, pred))

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss += loss.numpy()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        ckpt_path = ckpt_manager.save(checkpoint_number=epoch)
        print(f"Epoch {epoch+1}/{EPOCHS} | loss={avg_loss:.4f} | ckpt={ckpt_path}")

    print(f"\n训练完成.")
    print(f"checkpoint 目录内容: {os.listdir(CHECKPOINT_DIR)}")
    print(f"注意: checkpoint 包含 data-* (权重) + index (索引), 以及 optimizer 状态")
    print(f"      checkpoint 不自包含图结构, 不适合直接用于推理!")

    return model


# ============================================================
# 阶段 3: 导出 saved_model
# ============================================================
# 文档对应:
#   - "saved_model 是自包含的推理产物: 计算图 + 权重 + 签名"
#   - "导出把训练专用子图 (优化器、梯度、dropout) 剔除, 只保留正向推理图"
#   - "export_path 带版本号, TF Serving 按数字版本号自动发现"
#   - "signature 定义模型的函数接口: 输入张量名/dtype/shape, 输出张量"
# ============================================================

def export_saved_model(model):
    """
    阶段 3: 导出 saved_model.

    Functional API 模型直接 model.save() 即可:
      - 建模时 Input 层已经显式声明了输入名和 shape
      - Keras 自动生成正确的 serving_default 签名
      - 不需要 ExportRoot、不需要手写变量、不需要手写图
    """
    print("\n" + "=" * 60)
    print("阶段 3: 导出 saved_model (model.save)")
    print("=" * 60)

    export_path = os.path.join(EXPORT_BASE_DIR, str(EXPORT_VERSION))
    model.export(export_path)

    print(f"saved_model 已导出到: {export_path}")
    print(f"\nsaved_model 目录结构:")
    for root, dirs, files in os.walk(export_path):
        level = root.replace(export_path, "").count(os.sep)
        indent = "  " * level
        dir_name = os.path.basename(root) or "."
        print(f"{indent}{dir_name}/")
        for f in files:
            print(f"{indent}  {f}")

    return export_path


# ============================================================
# 阶段 4: 正向推理验证
# ============================================================
# 文档对应:
#   - "用 run_model 在原 TF 上跑固定输入集, 产出基线打分"
#   - 对拍验证: 训练图前向 vs 导出图前向, 确认没丢算子
#   - saved_model_cli show 核对 signature
# ============================================================

def verify_saved_model(model, export_path):
    """
    阶段 4: 加载 saved_model 并做验证.
    
    验证步骤 (对应文档的验收标准):
      1. 打印 signature 信息 (类似 saved_model_cli show)
      2. 用同一批输入, 比对「训练图前向」vs「导出图前向」的打分
      3. 检查输出量级是否合理
    """
    print("\n" + "=" * 60)
    print("阶段 4: 加载 & 验证 saved_model")
    print("=" * 60)
    
    # 4.1 打印 signature 信息 (类似 saved_model_cli show)
    print("\n--- 4.1 Signature 信息 (模拟 saved_model_cli show) ---")
    loaded = tf.saved_model.load(export_path)
    sig = loaded.signatures["serving_default"]
    print(f"  signature: serving_default")
    print(f"  输入 (用户可见):")
    for t in sig.inputs:
        if t.dtype == tf.resource:
            continue  # 跳过 TF 内部变量 resource tensor
        shape_str = "batch × 3" if t.shape.rank == 2 else str(t.shape)
        name = t.name.split(":")[0] if ":" in t.name else t.name
        print(f"    {name}  dtype={t.dtype.name}  shape={shape_str}")
    print(f"  输出:")
    for t in sig.outputs:
        name = t.name.split(":")[0] if ":" in t.name else t.name
        print(f"    {name}  dtype={t.dtype.name}  shape=batch × 1")

    cli_cmd = f"saved_model_cli show --dir {export_path} --all"
    print(f"\n也可用命令行查看: {cli_cmd}")
    print("  (saved_model_cli 会打印更完整的 MetaGraphDef, 包括 tag、method_name 等)")
    
    # 4.2 准备固定输入 (batch_size=5, 后续对拍都用这批数据)
    print("\n--- 4.2 对拍验证: 训练图 vs 导出图 ---")
    np.random.seed(42)
    test_user = np.random.randn(5, 3).astype(np.float32)
    test_item = np.random.randn(5, 3).astype(np.float32)
    
    # 训练图前向 (Functional API: 输入是 list)
    train_pred = model([test_user, test_item], training=False).numpy()
    
    # 导出图前向 (model.export() 输出 key 为 output_0)
    serving_fn = loaded.signatures["serving_default"]
    serving_output = serving_fn(
        user_feat=tf.constant(test_user),
        item_feat=tf.constant(test_item),
    )
    out_key = list(serving_output.keys())[0]
    serving_pred = serving_output[out_key].numpy()
    
    # 比对打分
    print(f"\n训练图产出: {train_pred}")
    print(f"导出图产出: {serving_pred}")
    
    diff = np.abs(train_pred - serving_pred).max()
    print(f"\n最大差异: {diff:.10f}")
    
    if diff < 1e-5:
        print("✓ 对拍通过: 训练图和导出图打分一致!")
    else:
        print("✗ 对拍失败: 导出图与训练图打分不一致, 可能丢了算子或 BN/dropout 没固化!")
    
    # 4.3 肉眼确认输出合理
    print(f"\n--- 4.3 输出合理性检查 ---")
    print(f"score 范围: [{serving_pred.min():.4f}, {serving_pred.max():.4f}]")
    print(f"score 均值: {serving_pred.mean():.4f}")
    if serving_pred.min() >= 0 and serving_pred.max() <= 1:
        print("输出在 [0,1] 区间, 合理.")
    else:
        print("输出超出 [0,1], 可能有异常.")
    
    return serving_pred


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("双塔模型生产流程 Demo")
    print("样本构造 → 训练(checkpoint) → 导出(saved_model) → 验证")
    print("=" * 60)
    
    cleanup()
    
    # 阶段 1: 样本构造
    build_samples_tfrecord()
    
    # 阶段 2: 训练 (产出 checkpoint)
    model = train_model()
    
    # 阶段 3: 导出 saved_model
    export_path = export_saved_model(model)
    
    # 阶段 4: 正向推理验证
    verify_saved_model(model, export_path)
    
    # ============================================================
    # 总结: checkpoint vs saved_model
    # ============================================================
    print("\n" + "=" * 60)
    print("总结: checkpoint vs saved_model")
    print("=" * 60)
    print(f"""
 checkpoint 目录: {CHECKPOINT_DIR}
   ├── 内容: 模型权重 + optimizer 状态 (Adam m/v) + global_step
   ├── 用途: 续训 / 容错恢复 (只对训练进程有意义)
   └── 缺陷: 不自包含图结构, 需重新跑构图代码才能 restore

 saved_model 目录: {export_path}
   ├── 内容: saved_model.pb (正向图 + signature) + variables/ (权重)
   ├── 用途: 推理交付 / 上线 serving / 迁移到自研平台
   └── 优点: 自包含, 脱离训练代码可独立加载运行

 一句话: checkpoint → 训练态; saved_model → 推理态
 导出过程: 从 checkpoint restore 变量 → 构建正向推理图 → 绑定 signature → 写成 saved_model

 后续流程 (参考文档):
   1. saved_model 是迁移的输入产物, 不碰 checkpoint
   2. 用 saved_model_cli show 核对输入接口
   3. 对齐 signature 后做 user 特征压缩、XLA、低精度等优化
   4. 每步优化都对拍验证, 确保打分一致性
""")


if __name__ == "__main__":
    main()
