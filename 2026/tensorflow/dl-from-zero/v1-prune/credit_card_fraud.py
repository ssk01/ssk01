"""逻辑回归: 信用卡欺诈检测 (二分类, 高度不平衡)

数据: 真实 Kaggle creditcard.csv 不在本地, 用 sklearn make_classification 生成
同分布的合成数据 —— 30 特征, ~0.3% 欺诈率, 随机噪声, 高度不平衡。

模型:  p = sigmoid(X @ w + b)
损失:  加权二分类交叉熵  -(pos_weight * y * ln p + (1-y) * ln(1-p))
       pos_weight 按 (1-pos_frac)/pos_frac 设置, 处理类别不平衡。

评价:  欺诈场景看 recall/AUC 而不是 accuracy(全判负也有 99.7% 准确率)。
"""
import sys
import time

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
import lfpy


def main():
    print("generating synthetic credit-card-fraud-like data ...")
    X, y = make_classification(
        n_samples=50000, n_features=30, n_informative=10, n_redundant=0,
        n_repeated=0, n_classes=2, weights=[0.997, 0.003],
        class_sep=2.5, flip_y=0.002, random_state=42)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42,
                                          stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    N, F = Xtr.shape
    pos_frac = float(ytr.mean())
    pos_weight = (1.0 - pos_frac) / pos_frac
    print(f"train={N} test={Xte.shape[0]} features={F} "
          f"fraud_rate={pos_frac:.4f} pos_weight={pos_weight:.1f}")

    # ---- 建图: p = sigmoid(X @ w + b), 加权 BCE ----
    g = lfpy.Graph()
    Xn = g.placeholder("X", [N, F])
    yn = g.placeholder("y", [N])
    w = g.variable_vec("w", F, 0.0)
    # bias 初始化为先验 log-odds: 让初始 p≈fraud_rate, 打破 p=0.5 的梯度固定点
    b = g.variable("b", float(np.log(pos_frac / (1.0 - pos_frac))))
    pw = g.variable("pos_weight", pos_weight)
    one = g.variable("one", 1.0)
    mone = g.variable("minus_one", -1.0)

    logits = g.add(g.matmul(Xn, w), b)
    p = g.sigmoid(logits)
    pos = g.mul(g.mul(yn, pw), g.log(p))
    neg = g.mul(g.sub(one, yn), g.log(g.sub(one, p)))
    loss = g.mul(g.mean(g.add(pos, neg)), mone)
    print(f"train graph: {g.num_nodes} nodes {g.node_names}")

    # ---- 训练 (SGD, 手动更新变量, 对应 Variable.assign) ----
    sess = lfpy.Session()
    feeds = {Xn: lfpy.Tensor(Xtr.flatten().tolist(), [N, F]), yn: ytr.tolist()}
    lr, epochs = 0.5, 200
    t0 = time.time()
    for epoch in range(epochs):
        sess.run(g, [loss], feeds, loss_node=loss)
        lw, lb = w.grad.data, b.grad.data
        w.assign([w.output.data[i] - lr * lw[i] for i in range(F)])
        b.assign(b.output.data[0] - lr * lb[0])
        if epoch % 20 == 0 or epoch == epochs - 1:
            sess.run(g, [loss], feeds)
            print(f"epoch {epoch:3d}: loss={loss.output.data[0]:8.4f}  "
                  f"elapsed={time.time()-t0:.1f}s")

    # ---- 评估 ----
    Xte_t = lfpy.Tensor(Xte.flatten().tolist(), [Xte.shape[0], F])
    sess.run(g, [p], {Xn: Xte_t})
    proba = np.array(p.output.data)
    pred = (proba > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    auc = roc_auc_score(yte, proba)
    print(f"\nconfusion matrix (test):")
    print(f"[[TN={tn} FP={fp}]")
    print(f" [FN={fn} TP={tp}]]")
    print(f"fraud recall   = {recall:.4f}   (抓到的欺诈比例 —— 核心指标)")
    print(f"fraud precision= {precision:.4f}")
    print(f"AUC            = {auc:.4f}")
    print(f"注: 全判负的 baseline 准确率是 {1 - pos_frac:.4f}, 所以只看 accuracy 无意义")

    # ---- 图裁剪: 训练图 -> 推理图, 只保留 p 可达节点 ----
    print("\n--- prune: train graph -> inference graph ---")
    print(f"before: {g.num_nodes} nodes {g.node_names}")
    lfpy.prune(g, [p])
    print(f"after : {g.num_nodes} nodes {g.node_names}")
    sess.run(g, [p], {Xn: Xte_t})
    print(f"pruned inference proba[0:5] = {p.output.data[0:5]}")


if __name__ == "__main__":
    main()
