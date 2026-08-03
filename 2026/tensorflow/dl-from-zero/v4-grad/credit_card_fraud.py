"""逻辑回归: 信用卡欺诈检测 (二分类, 高度不平衡) — dl-from-zero-4-grad 版

梯度 = 独立子图 (build_gradients), 训练 fetch sgd_step。
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

    g = lfpy.Graph()
    Xn = g.placeholder("X", [N, F])
    yn = g.placeholder("y", [N])
    w = g.variable_vec("w", F, 0.0)
    # bias 初始化为先验 log-odds, 打破 p=0.5 的梯度固定点
    b = g.variable("b", float(np.log(pos_frac / (1.0 - pos_frac))))
    pw = g.variable("pos_weight", pos_weight)
    one = g.variable("one", 1.0)
    mone = g.variable("minus_one", -1.0)

    logits = g.add(g.matmul(Xn, w), b)
    p = g.sigmoid(logits)
    pos = g.mul(g.mul(yn, pw), g.log(p))
    neg = g.mul(g.sub(one, yn), g.log(g.sub(one, p)))
    loss = g.mul(g.mean(g.add(pos, neg)), mone)

    grads = lfpy.build_gradients(g, loss)
    train_w = g.sgd_step(w, grads[w], 0.5)   # 只更新 w 和 b
    train_b = g.sgd_step(b, grads[b], 0.5)

    sess = lfpy.Session()
    feeds = {Xn: lfpy.Tensor(Xtr.flatten().tolist(), [N, F]), yn: ytr.tolist()}
    epochs = 200
    t0 = time.time()
    for epoch in range(epochs):
        sess.run(g, [train_w, train_b], feeds)
        if epoch % 40 == 0 or epoch == epochs - 1:
            loss_val = sess.run(g, [loss], feeds)[0].data[0]
            print(f"epoch {epoch:3d}: loss={loss_val:8.4f}  elapsed={time.time()-t0:.1f}s")

    Xte_t = lfpy.Tensor(Xte.flatten().tolist(), [Xte.shape[0], F])
    proba = np.array(sess.run(g, [p], {Xn: Xte_t})[0].data)
    pred = (proba > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    auc = roc_auc_score(yte, proba)
    print(f"\nconfusion matrix: [[TN={tn} FP={fp}] [FN={fn} TP={tp}]]")
    print(f"fraud recall={recall:.4f} precision={precision:.4f} AUC={auc:.4f}")

    print("\n--- prune: 梯度子图 + loss + sgd 被剪掉 ---")
    print(f"before: {g.num_nodes} nodes")
    lfpy.prune(g, [p])
    print(f"after : {g.num_nodes} nodes {g.node_names}")


if __name__ == "__main__":
    main()
