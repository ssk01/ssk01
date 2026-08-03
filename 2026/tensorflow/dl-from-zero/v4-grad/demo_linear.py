"""dl-from-zero-4-grad python binding demo: 线性回归
梯度 = 图里的独立子图 (build_gradients), 训练 fetch sgd_step 节点即可
"""
import sys
import random

sys.path.insert(0, ".")
import lfpy


def main():
    random.seed(42)
    N, epochs, lr = 100, 200, 0.01
    true_a, true_b = 2.0, 3.0

    x_data = [random.uniform(-5.0, 5.0) for _ in range(N)]
    y_data = [true_a * x + true_b + random.gauss(0, 0.5) for x in x_data]

    g = lfpy.Graph()
    x = g.placeholder("x", [N])
    y = g.placeholder("y", [N])
    a = g.variable("a", 0.0)
    b = g.variable("b", 0.0)
    y_pred = g.add(g.mul(x, a), b)
    loss = g.mean(g.square(g.sub(y_pred, y)))

    # 独立的梯度子图: {a: grad_a_node, b: grad_b_node, ...}
    grads = lfpy.build_gradients(g, loss)
    train_a = g.sgd_step(a, grads[a], lr)
    train_b = g.sgd_step(b, grads[b], lr)
    print(f"graph: {g.num_nodes} nodes (forward + gradient subgraph + sgd)")

    sess = lfpy.Session()
    feeds = {x: x_data, y: y_data}
    for epoch in range(epochs):
        sess.run(g, [train_a, train_b], feeds)          # 训练: 正向 + 梯度子图 + 应用
        if epoch in (0, 99, 199):
            loss_val = sess.run(g, [loss], feeds)[0].data[0]
            print(f"epoch {epoch}: loss={loss_val:.4f} "
                  f"a={sess.var_value(a).data[0]:.4f} b={sess.var_value(b).data[0]:.4f}")

    print("\n--- prune: 梯度子图 + loss + sgd 全部被剪掉 ---")
    print(f"before: {g.num_nodes} nodes {g.node_names}")
    lfpy.prune(g, [y_pred])
    print(f"after : {g.num_nodes} nodes {g.node_names}")

    print("\n--- inference ---")
    for tx in (-3.0, -1.0, 0.0, 1.0, 3.0, 5.0):
        pred = sess.run(g, [y_pred], {x: [tx]})[0].data[0]
        print(f"x={tx:>5}  y_pred={pred:8.4f}  y_true={true_a * tx + true_b:8.4f}")


if __name__ == "__main__":
    main()
