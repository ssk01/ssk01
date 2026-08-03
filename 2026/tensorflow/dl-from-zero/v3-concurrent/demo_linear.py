"""dl-from-zero-3-concurrent python binding demo: 线性回归 + 图裁剪
TF-like: 图是静态的, 优化器是图上的节点(sgd_step), sess.run 返回 fetch 张量
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
    diff = g.sub(y_pred, y)
    loss = g.mean(g.square(diff))
    g.sgd_step(a, lr)   # 优化器是图上的节点 (对应 TF ApplyGradientDescent)
    g.sgd_step(b, lr)

    print(f"training graph: {g.num_nodes} nodes {g.node_names}")

    sess = lfpy.Session()
    feeds = {x: x_data, y: y_data}
    for epoch in range(epochs):
        loss_val = sess.run(g, [loss], feeds, loss_node=loss)[0].data[0]
        if epoch in (0, 99, 199):
            print(f"epoch {epoch}: loss={loss_val:.4f} "
                  f"a={sess.var_value(a).data[0]:.4f} b={sess.var_value(b).data[0]:.4f}")

    print("\n--- prune: train graph -> inference graph ---")
    print(f"before: {g.num_nodes} nodes {g.node_names}")
    lfpy.prune(g, [y_pred])
    print(f"after : {g.num_nodes} nodes {g.node_names}")

    print("\n--- inference (y was pruned, feed x only) ---")
    for tx in (-3.0, -1.0, 0.0, 1.0, 3.0, 5.0):
        pred = sess.run(g, [y_pred], {x: [tx]})[0].data[0]
        print(f"x={tx:>5}  y_pred={pred:8.4f}  y_true={true_a * tx + true_b:8.4f}")


if __name__ == "__main__":
    main()
