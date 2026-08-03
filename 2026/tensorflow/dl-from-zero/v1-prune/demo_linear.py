"""dl-from-zero-1-prune python binding 演示: 线性回归 + 图裁剪
镜像 TF Python API: Graph / placeholder / variable / Session.run / feed_dict
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

    print(f"training graph: {g.num_nodes} nodes {g.node_names}")

    sess = lfpy.Session()
    feeds = {x: x_data, y: y_data}
    for epoch in range(epochs):
        sess.run(g, [loss], feeds, loss_node=loss)
        ga, gb = a.grad.data[0], b.grad.data[0]
        a.assign(a.output.data[0] - lr * ga)
        b.assign(b.output.data[0] - lr * gb)
        if epoch in (0, 99, 199):
            sess.run(g, [loss], feeds)
            print(f"epoch {epoch}: loss={loss.output.data[0]:.4f} "
                  f"a={a.output.data[0]:.4f} b={b.output.data[0]:.4f}")

    print("\n--- prune: train graph -> inference graph ---")
    print(f"before: {g.num_nodes} nodes {g.node_names}")
    lfpy.prune(g, [y_pred])
    print(f"after : {g.num_nodes} nodes {g.node_names}")

    print("\n--- inference (y was pruned, feed x only) ---")
    for tx in (-3.0, -1.0, 0.0, 1.0, 3.0, 5.0):
        sess.run(g, [y_pred], {x: [tx]})
        pred = y_pred.output.data[0]
        print(f"x={tx:>5}  y_pred={pred:8.4f}  y_true={true_a * tx + true_b:8.4f}")


if __name__ == "__main__":
    main()
