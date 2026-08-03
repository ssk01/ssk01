"""并发推理: 同一张图被多个线程同时跑

图是纯静态的, 每轮的值都在各自 run 的 RunState 里 -> 互不干扰。
(CPython 有 GIL, 这里展示的是 API 语义上的正确性; 真实并行见 C++ main.cpp)
"""
import sys
import random
import threading

sys.path.insert(0, ".")
import lfpy


def main():
    random.seed(42)
    N, epochs, lr = 100, 300, 0.01
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
    g.sgd_step(a, lr)
    g.sgd_step(b, lr)

    sess = lfpy.Session()
    feeds = {x: x_data, y: y_data}
    for _ in range(epochs):
        sess.run(g, [loss], feeds, loss_node=loss)

    # 训练图 -> 推理图, 只保留 y_pred
    lfpy.prune(g, [y_pred])

    # 4 个线程同时跑同一张图, 各喂不同的 x
    errors = [None] * 4
    def worker(t, tx):
        err = 0.0
        for _ in range(500):
            pred = sess.run(g, [y_pred], {x: [tx]})[0].data[0]
            err += abs(pred - (true_a * tx + true_b))
        errors[t] = err / 500

    threads = [threading.Thread(target=worker, args=(t, -4.0 + t)) for t in range(4)]
    for th in threads: th.start()
    for th in threads: th.join()

    print("concurrent inference on the same graph (4 threads x 500 runs):")
    for t in range(4):
        print(f"  thread {t}  x={-4.0 + t}  mean_abs_err={errors[t]:.4f}")


if __name__ == "__main__":
    main()
