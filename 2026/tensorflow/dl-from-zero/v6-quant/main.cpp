#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>
#include "framework/tensor.h"
#include "graph/gradients.h"
#include "graph/graph.h"
#include "graph/prune.h"
#include "graph/quantize.h"
#include "session.h"

using namespace lf;

static void print_count(const char* label, const Graph& g) {
    std::cout << "  " << label << ": " << g.nodes().size() << " nodes" << std::endl;
}

// 两张图逐值一致?
static bool allclose(const Tensor& a, const Tensor& b, float eps = 1e-5f) {
    if (a.size() != b.size()) return false;
    for (int i = 0; i < a.size(); i++)
        if (std::fabs(a.data[i] - b.data[i]) > eps) return false;
    return true;
}

int main() {
    // ============================================================
    // demo 1: CSE —— 同一子表达式被多个消费者各算一遍
    // y = sigmoid(x) + sigmoid(x): 两个 sigmoid 是各自构建的节点
    // ============================================================
    std::cout << "== demo 1: CSE 合并重复子表达式 ==" << std::endl;
    {
        std::vector<float> xd = {-3, -2, -1, 0, 1, 2, 3, 4};
        Tensor feed(xd, {8});

        Graph g1;
        auto x1 = g1.placeholder("x", {8});
        auto y1 = g1.add(g1.sigmoid(x1), g1.sigmoid(x1));  // 两个 sigmoid 节点
        print_count("before 1st run", g1);                 // 4
        Session s1;
        auto r1 = s1.run(g1, {y1}, {{x1, feed}});          // 第一次 run: 自动 CSE
        print_count("after  1st run (CSE merged sigmoid)", g1);  // 3
        auto r1b = s1.run(g1, {y1}, {{x1, feed}});
        bool ok = true;
        for (int i = 0; i < feed.size(); i++) {
            float expect = 2.0f / (1.0f + std::exp(-feed.data[i]));
            if (std::fabs(r1b[0].data[i] - expect) > 1e-5f) ok = false;
        }
        std::cout << "  values == direct 2*sigmoid(x): " << (ok ? "yes" : "NO")
                  << std::endl;
    }

    // ============================================================
    // demo 2: 常量折叠 —— 编译期就能算出的表达式, 不用每次 run 重算
    // 2a: y = x + 2*3          → mul 节点原地折叠成 const(6), 孤儿常量 2/3 被清
    // 2b: y = x + mean(square(2))  → 整条常量链折叠; 折叠出的两个 const(4)
    //     被第二轮 CSE 合并 (pass 联动)
    // ============================================================
    std::cout << "\n== demo 2: 常量折叠 ==" << std::endl;
    {
        std::vector<float> xd = {1, 2, 3, 4};
        Tensor feed(xd, {4});

        Graph g2;
        auto x2 = g2.placeholder("x", {4});
        auto m2 = g2.mul(g2.constant("c2", Tensor(2.0f)), g2.constant("c3", Tensor(3.0f)));
        auto y2 = g2.add(x2, m2);
        print_count("2a before 1st run", g2);              // 5
        Session s2;
        auto r2 = s2.run(g2, {y2}, {{x2, feed}});
        print_count("2a after  1st run (2*3 folded)", g2); // 3: x, const6, add
        Tensor expect2({7, 8, 9, 10}, {4});  // feed={1,2,3,4}, y = x + 6
        std::cout << "  values == x+6: " << (allclose(r2[0], expect2) ? "yes" : "NO")
                  << std::endl;

        Graph g2b;
        auto xb = g2b.placeholder("x", {4});
        auto kb = g2b.constant("k", Tensor(2.0f));
        auto yb = g2b.add(xb, g2b.mean(g2b.square(kb)));
        print_count("2b before 1st run", g2b);             // 5
        Session s2b;
        auto r2b = s2b.run(g2b, {yb}, {{xb, feed}});
        print_count("2b after  1st run (chain folded + CSE dedup)", g2b);  // 3
        Tensor expect2b({5, 6, 7, 8}, {4});  // feed={1,2,3,4}, y = x + 4
        std::cout << "  values == x+4: " << (allclose(r2b[0], expect2b) ? "yes" : "NO")
                  << std::endl;
    }

    // ============================================================
    // demo 4: 负例 —— CSE 不合并有状态节点
    // 两个完全相同的 sgd_step(v, g) (type/inputs/lr 全等): 一旦被 CSE 合并,
    // 梯度只应用一次 (v=0.8); 正确行为是两个节点各自应用一次 (v=0.6)。
    // 对应 TF Equivalent() 的 is_stateful 检查 (Variable/ApplyGradientDescent
    // 永不参与合并)。
    // ============================================================
    std::cout << "\n== demo 4: 负例 —— 有状态节点不合并 ==" << std::endl;
    {
        Graph g4;
        auto v4 = g4.variable("v", 1.0f);
        auto g4in = g4.placeholder("g", {});
        auto u1 = g4.sgd_step(v4, g4in, 0.1f);
        g4.sgd_step(v4, g4in, 0.1f);  // 与 u1 完全相同的节点, 只留在图里当"诱饵"
        print_count("graph before 1st run", g4);  // 4
        Session s4;
        s4.run(g4, {u1}, {{g4in, Tensor(2.0f)}});  // 第一次 run: 触发优化
        print_count("after 1st run (sgd kept as 2 nodes)", g4);  // 仍为 4
        float v = s4.var_value(v4).data[0];
        std::cout << "  v = " << v << "  (expect 0.6: 两个 sgd 各应用一次, 未合并)"
                  << std::endl;
    }

    // ============================================================
    // demo 3: 三 pass 串联 —— 训练图 → 推理图 (呼应 v1 的 54→6)
    //   训练图 = 前向 + 重复的监控指标 + 梯度子图 + sgd
    //   loss 和监控指标各自构建了相同的 mean(square(diff)) (真实代码:
    //   loss 函数和 eval 函数分开写, 同一指标算了两次) → CSE 合并
    // ============================================================
    std::cout << "\n== demo 3: prune + CSE + 常量折叠 串联 ==" << std::endl;
    const int N = 100;
    const float true_a = 2.0f, true_b = 3.0f;
    const int epochs = 200;
    const float lr = 0.01f;

    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 0.5f);
    std::uniform_real_distribution<float> x_dist(-5.0f, 5.0f);
    std::vector<float> x_data(N), y_data(N);
    for (int i = 0; i < N; i++) {
        x_data[i] = x_dist(rng);
        y_data[i] = true_a * x_data[i] + true_b + noise(rng);
    }

    Graph g;
    auto x = g.placeholder("x", {N});
    auto y = g.placeholder("y", {N});
    auto a = g.variable("a", 0.0f);
    auto b = g.variable("b", 0.0f);
    auto y_pred = g.add(g.mul(x, a), b);
    auto diff = g.sub(y_pred, y);
    auto loss = g.mean(g.square(diff));
    auto monitor = g.mean(g.square(g.sub(y_pred, y)));  // 重复子表达式, 独立构建
    (void)monitor;  // 构建即达目的: 首次 run 的 CSE 会把它的三个节点合并进 loss 链
    auto grads = build_gradients(g, loss);
    auto train_a = g.sgd_step(a, grads[a], lr);
    auto train_b = g.sgd_step(b, grads[b], lr);
    print_count("full graph (forward + dup monitor + grad + sgd)", g);

    Session sess;
    std::unordered_map<const Node*, Tensor> feeds = {
        {x, Tensor(x_data, {N})}, {y, Tensor(y_data, {N})}};
    // 第一次 run: 自动跑 CSE → 重复子表达式合并 (const fold 在此图无可折叠节点,
    // 是 no-op —— 折叠的演示见 demo 2)
    sess.run(g, {train_a, train_b}, feeds);
    print_count("after 1st run (auto optimize: dup monitor merged)", g);

    for (int epoch = 0; epoch < epochs; epoch++) {
        sess.run(g, {train_a, train_b}, feeds);
        if (epoch % 40 == 0 || epoch == epochs - 1) {
            auto lo = sess.run(g, {loss}, feeds);
            std::cout << "  epoch " << epoch << ": loss=" << lo[0].data[0] << std::endl;
        }
    }
    std::cout << "  trained: a=" << sess.var_value(a).data[0]
              << "  b=" << sess.var_value(b).data[0] << std::endl;

    // 推理切换: prune 掉梯度子图/loss/监控 (对应 v1 的训练图→推理图)
    prune(g, {y_pred});
    print_count("after prune{y_pred} (inference graph)", g);  // 5

    // 并发推理: 同一张图被多个线程同时跑 (v3/v4 保留节目)
    const int n_threads = 8, iters = 5000;
    std::vector<std::thread> threads;
    std::vector<double> mean_err(n_threads, 0.0);
    for (int t = 0; t < n_threads; t++) {
        threads.emplace_back([&, t]() {
            float tx = -4.0f + t;
            std::unordered_map<const Node*, Tensor> feed = {{x, Tensor(tx)}};
            double err = 0.0;
            for (int i = 0; i < iters; i++) {
                auto r = sess.run(g, {y_pred}, feed);
                err += std::fabs(r[0].data[0] - (true_a * tx + true_b));
            }
            mean_err[t] = err / iters;
        });
    }
    for (auto& th : threads) th.join();

    std::cout << "\n  concurrent inference: " << n_threads << " threads x " << iters
              << " runs on the same graph:" << std::endl;
    for (int t = 0; t < n_threads; t++) {
        float tx = -4.0f + t;
        std::cout << "    thread " << t << "  x=" << tx
                  << "  mean_abs_err=" << mean_err[t] << std::endl;
    }

    // ============================================================
    // demo 5: 量化推理 (int8) vs fp32
    //   对应 TF 首个量化 commit ca4e053aa52 (2016-04-22) + 论文 v2 §5。
    //   CTR 预估的最小形态: 多特征 logistic 回归
    //   训练 → prune 成推理图 → 校准激活范围 → 量化图变换
    //   (matmul → Quantize / QuantizedMatMul / Dequantize) → int8 vs fp32
    // ============================================================
    std::cout << "\n== demo 5: 量化推理 (int8) vs fp32 ==" << std::endl;
    {
        const int F = 16, N_train = 4000, N_test = 2000, epochs = 40;
        std::mt19937 rng(7);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        std::uniform_real_distribution<float> wdist(-1.0f, 1.0f);
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);
        std::vector<float> w_star(F);
        for (int f = 0; f < F; f++) w_star[f] = wdist(rng);
        const float b_star = 0.3f;

        // 合成数据: y ~ Bernoulli(sigmoid(x·w* + b*))
        auto gen = [&](int N, std::vector<float>& X, std::vector<float>& Y) {
            X.resize(N * F);
            Y.resize(N);
            for (int i = 0; i < N; i++) {
                float logit = b_star;
                for (int f = 0; f < F; f++) {
                    X[i * F + f] = gauss(rng);
                    logit += X[i * F + f] * w_star[f];
                }
                Y[i] = (prob(rng) < 1.0f / (1.0f + std::exp(-logit))) ? 1.0f : 0.0f;
            }
        };
        std::vector<float> x_tr, y_tr, x_te, y_te;
        gen(N_train, x_tr, y_tr);
        gen(N_test, x_te, y_te);

        Graph g5;
        auto x5 = g5.placeholder("x", {N_train, F});
        auto y5 = g5.placeholder("y", {N_train});
        auto w5 = g5.variable_vec("w", F, 0.0f);
        auto b5 = g5.variable("b", 0.0f);
        auto logit5 = g5.add(g5.matmul(x5, w5), b5);
        auto pred5 = g5.sigmoid(logit5);
        // BCE: loss = -mean(y*log(p) + (1-y)*log(1-p))
        auto one = g5.constant("one", Tensor(1.0f));
        auto p_loss = g5.mul(y5, g5.log(pred5));
        auto n_loss = g5.mul(g5.sub(one, y5), g5.log(g5.sub(one, pred5)));
        auto loss5 = g5.mul(g5.mean(g5.add(p_loss, n_loss)),
                            g5.constant("minus_one", Tensor(-1.0f)));
        auto grads5 = build_gradients(g5, loss5);
        auto tr_w5 = g5.sgd_step(w5, grads5[w5], 0.3f);
        auto tr_b5 = g5.sgd_step(b5, grads5[b5], 0.3f);
        std::unordered_map<const Node*, Tensor> feed5 = {
            {x5, Tensor(x_tr, {N_train, F})}, {y5, Tensor(y_tr, {N_train})}};
        Session s5;
        for (int e = 0; e < epochs; e++) s5.run(g5, {tr_w5, tr_b5}, feed5);
        std::cout << "  训练完成: loss=" << s5.run(g5, {loss5}, feed5)[0].data[0]
                  << std::endl;

        // fp32 基线: 先在量化前跑测试集留底 (量化会改写图)
        prune(g5, {pred5});  // 推理图: x, w, b, matmul, add, sigmoid = 6 节点
        std::cout << "  推理图: " << g5.nodes().size() << " nodes" << std::endl;
        std::unordered_map<const Node*, Tensor> feed_te = {
            {x5, Tensor(x_te, {N_test, F})}};
        auto logits_fp32 = s5.run(g5, {logit5}, feed_te)[0];

        // 校准: 激活 (x) 的 min/max 从校准数据统计 —— 对应 TF 早期"用户提供
        // min/max"的做法 (自动校准是后面 quantize_graph.py eightbit 模式的事)
        auto [xmin, xmax] = tensor_minmax(Tensor(x_tr, {N_train, F}));
        nudge_range(xmin, xmax);
        std::unordered_map<const Node*, Tensor> vars5;
        for (const auto& un : g5.nodes())
            if (un->type == VARIABLE) vars5[un.get()] = s5.var_value(un.get());
        quantize_inference(g5, vars5, {{x5, {xmin, xmax}}});
        std::cout << "  量化图: " << g5.nodes().size() << " nodes" << std::endl;
        auto logits_i8 = s5.run(g5, {logit5}, feed_te)[0];

        // 量化细节: 范围 → scale → zero-point (对应 TF 的数值)
        auto [wmin, wmax] = tensor_minmax(vars5.at(w5));
        nudge_range(wmin, wmax);
        std::cout << "  校准: x ∈ [" << xmin << ", " << xmax << "]  scale="
                  << 255.0f / (xmax - xmin) << "  zero_point="
                  << quantized_offset(xmin, xmax) << std::endl;
        std::cout << "  权重: w ∈ [" << wmin << ", " << wmax << "]  scale="
                  << 255.0f / (wmax - wmin) << "  zero_point="
                  << quantized_offset(wmin, wmax) << std::endl;

        // 输入量化往返误差 (量化到 int8 再反量化): 分辨率 = 1 个 level
        double rt_err = 0;
        for (int i = 0; i < N_train * F; i++) {
            float q = quantize_one(x_tr[i], xmin, xmax);
            rt_err += std::fabs(dequantize_one_i8(q, xmin, xmax) - x_tr[i]);
        }
        std::cout << "  输入量化往返误差 (平均): " << rt_err / (N_train * F)
                  << std::endl;

        // 指标: AUC (rank-based Mann-Whitney) + 准确率
        auto auc = [&](const Tensor& logits) {
            std::vector<int> idx(N_test);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](int i, int j) {
                return logits.data[i] < logits.data[j];
            });
            double rank_sum = 0;
            int n_pos = 0;
            for (int r = 0; r < N_test; r++)
                if (y_te[idx[r]] > 0.5f) { rank_sum += r + 1; n_pos++; }
            double n_neg = N_test - n_pos;
            return (rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg);
        };
        auto acc = [&](const Tensor& logits) {
            int hit = 0;
            for (int i = 0; i < N_test; i++)
                hit += ((logits.data[i] > 0) == (y_te[i] > 0.5f));
            return double(hit) / N_test;
        };
        float max_err = 0, mean_err_sum = 0;
        for (int i = 0; i < N_test; i++) {
            float e = std::fabs(logits_fp32.data[i] - logits_i8.data[i]);
            max_err = std::max(max_err, e);
            mean_err_sum += e;
        }
        std::cout << "  fp32: AUC=" << auc(logits_fp32) << "  acc="
                  << acc(logits_fp32) << std::endl;
        std::cout << "  int8: AUC=" << auc(logits_i8) << "  acc="
                  << acc(logits_i8) << std::endl;
        std::cout << "  logits 误差: max=" << max_err << "  mean="
                  << mean_err_sum / N_test << std::endl;
    }
    return 0;
}
