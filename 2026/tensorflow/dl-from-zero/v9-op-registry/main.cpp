#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>
#include "core/place.h"
#include "framework/tensor.h"
#include "graph/gradients.h"
#include "graph/graph.h"
#include "graph/prune.h"
#include "kernels/kernels.h"
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

// 7c 计时: noinline 普通函数 + 函数指针参数 (避免 lambda 展开的不可预测
// 布局; 曾误判为编译器 bug —— 计时恒 0 的真凶是 sme2_gemm.h 的 asm 缺
// v8-v15 clobber: smstart/smstop 清零 callee-saved d8-d15, 跨调用缓存的
// best (d8) / 1e6 (d9) 被静默破坏 → fdiv 除零 inf、best 恒 0)。
typedef void (*kern_fn)(int, int, int, const float*, const float*, float*);
__attribute__((noinline)) static double call_bench(kern_fn kern, int mm, int nn,
                                                   int kk, const float* a,
                                                   const float* b, float* c) {
    kern(mm, nn, kk, a, b, c);  // 暖机
    double best = 1e30;
    for (int t = 0; t < 10; t++) {
        auto t0 = std::chrono::steady_clock::now();
        kern(mm, nn, kk, a, b, c);
        // volatile 逃逸: 7c 之后 got/g_got 不再被读取, -O2 下编译器会判定
        // kern 调用结果无用而整段删除 → 计时恒 0
        volatile float sink = c[0];
        (void)sink;
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
        if (ms < best) best = ms;
    }
    return best;
}

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);  // 重定向时也逐行刷 (调试友好)
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
    //   耗时是 convoy 验收指标: 旧版固定 worker 池一个 run 卡 GPU 等待时
    //   独占全部池线程, 并发 run 串行 (趋近一次一个请求); closure 模型下
    //   空闲 run 不占线程, 8 线程应真并发。
    const int n_threads = 8, iters = 5000;
    std::vector<std::thread> threads;
    std::vector<double> mean_err(n_threads, 0.0);
    auto t_conc_0 = std::chrono::steady_clock::now();
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
    double conc_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t_conc_0)
                         .count();

    std::cout << "\n  concurrent inference: " << n_threads << " threads x " << iters
              << " runs on the same graph: " << conc_ms << " ms total" << std::endl;
    for (int t = 0; t < n_threads; t++) {
        float tx = -4.0f + t;
        std::cout << "    thread " << t << "  x=" << tx
                  << "  mean_abs_err=" << mean_err[t] << std::endl;
    }

    // ============================================================
    // demo 5: 执行队列 —— op 并发执行 (对应 TF commit 1 的 executor.cc)
    //   图: 8 个互不依赖的 matmul 分支 (天然可并发) + 树状 add 合并 + mean
    //   同一套调度代码, 只是 worker 数不同: 1 (顺序) vs 硬件并发数 (并行)
    //   调度只影响执行顺序, 每个 op 的 kernel 不变 → 输出必须逐位一致
    //   注意: matmul 显式 CPU —— 本 demo 单独演示执行队列 (worker 池调度),
    //   设备放置 (auto-GPU) 由 demo 6 演示
    //   matmul 关掉 SME2 走标量: ZA (AMX) 在并发线程间不隔离, 必须全局互斥,
    //   并发 matmul 会退化为顺序 (实测加速比 ~1.0x, 诚实记录在 README) ——
    //   demo 5 要展示的是执行队列的并行调度, matmul kernel 的速度是 demo 7
    //   的舞台 (7c: SME2 gemm 2.6x); 块内开关, 跑完恢复
    // ============================================================
    std::cout << "\n== demo 5: 执行队列 (op 并发执行) ==" << std::endl;
    {
        const int N = 8192, F = 512, n_branch = 8;
        matmul_use_sme2 = false;  // 标量路径: 无锁, 真并发 (见上面注释)
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);

        Graph g;
        auto x = g.placeholder("x", {N, F});
        std::vector<Node*> branches;
        for (int i = 0; i < n_branch; i++) {
            Tensor w(std::vector<int>{F});
            for (int f = 0; f < F; f++) w.data[f] = u(rng);
            Node* mm = g.matmul(x, g.constant("w" + std::to_string(i), w));
            g.on(mm, Device::CPU);  // 显式 CPU (见上面注释)
            branches.push_back(g.sigmoid(mm));
        }
        while (branches.size() > 1) {  // 树状合并
            std::vector<Node*> next;
            for (size_t i = 0; i + 1 < branches.size(); i += 2)
                next.push_back(g.add(branches[i], branches[i + 1]));
            if (branches.size() % 2) next.push_back(branches.back());
            branches = next;
        }
        auto y = g.mean(branches[0]);

        Tensor feed(std::vector<int>{N, F});
        for (int i = 0; i < N * F; i++) feed.data[i] = u(rng);

        auto timed = [&](Session& s, double& ms, Tensor& out) {
            s.run(g, {y}, {{x, feed}});  // 暖机: 优化 + 编译 + 线程池就绪
            auto t0 = std::chrono::steady_clock::now();
            out = s.run(g, {y}, {{x, feed}})[0];
            ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t0)
                     .count();
        };

        Session s_seq;
        s_seq.SetWorkers(1);  // 顺序: 单 worker
        double ms_seq;
        Tensor r_seq;
        timed(s_seq, ms_seq, r_seq);

        const int n_workers = static_cast<int>(std::thread::hardware_concurrency());
        Session s_par;
        s_par.SetWorkers(n_workers);  // 并行: 硬件并发数 worker
        double ms_par;
        Tensor r_par;
        timed(s_par, ms_par, r_par);

        std::cout << "  图: " << n_branch << " 个独立 matmul 分支 + 树状合并 ("
                  << g.nodes().size() << " nodes, [8192,512]×[512])" << std::endl;
        std::cout << "  顺序 (1 worker):  " << ms_seq << " ms   y=" << r_seq.data[0]
                  << std::endl;
        std::cout << "  并行 (" << n_workers << " workers): " << ms_par
                  << " ms   y=" << r_par.data[0] << std::endl;
        bool bit_ok = allclose(r_seq, r_par, 0.0f);
        std::cout << "  输出逐位一致: " << (bit_ok ? "yes" : "NO") << std::endl;
        if (!bit_ok) {  // 临时调试: NO 时重跑 par 验证稳定性
            auto r_par2 = s_par.run(g, {y}, {{x, feed}})[0];
            std::cout << "    rerun par: y=" << r_par2.data[0]
                      << " (seq=" << r_seq.data[0] << " par1=" << r_par.data[0]
                      << ")" << std::endl;
        }
        std::cout << "  加速比: " << ms_seq / ms_par << "x" << std::endl;
        matmul_use_sme2 = true;  // 恢复: 后续 demo (训练/7/8) 走 SME2
    }

    // ============================================================
    // demo 6: 设备放置 —— op 落在哪个设备跑 (对应 commit 1 的 simple_placer)
    //   matmul 显式 GPU (Metal compute shader 异步执行); 逐元素 op 无 GPU
    //   kernel → 自动 CPU。参考图显式 CPU: 否则自动放置也把 matmul 放 GPU,
    //   对照就不成立了。GPU/CPU 各跑各的 kernel, 结果容差内一致 (累加顺序
    //   相同, 但 FMA 收缩等编译差异允许最后一位不同)。
    // ============================================================
    std::cout << "\n== demo 6: 设备放置 (simple_placer) ==" << std::endl;
    {
        const int N = 4096, F = 256;
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);

        Tensor wv(std::vector<int>{F});  // 两张图共用同一份权重 (公平对照)
        for (int f = 0; f < F; f++) wv.data[f] = u(rng);
        auto build = [&](bool gpu_matmul) {
            Graph g;
            auto x = g.placeholder("x", {N, F});
            auto mm = g.matmul(x, g.constant("w", wv));
            auto y = g.sigmoid(g.add(mm, g.constant("b", Tensor(0.1f))));
            g.on(mm, gpu_matmul ? Device::GPU : Device::CPU);  // 显式放置
            return std::make_tuple(std::move(g), x, y);
        };
        auto [g_gpu, x_g, y_g] = build(true);
        auto [g_cpu, x_c, y_c] = build(false);

        // 放置结果: 显式 GPU 优先; 未指定的 matmul 若 Metal 可用也自动 GPU
        // (kernel 注册表), 其余 op 无 GPU kernel → 自动 CPU
        auto placed = SimplePlace(g_gpu, true);
        std::cout << "  混设备图 placement (name: device):" << std::endl;
        for (const auto& un : g_gpu.nodes()) {
            std::cout << "    " << un->name << ": "
                      << (placed[un->id] == Device::GPU ? "GPU" : "CPU");
            if (un->device == Device::GPU) std::cout << "  (显式)";
            std::cout << std::endl;
        }

        Tensor feed(std::vector<int>{N, F});
        for (int i = 0; i < N * F; i++) feed.data[i] = u(rng);

        Session s_gpu, s_cpu;
        auto r_gpu = s_gpu.run(g_gpu, {y_g}, {{x_g, feed}})[0];
        auto r_cpu = s_cpu.run(g_cpu, {y_c}, {{x_c, feed}})[0];
        std::cout << "  混设备结果 == 全 CPU 参考: "
                  << (allclose(r_gpu, r_cpu, 1e-5f) ? "yes" : "NO") << std::endl;

        // 耗时对比: 混设备 = GPU async matmul (Metal dispatch) + CPU 逐元素,
        // 与 CPU matmul 单次执行重叠; 小形状下 dispatch 开销占比大 (诚实数字)
        auto timed = [&](Session& s, Graph& g, Node* y, Node* x) {
            s.run(g, {y}, {{x, feed}});  // 暖机
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 20; i++) s.run(g, {y}, {{x, feed}});
            return std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t0)
                       .count() /
                   20;
        };
        std::cout << "  平均单次耗时: 混设备 " << timed(s_gpu, g_gpu, y_g, x_g)
                  << " ms  vs  全 CPU " << timed(s_cpu, g_cpu, y_c, x_c) << " ms"
                  << std::endl;
    }

    // ============================================================
    // demo 7: SME2/AMX CPU matmul (对应 commit 1 的 Eigen → 本机 AMX)
    //   framework 的 matmul ([N,F]×[F]→[N]) 在 M4 上走 sme2_gemm:
    //   mopa 外积 → ZA tile (AMX), 与 Metal 的 GPU matmul 呼应 ——
    //   commit 1 的 CPU 设备用 Eigen (SIMD), 这里换成手写 SME2 asm。
    //   7a: 框架形态正确性 (整数数据 → 与标量逐位一致)
    //   7b: GEMM 形态 (N%4==0, 4×4 tile 吃满)
    //   7c: 实测耗时 —— 诚实数字: matvec 只用 ZA 一列 (1/4 tile),
    //       且 asm 首版未 unroll (每 k 8 条 insr 串行), 加速有限;
    //       GEMM 4×4 每指令 16 MAC, 是 AMX 的正确打开方式
    // ============================================================
    std::cout << "\n== demo 7: SME2/AMX CPU matmul ==" << std::endl;
    {
        std::cout << "  CPU: ";
#if LF_SME2
        // streaming mode VL 固定 128bit (ZA 4x4 f32); svcntw() 只能在
        // streaming 函数里调, 这里直接编译期写死
        std::cout << "M4 FEAT_SME2 (ZA 4x4 f32, VL=128bit streaming)";
#else
        std::cout << "无 SME2 → cblas_sgemm fallback";
#endif
        std::cout << std::endl;

        // 7a: 框架形态 [8192,512]×[512] → [8192]
        const int N = 8192, F = 512;
        Tensor x(std::vector<int>{N, F}), w(std::vector<int>{F});
        for (int i = 0; i < N * F; i++) x.data[i] = (float)(i % 1000);
        for (int f = 0; f < F; f++) w.data[f] = (float)(f + 1);
        Tensor ref(std::vector<int>{N}), got(std::vector<int>{N});
        gemm_scalar(N, 1, F, x.data.data(), w.data.data(), ref.data.data());
        sme2_gemm(N, 1, F, x.data.data(), w.data.data(), got.data.data());
        std::cout << "  7a matvec [8192,512]x[512] == scalar: "
                  << (allclose(ref, got, 0.0f) ? "yes (逐位)" : "NO") << std::endl;

        // 7b: GEMM 形态 [1024,256]×[256,128]
        const int M = 1024, NG = 128, K = 256;
        Tensor A(std::vector<int>{M * K}), B(std::vector<int>{K * NG});
        for (int i = 0; i < M * K; i++) A.data[i] = (float)(i % 1000);
        for (int j = 0; j < K * NG; j++) B.data[j] = (float)(j % 1000);
        Tensor g_ref(std::vector<int>{M * NG}), g_got(std::vector<int>{M * NG});
        gemm_scalar(M, NG, K, A.data.data(), B.data.data(), g_ref.data.data());
        sme2_gemm(M, NG, K, A.data.data(), B.data.data(), g_got.data.data());
        bool gok = true;
        for (int i = 0; i < M * NG; i++)
            if (g_ref.data[i] != g_got.data[i]) { gok = false; break; }
        std::cout << "  7b gemm  [1024,256]x[256,128] == scalar: "
                  << (gok ? "yes (逐位)" : "NO") << std::endl;


        double mv_sc = call_bench(gemm_scalar, N, 1, F, x.data.data(), w.data.data(),
                             got.data.data());
        double mv_za = call_bench(sme2_gemm, N, 1, F, x.data.data(), w.data.data(),
                             got.data.data());
        double gm_sc = call_bench(gemm_scalar, M, NG, K, A.data.data(), B.data.data(),
                             g_got.data.data());
        double gm_za = call_bench(sme2_gemm, M, NG, K, A.data.data(), B.data.data(),
                             g_got.data.data());
        double mv_gflops = 2.0 * N * F / 1e9 / (mv_za / 1e3);
        double gm_gflops = 2.0 * M * NG * K / 1e9 / (gm_za / 1e3);
        std::cout << "  7c matvec: scalar " << mv_sc << " ms  vs  SME2 " << mv_za
                  << " ms  (" << mv_sc / mv_za << "x, " << mv_gflops << " GFLOPS)"
                  << std::endl;
        std::cout << "  7c gemm : scalar " << gm_sc << " ms  vs  SME2 " << gm_za
                  << " ms  (" << gm_sc / gm_za << "x, " << gm_gflops << " GFLOPS)"
                  << std::endl;
    }

    // ============================================================
    // demo 8: 并发 GPU 推理 —— closure 模型的 convoy 修复验收 (v8)
    //   场景: 混设备图 (GPU matmul 异步 + CPU 逐元素) × 多 worker × 8 线程并发
    //   旧版固定 worker 池: run A 的 worker 全部 park 在 GPU 异步等待, 独占
    //   池线程; run B 的任务饿死 → 并发 run 串行 (总耗时 ≈ run 数 × 单次)。
    //   实测基线 (v7 初版 executor): 8 线程 x 200 runs ≈ 933 ms。
    //   closure 模型: GPU 节点提交后闭包返回、线程归还, 其他 run 的 CPU
    //   计算插空执行 → GPU 等待与其他 run 重叠, 应显著低于串行基线。
    // ============================================================
    std::cout << "\n== demo 8: 并发 GPU 推理 (convoy 修复验收) ==" << std::endl;
    {
        const int N = 4096, F = 256;
        const int n_threads = 8, iters = 200;
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        Tensor wv(std::vector<int>{F});
        for (int f = 0; f < F; f++) wv.data[f] = u(rng);
        Graph g;
        auto x = g.placeholder("x", {N, F});
        auto mm = g.matmul(x, g.constant("w", wv));
        auto y = g.sigmoid(g.add(mm, g.constant("b", Tensor(0.1f))));
        g.on(mm, Device::GPU);  // 同 demo 6 的混设备图
        Tensor feed(std::vector<int>{N, F});
        for (int i = 0; i < N * F; i++) feed.data[i] = u(rng);

        Session sess;
        sess.SetWorkers(8);  // 多 worker: 旧版此场景全部 park 占池
        sess.run(g, {y}, {{x, feed}});  // 暖机

        std::vector<std::thread> threads;
        double mean_y = 0.0;
        auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < n_threads; t++) {
            threads.emplace_back([&, t]() {
                double acc = 0.0;
                for (int i = 0; i < iters; i++) {
                    auto r = sess.run(g, {y}, {{x, feed}});
                    acc += r[0].data[0];
                }
                if (t == 0) mean_y = acc / iters;
            });
        }
        for (auto& th : threads) th.join();
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
        std::cout << "  " << n_threads << " threads x " << iters << " runs: "
                  << ms << " ms total (mean_y=" << mean_y
                  << ", 旧版固定 worker 池基线 ≈ 933 ms)" << std::endl;
    }

    // ========== demo 9: 跨设备图分区 + Send/Recv (v8) ==========
    std::cout << "\n== demo 9: 跨设备图分区 (Graph Partition + Send/Recv) ==" << std::endl;
    {
        Graph g;
        // 跨设备图: x (CPU) -> matmul (GPU 显式) -> add (CPU) -> sigmoid (CPU)
        //                          ↑
        //                     w (GPU 显式)
        Node* x = g.placeholder("x", {4, 128});
        Node* w = g.variable_vec("w", 128, 0.1f);
        Node* matmul = g.matmul(x, w);
        Node* b = g.variable("b", 0.5f);
        Node* add = g.add(matmul, b);
        Node* y = g.sigmoid(add);

        // 显式设备放置: matmul 和 w 放 GPU, 其余 CPU
        g.on(matmul, Device::GPU);
        g.on(w, Device::GPU);

        Session sess;
        sess.SetWorkers(2);

        // 第一次 run: 触发图分区 (在跨设备边插入 Send/Recv)
        Tensor x_val({4, 128});
        std::fill(x_val.data.begin(), x_val.data.end(), 0.1f);
        auto result = sess.run(g, {y}, {{x, x_val}});

        std::cout << "  分区后图节点数: " << g.nodes().size() << " nodes" << std::endl;
        std::cout << "  (原始 6 nodes → 分区后应插入 Send/Recv 节点)" << std::endl;

        // 验证: 全 CPU 参考
        Graph g_ref;
        Node* x_ref = g_ref.placeholder("x", {4, 128});
        Node* w_ref = g_ref.variable_vec("w", 128, 0.1f);
        Node* matmul_ref = g_ref.matmul(x_ref, w_ref);
        Node* b_ref = g_ref.variable("b", 0.5f);
        Node* add_ref = g_ref.add(matmul_ref, b_ref);
        Node* y_ref = g_ref.sigmoid(add_ref);

        Session sess_ref;
        auto result_ref = sess_ref.run(g_ref, {y_ref}, {{x_ref, x_val}});

        std::cout << "  跨设备结果 == 全 CPU 参考: "
                  << (allclose(result[0], result_ref[0]) ? "yes" : "NO") << std::endl;
        std::cout << "  (验证 Send/Recv 正确传输数据)" << std::endl;

        // 打印图结构 (查看插入的 Send/Recv)
        std::cout << "  分区后的节点列表:" << std::endl;
        for (const auto& un : g.nodes()) {
            Node* n = un.get();
            const char* dev = (n->device == Device::GPU)   ? "GPU"
                              : (n->device == Device::CPU) ? "CPU"
                                                           : "AUTO";
            std::cout << "    " << n->name << " (" << dev << ")";
            if (n->type == SEND || n->type == RECV) {
                std::cout << " [key=" << n->rendezvous_key << "]";
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
