#pragma once
#include <functional>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "core/executor.h"
#include "core/metal_matmul.h"
#include "core/rendezvous.h"        // v8
#include "core/place.h"             // v8
#include "graph/graph.h"
#include "graph/optimize.h"
#include "graph/partition.h"        // v8
#include "kernels/kernels.h"
#include "runtime.h"

namespace lf {

// Session = 变量持久状态 + 编译缓存 + 每步执行
//  - vars_ : 变量值, 跨 run 持久 (对应 TF VariableOp 的 Var buffer)
//  - plan_ : 按 (图, 目标集) 缓存的执行计划 (对应 TF GetOrCreateExecutors)
//  - run   : 图是纯静态的, 状态全在局部 RunState, 同一张图可并发跑
//  - 执行  : v7 起走 Executor (就绪队列并发调度, 对应 commit 1 executor.cc);
//            SetWorkers(1) 即顺序执行 (同一条调度代码, 无并发)
// 梯度计算 = 图里的梯度子图 (build_gradients 生成), run 只做 forward + 应用优化器。
class Session {
public:
    Session() = default;

    // 并发度: 执行队列的 worker 数 (1 = 顺序)。重建 worker 池, 只应在
    // 没有并发 run 的时候调用 (demo 边界)。
    void SetWorkers(int n) { executor_ = std::make_unique<Executor>(std::max(1, n)); }

    void assign(const Node* var, const Tensor& v) {
        std::lock_guard<std::mutex> l(mu_);
        vars_[var] = v;
    }

    const Tensor& var_value(const Node* var) const {
        std::lock_guard<std::mutex> l(mu_);
        return vars_.at(var);
    }

    // 返回 targets 的输出 (TF 风格: sess.run(fetches) 返回张量列表)
    // 训练: targets 里带上 sgd_step 节点; 纯推理: 不带。
    std::vector<Tensor> run(Graph& graph,
                            const std::vector<Node*>& targets,
                            const std::unordered_map<const Node*, Tensor>& feeds) {
        ensure_vars(graph);
        optimize_once(graph);   // 图优化 pass pipeline (幂等: 图没变就跳过)

        // v8: 设备放置 + 图分区
        partition_once(graph);

        const GraphPlan& p = plan(graph, targets);
        RunState st;
        st.vars = &vars_;
        st.resize(graph.id_count());
        for (const auto& [n, t] : feeds) st.out(n) = t;

        // v8: 如果有跨设备边, 创建 rendezvous
        Rendezvous rdv;
        if (has_cross_device_) {
            st.rendezvous = &rdv;
        }

        executor_->Run(p, st);   // 执行队列调度 (v7: op 并发执行)

        apply_updates(graph, st);   // 应用 SGD_STEP(只处理本轮算出了梯度的)

        std::vector<Tensor> results;
        results.reserve(targets.size());
        for (Node* t : targets) results.push_back(st.out(t));
        return results;
    }

private:
    mutable std::mutex mu_;
    std::unordered_map<const Node*, Tensor> vars_;
    mutable std::mutex plan_mu_;
    std::unordered_map<size_t, GraphPlan> plan_cache_;
    mutable std::mutex opt_mu_;
    int optimized_gen_ = -1;
    mutable std::mutex partition_mu_;        // v8
    int partitioned_gen_ = -1;               // v8
    bool has_cross_device_ = false;          // v8
    std::unique_ptr<Executor> executor_ = std::make_unique<Executor>(1);

    void ensure_vars(Graph& graph) {
        std::lock_guard<std::mutex> l(mu_);
        for (auto& un : graph.nodes()) {
            Node* n = un.get();
            if (n->type == VARIABLE && !vars_.count(n)) vars_[n] = n->init;
        }
    }

    // 每次 run 前跑图优化 pass (对应论文 §5 master 的优化阶段; TF 在 master 端
    // 对每个 step 的子图做优化并缓存)。幂等: 图的结构代数没变就跳过 ——
    // 第一次 run 承担优化成本, 后续 run 复用。
    // 注意: 优化会变换图 (CSE 合并/删除节点), 持有被删节点指针再 run 是未定义
    // 行为 (真实 TF 图构造后不可变, 优化在子图副本上进行)。
    void optimize_once(Graph& graph) {
        std::lock_guard<std::mutex> l(opt_mu_);
        if (graph.generation() == optimized_gen_) return;
        optimize(graph);
        optimized_gen_ = graph.generation();
    }

    // v8: 图分区 (幂等: 图没变就跳过)
    // 设备放置 + 跨设备边插入 Send/Recv。对应 TF 的 simple_placer + graph_partition。
    void partition_once(Graph& graph) {
        std::lock_guard<std::mutex> l(partition_mu_);
        if (graph.generation() == partitioned_gen_) return;

        // 1. 设备放置
        auto devices = SimplePlace(graph, metal::Available());

        // 2. 图分区: 跨设备边插入 Send/Recv
        has_cross_device_ = PartitionGraph(graph, devices);

        partitioned_gen_ = graph.generation();
    }

    // 编译缓存: 同 (图, 目标集, 结构代数) 只算一次执行计划 (对应 TF executor 缓存;
    // 图被优化变换后代数变化 → 缓存自动失效)。计划 = 拓扑序 + 执行队列需要的
    // 稠密表 (by_id / consumers / in_degree) —— executor 编译产物的最小形态。
    const GraphPlan& plan(Graph& graph, const std::vector<Node*>& targets) {
        size_t key = std::hash<const void*>{}(&graph) * 31 +
                     static_cast<size_t>(graph.generation());
        for (Node* t : targets) key = key * 31 + std::hash<const void*>{}(t);
        {
            std::lock_guard<std::mutex> l(plan_mu_);
            auto it = plan_cache_.find(key);
            if (it != plan_cache_.end()) return it->second;
        }
        GraphPlan p;
        p.order = topo_sort(graph, targets);
        p.by_id.assign(graph.id_count(), nullptr);
        p.consumers.assign(graph.id_count(), {});
        p.in_degree.assign(graph.id_count(), 0);
        for (Node* n : p.order) {
            p.by_id[n->id] = n;
            p.in_degree[n->id] = static_cast<int>(n->inputs.size());
            for (Node* in : n->inputs) p.consumers[in->id].push_back(n->id);
        }
        // 设备放置 (simple_placer 对应): 显式 device 优先; 未指定的 matmul 若
        // Metal 可用 → GPU (kernel 注册表 GPU 优先), 其余 CPU。Metal 不可用 → 全 CPU。
        p.devices = SimplePlace(graph, metal::Available());
        std::lock_guard<std::mutex> l(plan_mu_);
        auto [it, _] = plan_cache_.emplace(key, std::move(p));
        return it->second;
    }

    // 拓扑排序, 两阶段直白版:
    //   1) 反向 BFS 收集 targets 可达集;  2) DFS 后序(沿 inputs 递归、退出时 push) = 上游在前的执行序
    static std::vector<Node*> topo_sort(Graph& graph,
                                        const std::vector<Node*>& targets) {
        std::unordered_set<Node*> needed;
        std::vector<Node*> queue(targets.begin(), targets.end());
        for (Node* t : targets) needed.insert(t);
        for (size_t i = 0; i < queue.size(); i++)
            for (Node* in : queue[i]->inputs)
                if (needed.insert(in).second) queue.push_back(in);

        std::unordered_set<Node*> visited;
        std::vector<Node*> order;
        std::function<void(Node*)> dfs = [&](Node* n) {
            if (!visited.insert(n).second) return;
            for (Node* in : n->inputs)
                if (needed.count(in)) dfs(in);
            order.push_back(n);
        };
        for (Node* n : needed) dfs(n);
        return order;
    }

    // 应用图上所有 SGD_STEP 节点: var -= lr * grad (对应 TF ApplyGradientDescent)
    // 只处理本轮确实算出了梯度的节点(即 targets 覆盖了梯度子图)。
    void apply_updates(Graph& graph, RunState& st) {
        std::lock_guard<std::mutex> l(mu_);
        for (auto& un : graph.nodes()) {
            Node* n = un.get();
            if (n->type != SGD_STEP) continue;
            Node* var = n->inputs[0];
            Node* grad_node = n->inputs[1];
            if (!st.computed(grad_node)) continue;   // 本轮没算梯度(纯推理), 不动
            const Tensor& grad = st.out(grad_node);
            Tensor& v = vars_[var];
            for (int i = 0; i < v.size(); i++) v.data[i] -= n->scalar * grad.data[i];
        }
    }
};

}  // namespace lf
