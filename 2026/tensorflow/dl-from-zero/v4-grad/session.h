#pragma once
#include <functional>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "graph/graph.h"
#include "kernels/kernels.h"
#include "runtime.h"

namespace lf {

// Session = 变量持久状态 + 编译缓存 + 每步执行
//  - vars_ : 变量值, 跨 run 持久 (对应 TF VariableOp 的 Var buffer)
//  - plan_ : 按 (图, 目标集) 缓存的拓扑执行计划 (对应 TF GetOrCreateExecutors)
//  - run   : 图是纯静态的, 状态全在局部 RunState, 同一张图可并发跑
// 梯度计算 = 图里的梯度子图 (build_gradients 生成), run 只做 forward + 应用优化器。
class Session {
public:
    Session() = default;

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

        std::vector<Node*> order = plan(graph, targets);
        RunState st;
        st.vars = &vars_;
        st.resize(graph.id_count());
        for (const auto& [n, t] : feeds) st.out(n) = t;

        for (Node* n : order) forward(n, st);

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
    std::unordered_map<size_t, std::vector<Node*>> plan_cache_;

    void ensure_vars(Graph& graph) {
        std::lock_guard<std::mutex> l(mu_);
        for (auto& un : graph.nodes()) {
            Node* n = un.get();
            if (n->type == VARIABLE && !vars_.count(n)) vars_[n] = n->init;
        }
    }

    // 编译缓存: 同 (图, 目标集) 只算一次拓扑序 (对应 TF executor 缓存)
    std::vector<Node*> plan(Graph& graph, const std::vector<Node*>& targets) {
        size_t key = std::hash<const void*>{}(&graph);
        for (Node* t : targets) key = key * 31 + std::hash<const void*>{}(t);
        {
            std::lock_guard<std::mutex> l(plan_mu_);
            auto it = plan_cache_.find(key);
            if (it != plan_cache_.end()) return it->second;
        }
        auto order = topo_sort(graph, targets);
        std::lock_guard<std::mutex> l(plan_mu_);
        plan_cache_[key] = order;
        return order;
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
