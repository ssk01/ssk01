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
//  - vars_    : 变量值, 跨 run 持久 (对应 TF VariableOp 的 Var buffer)
//  - plan_    : 按 (图, 目标集) 缓存的拓扑执行计划 (对应 TF GetOrCreateExecutors)
//  - run      : 图是纯静态的, 状态全在局部 RunState, 所以同一张图可并发跑
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
    std::vector<Tensor> run(Graph& graph,
                            const std::vector<Node*>& targets,
                            const std::unordered_map<const Node*, Tensor>& feeds,
                            Node* loss_node = nullptr) {
        ensure_vars(graph);

        std::vector<Node*> order = plan(graph, targets);
        RunState st;
        st.vars = &vars_;
        for (const auto& [n, t] : feeds) st.outputs[n] = t;

        for (Node* n : order) forward(n, st);

        if (loss_node) {
            for (const auto& [n, out] : st.outputs) st.grads[n] = Tensor(out.shape);
            st.grads[loss_node] = Tensor(1.0f);
            for (auto it = order.rbegin(); it != order.rend(); ++it) backward(*it, st);
            apply_updates(graph, st);
        }

        std::vector<Tensor> results;
        results.reserve(targets.size());
        for (Node* t : targets) results.push_back(st.outputs.at(t));
        return results;
    }

private:
    mutable std::mutex mu_;                    // 保护 vars_
    std::unordered_map<const Node*, Tensor> vars_;
    mutable std::mutex plan_mu_;               // 保护 plan_cache_
    std::unordered_map<size_t, std::vector<Node*>> plan_cache_;

    void ensure_vars(Graph& graph) {
        std::lock_guard<std::mutex> l(mu_);
        for (auto& un : graph.nodes()) {
            Node* n = un.get();
            if (n->type == VARIABLE && !vars_.count(n)) vars_[n] = n->init;
        }
    }

    // 编译缓存: 同 (图, 目标集) 只算一次拓扑序, 之后复用
    // (对应 TF 的 executor 缓存 —— "图只构建一次", 也省掉每步的排序开销)
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

    // 拓扑排序, 两阶段, 直白版:
    //   1) 反向 BFS: 只保留能算出 targets 的节点 (可达集 needed)
    //   2) DFS 后序再逆序: 前驱一定排在后继之前, 得到合法执行序
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
                if (needed.count(in)) dfs(in);   // needed 内必然都可达
            order.push_back(n);
        };
        for (Node* n : needed) dfs(n);
        // DFS 沿 inputs(上游)递归、退出时 push: 得到的就是"上游在前"的执行序
        return order;
    }

    // 应用图上所有 SGD_STEP 节点: var -= lr * grad (对应 TF ApplyGradientDescent)
    void apply_updates(Graph& graph, RunState& st) {
        std::lock_guard<std::mutex> l(mu_);
        for (auto& un : graph.nodes()) {
            Node* n = un.get();
            if (n->type != SGD_STEP) continue;
            Node* var = n->inputs[0];
            const Tensor& g = st.grads.at(var);
            Tensor& v = vars_[var];
            for (int i = 0; i < v.size(); i++) v.data[i] -= n->scalar * g.data[i];
        }
    }
};

}  // namespace lf
