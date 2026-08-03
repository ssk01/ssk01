#pragma once
#include <deque>
#include <unordered_set>
#include <vector>
#include "graph.h"

namespace lf {

// 训练图 → 推理图: 从 targets 反向 BFS, 只保留能算出 targets 的节点
// 对应 TF 的 PruneForReverseReachability (tensorflow/core/graph/algorithm.cc)
inline std::unordered_set<Node*> reachable_from(const std::vector<Node*>& targets) {
    std::unordered_set<Node*> keep;
    std::deque<Node*> q(targets.begin(), targets.end());
    for (Node* t : targets) keep.insert(t);
    while (!q.empty()) {
        Node* node = q.front(); q.pop_front();
        for (Node* in : node->inputs) {
            if (keep.insert(in).second) q.push_back(in);
        }
    }
    return keep;
}

inline void prune(Graph& g, const std::vector<Node*>& targets) {
    g.keep_only(reachable_from(targets));
}

}  // namespace lf
