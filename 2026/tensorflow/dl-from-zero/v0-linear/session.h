#pragma once
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <string>
#include "graph/graph.h"
#include "kernels/kernels.h"

namespace lf {

class Session {
public:
    Session() = default;

    void run(Graph& graph,
             const std::vector<Node*>& targets,
             const std::unordered_map<Node*, const Tensor*>& feed_dict,
             Node* loss_node = nullptr) {
        auto sorted = topo_sort(graph, targets);

        for (auto& [node, tensor] : feed_dict) {
            node->output = *tensor;
        }

        for (Node* node : sorted) {
            forward(node);
        }

        if (loss_node) {
            for (auto& n : graph.nodes()) {
                n->grad = Tensor(n->output.shape);
            }
            loss_node->grad = Tensor(1.0f);

            for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
                Node* node = *it;
                if (node->type == PLACEHOLDER) continue;
                backward(node);
            }
        }
    }

private:
    std::vector<Node*> topo_sort(Graph& graph,
                                 const std::vector<Node*>& targets) {
        std::unordered_map<Node*, std::vector<Node*>> children;
        std::unordered_set<Node*> needed;

        std::deque<Node*> q(targets.begin(), targets.end());
        for (Node* t : targets) needed.insert(t);
        while (!q.empty()) {
            Node* node = q.front(); q.pop_front();
            for (Node* input : node->inputs) {
                if (needed.insert(input).second) {
                    q.push_back(input);
                }
            }
        }

        for (Node* n : needed) {
            for (Node* input : n->inputs) {
                if (needed.count(input)) children[input].push_back(n);
            }
        }

        std::unordered_map<Node*, int> in_deg;
        for (Node* n : needed) in_deg[n] = 0;
        for (auto& [node, kids] : children) {
            for (Node* child : kids) in_deg[child]++;
        }

        std::deque<Node*> zq;
        for (auto& [n, deg] : in_deg)
            if (deg == 0) zq.push_back(n);

        std::vector<Node*> result;
        while (!zq.empty()) {
            Node* node = zq.front(); zq.pop_front();
            result.push_back(node);
            for (Node* child : children[node]) {
                if (--in_deg[child] == 0) zq.push_back(child);
            }
        }

        return result;
    }
};

}  // namespace lf
