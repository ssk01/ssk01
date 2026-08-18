#include <iostream>
#include "graph/graph.h"
#include "graph/partition.h"
#include "core/place.h"

using namespace lf;

// 简化的拓扑排序 (只用于调试)
std::vector<Node*> simple_topo(const Graph& g) {
    std::vector<Node*> result;
    std::unordered_set<Node*> visited;

    std::function<void(Node*)> dfs = [&](Node* n) {
        if (visited.count(n)) return;
        visited.insert(n);
        for (Node* in : n->inputs) dfs(in);
        result.push_back(n);
    };

    for (const auto& un : g.nodes()) {
        dfs(un.get());
    }
    return result;
}

int main() {
    std::cout << "== 调试: 图分区后的拓扑序 ==" << std::endl;

    Graph g;
    Node* x = g.placeholder("x", {4, 128});
    Node* w = g.variable_vec("w", 128, 0.1f);
    Node* matmul = g.matmul(x, w);
    Node* b = g.variable("b", 0.5f);
    Node* y = g.add(matmul, b);

    g.on(matmul, Device::GPU);
    g.on(w, Device::GPU);

    std::cout << "\n分区前的拓扑序:" << std::endl;
    auto order_before = simple_topo(g);
    for (size_t i = 0; i < order_before.size(); ++i) {
        std::cout << "  " << i << ". " << order_before[i]->name << std::endl;
    }

    // 图分区
    auto devices = SimplePlace(g, true);
    PartitionGraph(g, devices);

    std::cout << "\n分区后的拓扑序:" << std::endl;
    auto order_after = simple_topo(g);
    for (size_t i = 0; i < order_after.size(); ++i) {
        Node* n = order_after[i];
        const char* dev = (n->device == Device::GPU) ? "GPU" : "CPU";
        std::cout << "  " << i << ". " << n->name << " (" << dev << ")";
        if (n->type == SEND) std::cout << " <- inputs[0]=" << n->inputs[0]->name;
        if (n->type == RECV) std::cout << " [key=" << n->rendezvous_key << "]";
        std::cout << std::endl;
    }

    std::cout << "\n检查 Send/Recv 顺序:" << std::endl;
    for (size_t i = 0; i < order_after.size(); ++i) {
        Node* n = order_after[i];
        if (n->type == RECV) {
            // 找对应的 Send
            Node* matching_send = nullptr;
            for (size_t j = 0; j < i; ++j) {
                Node* prev = order_after[j];
                if (prev->type == SEND && prev->rendezvous_key == n->rendezvous_key) {
                    matching_send = prev;
                    break;
                }
            }
            if (matching_send) {
                std::cout << "  ✓ " << n->name << " 在对应 Send 之后" << std::endl;
            } else {
                std::cout << "  ✗ " << n->name << " 在对应 Send 之前 (会死锁!)" << std::endl;
            }
        }
    }

    return 0;
}
