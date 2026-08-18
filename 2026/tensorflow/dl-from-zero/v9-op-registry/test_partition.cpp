#include <iostream>
#include "graph/graph.h"
#include "graph/partition.h"
#include "core/place.h"
#include "session.h"

using namespace lf;

int main() {
    std::cout << "== v8 Graph Partition Test ==" << std::endl;

    Graph g;
    // 跨设备图: x (CPU) -> matmul (GPU) -> add (CPU)
    Node* x = g.placeholder("x", {4, 128});
    Node* w = g.variable_vec("w", 128, 0.1f);
    Node* matmul = g.matmul(x, w);
    Node* b = g.variable("b", 0.5f);
    Node* y = g.add(matmul, b);

    // 显式设备放置
    g.on(matmul, Device::GPU);
    g.on(w, Device::GPU);

    std::cout << "原始图节点数: " << g.nodes().size() << " nodes" << std::endl;

    // 设备放置
    auto devices = SimplePlace(g, true);

    std::cout << "设备放置:" << std::endl;
    for (const auto& un : g.nodes()) {
        Node* n = un.get();
        const char* dev = (devices[n->id] == Device::GPU) ? "GPU" : "CPU";
        std::cout << "  " << n->name << " -> " << dev << std::endl;
    }

    // 图分区
    bool has_cross = PartitionGraph(g, devices);

    std::cout << "\n图分区完成: " << (has_cross ? "有跨设备边" : "无跨设备边") << std::endl;
    std::cout << "分区后节点数: " << g.nodes().size() << " nodes" << std::endl;

    std::cout << "\n分区后的节点:" << std::endl;
    for (const auto& un : g.nodes()) {
        Node* n = un.get();
        const char* dev = (n->device == Device::GPU) ? "GPU"
                          : (n->device == Device::CPU) ? "CPU" : "AUTO";
        std::cout << "  " << n->name << " (" << dev << ")";
        if (n->type == SEND || n->type == RECV) {
            std::cout << " [key=" << n->rendezvous_key << "]";
        }
        std::cout << std::endl;
    }

    return 0;
}
