#include <iostream>
#include "graph/graph.h"
#include "core/place.h"
#include "core/metal_matmul.h"
#include "graph/partition.h"

using namespace lf;

int main() {
    std::cout << "== 调试 SimplePlace 的设备分配 ==" << std::endl;

    Graph g;
    Node* x = g.placeholder("x", {4, 128});
    Node* w = g.variable_vec("w", 128, 0.1f);
    Node* matmul = g.matmul(x, w);
    Node* y = g.sigmoid(matmul);

    std::cout << "\n设备放置结果 (Metal 可用=" << metal::Available() << "):" << std::endl;
    auto devices = SimplePlace(g, metal::Available());

    for (const auto& un : g.nodes()) {
        Node* n = un.get();
        const char* dev = (devices[n->id] == Device::GPU) ? "GPU"
                          : (devices[n->id] == Device::CPU) ? "CPU"
                                                             : "AUTO";
        std::cout << "  " << n->name << " -> " << dev;
        std::cout << " (node->device=" << (int)n->device << ")" << std::endl;
    }

    // 检查是否有跨设备边
    bool has_cross = false;
    for (const auto& un : g.nodes()) {
        Node* n = un.get();
        for (Node* input : n->inputs) {
            if (devices[input->id] != devices[n->id]) {
                std::cout << "\n跨设备边: " << input->name
                          << " (" << (devices[input->id] == Device::GPU ? "GPU" : "CPU") << ")"
                          << " -> " << n->name
                          << " (" << (devices[n->id] == Device::GPU ? "GPU" : "CPU") << ")"
                          << std::endl;
                has_cross = true;
            }
        }
    }

    if (!has_cross) {
        std::cout << "\n无跨设备边（全同设备）" << std::endl;
    }

    return 0;
}
