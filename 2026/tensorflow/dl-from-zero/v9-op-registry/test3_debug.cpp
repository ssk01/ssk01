#include <iostream>
#include "graph/graph.h"
#include "session.h"

using namespace lf;

int main() {
    std::cout << "== 测试: 全 CPU 图不应分区 ==" << std::endl;

    Graph g;
    Node* x = g.placeholder("x", {4, 128});
    Node* w = g.variable_vec("w", 128, 0.1f);
    Node* matmul = g.matmul(x, w);
    Node* y = g.sigmoid(matmul);

    // 不显式指定设备 → 应该全 CPU（假设 Metal 不可用或 matmul 自动放 CPU）
    std::cout << "原始图节点数: " << g.nodes().size() << std::endl;

    Session sess;
    Tensor x_val({4, 128});
    std::fill(x_val.data.begin(), x_val.data.end(), 0.1f);

    // 第一次 run
    auto r1 = sess.run(g, {y}, {{x, x_val}});
    std::cout << "第一次 run 后节点数: " << g.nodes().size() << std::endl;

    // 第二次 run（应该复用缓存，节点数不变）
    auto r2 = sess.run(g, {y}, {{x, x_val}});
    std::cout << "第二次 run 后节点数: " << g.nodes().size() << std::endl;

    // 打印所有节点
    std::cout << "\n节点列表:" << std::endl;
    for (const auto& un : g.nodes()) {
        Node* n = un.get();
        std::cout << "  " << n->name;
        if (n->type == SEND) std::cout << " (SEND)";
        if (n->type == RECV) std::cout << " (RECV)";
        std::cout << std::endl;
    }

    return 0;
}
