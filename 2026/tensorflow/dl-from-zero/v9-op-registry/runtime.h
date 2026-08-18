#pragma once
#include <vector>
#include "framework/tensor.h"
#include "graph/graph.h"
#include "core/rendezvous.h"  // v8

namespace lf {

// 每次 run 的临时值槽 —— 对应 TF ExecutorState 里的 Entry。
// 用**稠密数组按 node->id 索引** (对应 TF: 编译期定槽 + 运行时纯下标访问),
// 而不是 unordered_map —— O(1) 且无哈希开销、cache 友好。
struct RunState {
    // Session 持有的持久变量表 (对应 TF VariableOp 的 Var buffer), 只读访问
    const std::unordered_map<const Node*, Tensor>* vars = nullptr;

    std::vector<Tensor> outputs;          // 索引 = node->id
    std::vector<unsigned char> has;       // 该槽位本轮是否被写入 (对应 TF Entry.has_value)

    Rendezvous* rendezvous = nullptr;     // v8: 跨设备通信 (Send/Recv 通过它配对)

    void resize(int n) {
        outputs.resize(n);
        has.assign(n, 0);
    }
    // 写: 记 has
    Tensor& out(const Node* n) {
        has[n->id] = 1;
        return outputs[n->id];
    }
    // 读
    const Tensor& out(const Node* n) const { return outputs[n->id]; }
    bool computed(const Node* n) const {
        return n->id < static_cast<int>(has.size()) && has[n->id];
    }
};

}  // namespace lf
