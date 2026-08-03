#pragma once
#include <unordered_map>
#include "framework/tensor.h"
#include "graph/graph.h"

namespace lf {

// 每次 run 的临时值槽 —— 对应 TF ExecutorState 里的 Entry (executor.cc:343)
// 图节点本身是纯静态的, 所有运行期的值都存在这里:
//   outputs : 每个节点本轮的输出
//   grads   : 每个节点本轮的梯度
// 因为状态跟 run 走而不是跟图走, 同一张图可以被多个线程并发跑。
struct RunState {
    // Session 持有的持久变量表 (对应 TF VariableOp 的 Var buffer), 只读访问
    const std::unordered_map<const Node*, Tensor>* vars = nullptr;

    std::unordered_map<const Node*, Tensor> outputs;
    std::unordered_map<const Node*, Tensor> grads;

    const Tensor& out(const Node* n) const { return outputs.at(n); }
    Tensor& out(const Node* n) { return outputs[n]; }
    Tensor& grad(const Node* n) { return grads[n]; }
};

}  // namespace lf
