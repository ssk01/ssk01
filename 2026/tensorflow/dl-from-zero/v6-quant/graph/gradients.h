#pragma once
#include <deque>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include "graph.h"

namespace lf {

// 独立的梯度子图 —— 对应 TF gradients.py 的 BackpropBuilder。
//
// 遍历 forward 图(loss 反向可达集), 为每个 op 的每个输入生成"梯度节点",
// 累加进 grad_map(forward 节点 -> 它的梯度节点)。
// 梯度节点和正向节点在**同一张图**里, 所以:
//   - 训练时 fetch sgd_step(它消费 grad_of[var]), 梯度子图自然可达、随正向一起执行
//   - 推理时 prune 到 y_pred, 梯度子图/loss/sgd_step 不可达, 被自动剪掉
//   - 不需要之前那套"独立反向遍历"(backward 函数)了
class GradientBuilder {
public:
    GradientBuilder(Graph& g, Node* loss) : g_(g), loss_(loss) {}

    void build() {
        // 1) forward 可达集 + 拓扑序 (上游在前)
        std::unordered_set<Node*> fwd;
        std::vector<Node*> topo;
        {
            std::vector<Node*> queue = {loss_};
            fwd.insert(loss_);
            for (size_t i = 0; i < queue.size(); i++)
                for (Node* in : queue[i]->inputs)
                    if (fwd.insert(in).second) queue.push_back(in);

            std::unordered_set<Node*> visited;
            std::function<void(Node*)> dfs = [&](Node* n) {
                if (!visited.insert(n).second) return;
                for (Node* in : n->inputs)
                    if (fwd.count(in)) dfs(in);
                topo.push_back(n);
            };
            for (Node* n : fwd) dfs(n);
        }

        // 2) 种子: dL/dL = 1
        grad_map_[loss_] = g_.constant("grad_seed", Tensor(1.0f));

        // 3) 逆拓扑: 每个节点把已累加的梯度按链式法则分摊给它的输入
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            Node* n = *it;
            Node* g = grad_map_.at(n);   // 消费者(下游)已把本节点的梯度累加好
            switch (n->type) {
            case ADD:
                add_contrib(n->inputs[0], g);
                add_contrib(n->inputs[1], g);
                break;
            case MUL:
                add_contrib(n->inputs[0], g_.mul(g, n->inputs[1]));
                add_contrib(n->inputs[1], g_.mul(g, n->inputs[0]));
                break;
            case SUB:
                add_contrib(n->inputs[0], g);
                add_contrib(n->inputs[1], g_.mul(g, g_.constant("c1", Tensor(-1.0f))));
                break;
            case SQUARE:
                add_contrib(n->inputs[0],
                            g_.mul(g_.mul(g, g_.constant("c2", Tensor(2.0f))), n->inputs[0]));
                break;
            case MEAN:
                add_contrib(n->inputs[0], g_.mean_grad(g, n->inputs[0]));
                break;
            case SIGMOID:
                // d/dx sigmoid(x) = p(1-p); p 是 sigmoid 自己的输出节点
                add_contrib(n->inputs[0],
                            g_.mul(g_.mul(g, n),
                                   g_.sub(g_.constant("c1", Tensor(1.0f)), n)));
                break;
            case LOG:
                add_contrib(n->inputs[0], g_.mul(g, g_.recip(n->inputs[0])));
                break;
            case MATMUL:
                add_contrib(n->inputs[0], g_.matmul_grad_a(g, n->inputs[1]));
                add_contrib(n->inputs[1], g_.matmul_grad_b(g, n->inputs[0]));
                break;
            case FAKE_QUANT:
                // STE (straight-through estimator): 舍入的梯度近似为恒等 —— 梯度
                // 原样传给输入, [min,max] 外的截断由 FAKE_QUANT_GRAD kernel 做
                // (对应 TF gradients.py 给 FakeQuant 注册的假量化梯度)
                add_contrib(n->inputs[0], g_.fake_quant_grad(g, n->inputs[0], n));
                break;
            default:
                break;  // PLACEHOLDER/VARIABLE/CONST/SGD_STEP/梯度 kernel: 无反向路径
            }
        }
    }

    // forward 节点 -> 它的梯度节点 (sgd_step 的梯度输入)
    Node* grad_of(Node* n) const {
        auto it = grad_map_.find(n);
        return it == grad_map_.end() ? nullptr : it->second;
    }

private:
    // 把 contrib 累加进 in 的梯度: 多元链式法则, 多条路径求和(ADD 链)
    void add_contrib(Node* in, Node* contrib) {
        // 标量输入(标量变量/占位符/常量)会被广播到整批: 它的梯度要把整批求和成标量
        // (对应 TF 梯度里 shape-match 的 reduce-sum, 旧实现里由 tensor_add_to 隐式完成)
        bool scalar_in = (in->type == VARIABLE || in->type == PLACEHOLDER || in->type == CONST)
                         && in->init.shape.empty();
        if (scalar_in) contrib = g_.reduce_sum(contrib);

        auto it = grad_map_.find(in);
        if (it == grad_map_.end())
            grad_map_[in] = contrib;
        else
            grad_map_[in] = g_.add(it->second, contrib);
    }

    Graph& g_;
    Node* loss_;
    std::unordered_map<Node*, Node*> grad_map_;
};

// 构建并返回 forward 节点 -> 梯度节点 的映射
inline std::unordered_map<Node*, Node*> build_gradients(Graph& g, Node* loss) {
    GradientBuilder b(g, loss);
    b.build();
    std::unordered_map<Node*, Node*> out;
    for (auto& un : g.nodes()) {
        Node* n = un.get();
        if (Node* gn = b.grad_of(n)) out[n] = gn;
    }
    return out;
}

}  // namespace lf
