#pragma once
#include <vector>
#include <string>
#include <memory>
#include <unordered_set>
#include "../framework/tensor.h"

namespace lf {

enum NodeType {
    PLACEHOLDER,
    VARIABLE,
    ADD,
    MUL,
    SUB,
    SQUARE,
    MEAN,
    MATMUL,
    SIGMOID,
    LOG,
    SGD_STEP,
};

// 纯静态的图节点: 只描述"算什么", 不存任何运行期的值 (对应 TF 的 Graph Node)
// 运行期的值都在每次 run 的 RunState 里; 变量持久值在 Session 里。
struct Node {
    NodeType type;
    std::string name;
    std::vector<Node*> inputs;
    float scalar = 0.0f;   // 静态属性, 如 SGD_STEP 的学习率
    Tensor init;           // 变量初始值 (静态元数据, 如 TF 的 initial_value)

    Node(NodeType t, const std::string& n, std::vector<Node*> in)
        : type(t), name(n), inputs(std::move(in)) {}

    Node(NodeType t, const std::string& n, std::vector<Node*> in, float s)
        : type(t), name(n), inputs(std::move(in)), scalar(s) {}
};

class Graph {
public:
    Graph() = default;

    Node* placeholder(const std::string& name, const std::vector<int>& shape) {
        auto n = std::make_unique<Node>(PLACEHOLDER, name, std::vector<Node*>{});
        n->init = Tensor(shape);
        Node* ptr = n.get();
        nodes_.push_back(std::move(n));
        return ptr;
    }

    Node* variable(const std::string& name, float init) {
        auto n = std::make_unique<Node>(VARIABLE, name, std::vector<Node*>{}, init);
        n->init = Tensor(init);
        Node* ptr = n.get();
        nodes_.push_back(std::move(n));
        return ptr;
    }

    // 1-D 向量变量, shape=[n], 全填 init
    Node* variable_vec(const std::string& name, int n, float init) {
        auto v = std::make_unique<Node>(VARIABLE, name, std::vector<Node*>{}, init);
        v->init = Tensor(std::vector<int>{n});
        std::fill(v->init.data.begin(), v->init.data.end(), init);
        Node* ptr = v.get();
        nodes_.push_back(std::move(v));
        return ptr;
    }

    Node* add(Node* a, Node* b) {
        return make(ADD, "add", {a, b});
    }
    Node* mul(Node* a, Node* b) {
        return make(MUL, "mul", {a, b});
    }
    Node* sub(Node* a, Node* b) {
        return make(SUB, "sub", {a, b});
    }
    Node* square(Node* a) {
        return make(SQUARE, "square", {a});
    }
    Node* mean(Node* a) {
        return make(MEAN, "mean", {a});
    }
    // a: [N, F], b: [F]  ->  [N]
    Node* matmul(Node* a, Node* b) {
        return make(MATMUL, "matmul", {a, b});
    }
    Node* sigmoid(Node* a) {
        return make(SIGMOID, "sigmoid", {a});
    }
    Node* log(Node* a) {
        return make(LOG, "log", {a});
    }

    // 优化器节点 (对应 TF 的 ApplyGradientDescent): backward 后把 lr*grad 应用到 var
    Node* sgd_step(Node* var, float lr) {
        return make(SGD_STEP, "sgd", {var}, lr);
    }

    const std::vector<std::unique_ptr<Node>>& nodes() const { return nodes_; }
    void clear() { nodes_.clear(); }

    // 只保留 keep 集合中的节点,其余全部删除(保序)
    void keep_only(const std::unordered_set<Node*>& keep) {
        std::vector<std::unique_ptr<Node>> survivors;
        survivors.reserve(keep.size());
        for (auto& n : nodes_) {
            if (keep.count(n.get())) survivors.push_back(std::move(n));
        }
        nodes_.swap(survivors);
    }

private:
    Node* make(NodeType t, const std::string& name, std::vector<Node*> in,
               float scalar = 0.0f) {
        auto n = std::make_unique<Node>(t, name, std::move(in), scalar);
        Node* ptr = n.get();
        nodes_.push_back(std::move(n));
        return ptr;
    }

    std::vector<std::unique_ptr<Node>> nodes_;
};

}  // namespace lf
