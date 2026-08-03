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

struct Node {
    NodeType type;
    std::string name;
    std::vector<Node*> inputs;
    Tensor output;
    Tensor grad;

    Node(NodeType t, const std::string& n, std::vector<Node*> in)
        : type(t), name(n), inputs(std::move(in)) {}

    Node(NodeType t, const std::string& n, std::vector<Node*> in, float scalar)
        : type(t), name(n), inputs(std::move(in)), output(scalar) {}
};

class Graph {
public:
    Graph() = default;

    Node* placeholder(const std::string& name, const std::vector<int>& shape) {
        auto n = std::make_unique<Node>(PLACEHOLDER, name, std::vector<Node*>{});
        n->output = Tensor(shape);
        Node* ptr = n.get();
        nodes_.push_back(std::move(n));
        return ptr;
    }

    Node* variable(const std::string& name, float init) {
        auto n = std::make_unique<Node>(VARIABLE, name, std::vector<Node*>{}, init);
        Node* ptr = n.get();
        nodes_.push_back(std::move(n));
        return ptr;
    }

    // 1-D 向量变量, shape=[n], 全填 init
    Node* variable_vec(const std::string& name, int n, float init) {
        auto v = std::make_unique<Node>(VARIABLE, name, std::vector<Node*>{}, init);
        v->output = Tensor(std::vector<int>{n});
        std::fill(v->output.data.begin(), v->output.data.end(), init);
        Node* ptr = v.get();
        nodes_.push_back(std::move(v));
        return ptr;
    }

    Node* add(Node* a, Node* b) {
        auto n = std::make_unique<Node>(ADD, "add", std::vector<Node*>{a, b});
        Node* ptr = n.get();
        nodes_.push_back(std::move(n));
        return ptr;
    }

    Node* mul(Node* a, Node* b) {
        auto n = std::make_unique<Node>(MUL, "mul", std::vector<Node*>{a, b});
        Node* ptr = n.get();
        nodes_.push_back(std::move(n));
        return ptr;
    }

    Node* sub(Node* a, Node* b) {
        auto n = std::make_unique<Node>(SUB, "sub", std::vector<Node*>{a, b});
        Node* ptr = n.get();
        nodes_.push_back(std::move(n));
        return ptr;
    }

    Node* square(Node* a) {
        auto n = std::make_unique<Node>(SQUARE, "square", std::vector<Node*>{a});
        Node* ptr = n.get();
        nodes_.push_back(std::move(n));
        return ptr;
    }

    Node* mean(Node* a) {
        auto n = std::make_unique<Node>(MEAN, "mean", std::vector<Node*>{a});
        Node* ptr = n.get();
        nodes_.push_back(std::move(n));
        return ptr;
    }

    // a: [N, F], b: [F]  ->  [N]
    Node* matmul(Node* a, Node* b) {
        auto n = std::make_unique<Node>(MATMUL, "matmul", std::vector<Node*>{a, b});
        Node* ptr = n.get();
        nodes_.push_back(std::move(n));
        return ptr;
    }

    Node* sigmoid(Node* a) {
        auto n = std::make_unique<Node>(SIGMOID, "sigmoid", std::vector<Node*>{a});
        Node* ptr = n.get();
        nodes_.push_back(std::move(n));
        return ptr;
    }

    Node* log(Node* a) {
        auto n = std::make_unique<Node>(LOG, "log", std::vector<Node*>{a});
        Node* ptr = n.get();
        nodes_.push_back(std::move(n));
        return ptr;
    }

    Node* sgd_step(Node* var, Node* grad, float lr) {
        auto n = std::make_unique<Node>(SGD_STEP, "sgd",
                                        std::vector<Node*>{var, grad}, lr);
        Node* ptr = n.get();
        nodes_.push_back(std::move(n));
        return ptr;
    }

    const std::vector<std::unique_ptr<Node>>& nodes() const { return nodes_; }
    void clear() { nodes_.clear(); }

    // 只保留 keep 集合中的节点,其余全部删除(保序)
    // 对应 TF 的 RemoveNode/PruneForReverseReachability 的删除环节
    void keep_only(const std::unordered_set<Node*>& keep) {
        std::vector<std::unique_ptr<Node>> survivors;
        survivors.reserve(keep.size());
        for (auto& n : nodes_) {
            if (keep.count(n.get())) survivors.push_back(std::move(n));
        }
        nodes_.swap(survivors);
    }

private:
    std::vector<std::unique_ptr<Node>> nodes_;
};

}  // namespace lf
