#pragma once
#include <vector>
#include <string>
#include <memory>
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

    Node* sgd_step(Node* var, Node* grad, float lr) {
        auto n = std::make_unique<Node>(SGD_STEP, "sgd",
                                        std::vector<Node*>{var, grad}, lr);
        Node* ptr = n.get();
        nodes_.push_back(std::move(n));
        return ptr;
    }

    const std::vector<std::unique_ptr<Node>>& nodes() const { return nodes_; }
    void clear() { nodes_.clear(); }

private:
    std::vector<std::unique_ptr<Node>> nodes_;
};

}  // namespace lf
