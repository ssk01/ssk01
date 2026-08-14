#pragma once
#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <unordered_set>
#include "../framework/tensor.h"
#include "../kernels/quantize.h"   // q_matmul 的输出范围公式 (仅纯公式, 无环)

namespace lf {

enum NodeType {
    PLACEHOLDER,
    VARIABLE,
    CONST,          // 常量张量 (init 即值)
    ADD,
    MUL,
    SUB,
    SQUARE,
    MEAN,
    MATMUL,
    SIGMOID,
    LOG,
    RECIP,          // 1/x (LOG 梯度需要)
    REDUCE_SUM,     // 把张量求和成标量 (标量变量的梯度归约, 对应 TF 的 shape-match reduce)
    SGD_STEP,       // 优化器: inputs=[var, grad_node]
    // ---- 形状相关算子的专用梯度 kernel (对应 TF 的 MeanGrad / MatMulGrad) ----
    MEAN_GRAD,      // inputs=[grad, x] : 把 grad 广播成 x 的形状再除以 size
    MATMUL_GRAD_A,  // inputs=[grad, B] : dX[N,F] = grad[n] * B[f]
    MATMUL_GRAD_B,  // inputs=[grad, A] : dW[F]   = sum_n grad[n] * A[n,f]
    // ---- 量化 ops (对应 TF 2016-04-22 的 QuantizeV2 / QuantizedMatMul / Dequantize) ----
    QUANTIZE,       // inputs=[x]: min/max 量化到 int8, qmin/qmax = 输出范围
    Q_MATMUL,       // inputs=[a,b] (int8): int32 累加矩阵乘, qmin/qmax = 输出范围
    DEQUANTIZE,     // inputs=[q]: 反量化回 float, qmin/qmax = 输入范围
};

// 纯静态的图节点: 只描述"算什么", 不存运行值 (对应 TF 的 Graph Node)
// id 在创建时单调分配, 运行时 RunState 用稠密数组按 id 索引 (对应 TF 的 Executor 槽位)
struct Node {
    NodeType type;
    int id;
    std::string name;
    std::vector<Node*> inputs;
    float scalar = 0.0f;   // 静态属性, 如 SGD_STEP 的学习率
    Tensor init;           // 变量/常量初始值 (静态元数据)
    float qmin = 0.0f, qmax = 0.0f;  // 量化范围 (QUANTIZE/Q_MATMUL/DEQUANTIZE/量化常量)

    Node(NodeType t, int i, const std::string& n, std::vector<Node*> in)
        : type(t), id(i), name(n), inputs(std::move(in)) {}

    Node(NodeType t, int i, const std::string& n, std::vector<Node*> in, float s)
        : type(t), id(i), name(n), inputs(std::move(in)), scalar(s) {}
};

class Graph {
public:
    Graph() = default;

    Node* placeholder(const std::string& name, const std::vector<int>& shape) {
        Node* n = make(PLACEHOLDER, name, {});
        n->init = Tensor(shape);
        return n;
    }

    Node* variable(const std::string& name, float init) {
        Node* n = make(VARIABLE, name, {}, init);
        n->init = Tensor(init);
        return n;
    }

    Node* variable_vec(const std::string& name, int n, float init) {
        Node* v = make(VARIABLE, name, {}, init);
        v->init = Tensor(std::vector<int>{n});
        std::fill(v->init.data.begin(), v->init.data.end(), init);
        return v;
    }

    Node* constant(const std::string& name, const Tensor& value) {
        Node* n = make(CONST, name, {});
        n->init = value;
        return n;
    }

    Node* add(Node* a, Node* b) { return make(ADD, "add", {a, b}); }
    Node* mul(Node* a, Node* b) { return make(MUL, "mul", {a, b}); }
    Node* sub(Node* a, Node* b) { return make(SUB, "sub", {a, b}); }
    Node* square(Node* a) { return make(SQUARE, "square", {a}); }
    Node* mean(Node* a) { return make(MEAN, "mean", {a}); }
    Node* matmul(Node* a, Node* b) { return make(MATMUL, "matmul", {a, b}); }
    Node* sigmoid(Node* a) { return make(SIGMOID, "sigmoid", {a}); }
    Node* log(Node* a) { return make(LOG, "log", {a}); }
    Node* recip(Node* a) { return make(RECIP, "recip", {a}); }
    Node* reduce_sum(Node* a) { return make(REDUCE_SUM, "reduce_sum", {a}); }

    // 梯度专用 kernel 节点 (用户一般不直接建, 由 build_gradients 生成)
    Node* mean_grad(Node* grad, Node* x) { return make(MEAN_GRAD, "mean_grad", {grad, x}); }
    Node* matmul_grad_a(Node* grad, Node* b) { return make(MATMUL_GRAD_A, "matmul_grad_a", {grad, b}); }
    Node* matmul_grad_b(Node* grad, Node* a) { return make(MATMUL_GRAD_B, "matmul_grad_b", {grad, a}); }

    // 优化器节点 (对应 TF ApplyGradientDescent): inputs=[var, grad_node], 用 lr 更新 var
    Node* sgd_step(Node* var, Node* grad, float lr) {
        return make(SGD_STEP, "sgd", {var, grad}, lr);
    }

    // ---- 量化 ops (对应 TF 2016-04-22 的 QuantizeV2 / QuantizedMatMul / Dequantize) ----
    // QUANTIZE: 输入 min/max 量化到 int8; qmin/qmax = 输出范围
    Node* quantize(Node* x, float min, float max) {
        Node* n = make(QUANTIZE, "quantize", {x});
        n->qmin = min;
        n->qmax = max;
        return n;
    }

    // Q_MATMUL: int8×int8 → int32; 输入必须是量化节点 (带 qmin/qmax),
    // 输出范围 = QuantizationRangeForMultiplication (quantization_utils.h)
    Node* q_matmul(Node* a, Node* b) {
        Node* n = make(Q_MATMUL, "q_matmul", {a, b});
        auto [rmin, rmax] = quantized_matmul_range(a->qmin, a->qmax, b->qmin, b->qmax);
        n->qmin = rmin;
        n->qmax = rmax;
        return n;
    }

    // DEQUANTIZE: 反量化回 float; 范围取自输入节点
    Node* dequantize(Node* q) {
        Node* n = make(DEQUANTIZE, "dequantize", {q});
        n->qmin = q->qmin;
        n->qmax = q->qmax;
        return n;
    }

    const std::vector<std::unique_ptr<Node>>& nodes() const { return nodes_; }
    // 已分配的节点总数 (id 上限), 用于给 RunState 的稠密数组定大小
    int id_count() const { return next_id_; }

    // 结构代数 (每次图被变换 +1): Session 用它判断编译缓存是否失效
    // (对应 TF 的版本化 graph; 图一变, executor 缓存就得重算)
    int generation() const { return generation_; }
    void touch() { ++generation_; }

    void keep_only(const std::unordered_set<Node*>& keep) {
        std::vector<std::unique_ptr<Node>> survivors;
        survivors.reserve(keep.size());
        for (auto& n : nodes_) {
            if (keep.count(n.get())) survivors.push_back(std::move(n));
        }
        nodes_.swap(survivors);
        ++generation_;
    }

    // 删除节点 (CSE 合并后调用)。注意: 调用方必须先保证没有其他节点引用它
    // (CSE 会把它的消费边全部重连到候选节点)。
    void remove_node(Node* n) {
        nodes_.erase(std::remove_if(nodes_.begin(), nodes_.end(),
                                    [n](const std::unique_ptr<Node>& p) {
                                        return p.get() == n;
                                    }),
                     nodes_.end());
        ++generation_;
    }

private:
    Node* make(NodeType t, const std::string& name, std::vector<Node*> in,
               float scalar = 0.0f) {
        auto n = std::make_unique<Node>(t, next_id_++, name, std::move(in), scalar);
        Node* ptr = n.get();
        nodes_.push_back(std::move(n));
        return ptr;
    }

    std::vector<std::unique_ptr<Node>> nodes_;
    int next_id_ = 0;
    int generation_ = 0;
};

}  // namespace lf
