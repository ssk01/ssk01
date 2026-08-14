#pragma once
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "graph.h"
#include "../kernels/quantize.h"

namespace lf {

// ============================================================
// 量化图变换: matmul → Quantize / QuantizedMatMul / Dequantize
//
// 对应 TF 的 quantization 图变换。2016 年没有自动工具 —— 用户手工拼量化
// ops (把 matmul 换成 QuantizedMatMul, 手动插入 Quantize/Dequantize 边界
// 节点, 提供 min/max); 自动化的 quantize_graph.py 是 2016-06 的 eightbit
// 模式。这里是程序化版本, 对 (已 prune 的) 推理图做同样的事:
//
//   matmul(x, w)  →  dequantize(q_matmul(quantize(x, [amin,amax]),
//                                         const_int8(w, [wmin,wmax])))
//
// 输入:
//   vars : 变量当前值 —— 权重要烘焙成量化常量 (对应部署时的 freeze 语义)
//   calib: 非静态输入 (激活) 的量化范围 —— 对应 TF 早期"用户提供 min/max",
//          由校准数据统计得到 (跑一遍数据看激活范围)
// ============================================================
// wrange (可选, QAT 用): 权重烘焙范围 —— 键是 matmul 的权重输入节点 (可能是
// FAKE_QUANT), 值是训练时冻结的范围。QAT 的模型是按训练范围训出来的, 导出时
// 沿用该范围 (值域统计的 tensor_minmax 只是兜底)。
inline void quantize_inference(
    Graph& g,
    const std::unordered_map<const Node*, Tensor>& vars,
    const std::unordered_map<const Node*, std::pair<float, float>>& calib,
    const std::unordered_map<const Node*, std::pair<float, float>>& wrange = {}) {
    // 先收集 matmul (遍历中改图会失效)
    std::vector<Node*> matmuls;
    for (const auto& un : g.nodes())
        if (un->type == MATMUL) matmuls.push_back(un.get());

    std::unordered_set<Node*> dead_weights;  // 烘焙后无消费者的变量 (freeze 语义)
    for (Node* mm : matmuls) {
        Node* x = mm->inputs[0];  // 激活 (运行时值, 范围来自校准)
        Node* w = mm->inputs[1];  // 权重 (变量/常量, 静态值 → 烘焙)

        // 1. 权重: 静态值直接量化成 int8 常量 (范围从值本身统计)
        //    QAT 图里权重是 fake_quant(variable) → 变量值 (fake_quant 前向是往返,
        //    值不变语义, 直接取原值)
        Node* wvar = w;
        if (w->type == FAKE_QUANT) wvar = w->inputs[0];
        const Tensor& wvals = (wvar->type == VARIABLE) ? vars.at(wvar) : wvar->init;
        auto [wmin, wmax] = tensor_minmax(wvals);
        nudge_range(wmin, wmax);
        if (auto it = wrange.find(w); it != wrange.end()) {
            // QAT: 用训练时冻结的范围 (模型按这个范围训出来的, 见 demo 6)
            wmin = it->second.first;
            wmax = it->second.second;
            nudge_range(wmin, wmax);
        }
        Node* cw = g.constant(w->name + "_q", quantize_tensor(wvals, wmin, wmax));
        cw->qmin = wmin;
        cw->qmax = wmax;

        // 2. 激活: 运行时量化 (范围来自校准, 对应 TF 用户提供的 min/max)
        auto it = calib.find(x);
        if (it == calib.end()) {
            std::cerr << "quantize_inference: 缺少输入 " << x->name
                      << " 的校准范围 (校准数据里没统计到这个节点)"
                      << std::endl;
            std::abort();
        }
        float amin = it->second.first, amax = it->second.second;
        nudge_range(amin, amax);
        Node* qx = g.quantize(x, amin, amax);

        // 3. 量化 matmul + 反量化 (输出范围由 q_matmul 构造器按输入范围算)
        Node* qm = g.q_matmul(qx, cw);
        Node* dq = g.dequantize(qm);

        // 4. 重连: matmul 的全部消费边改指 dequantize 输出, 然后删 matmul
        for (const auto& un : g.nodes()) {
            Node* c = un.get();
            if (c == mm) continue;
            for (auto& in : c->inputs)
                if (in == mm) in = dq;
        }
        g.remove_node(mm);
        // QAT: 权重链是 variable → fake_quant → matmul, 两个节点都可能变孤儿
        if (wvar->type == VARIABLE) {
            dead_weights.insert(wvar);
            if (w != wvar) dead_weights.insert(w);
        }
    }

    // 权重已烘焙进常量、不再被消费 → 删掉 (对应量化部署的 freeze: 图里只剩
    // 静态量化权重, 不再有变量)
    for (Node* w : dead_weights) {
        bool used = false;
        for (const auto& un : g.nodes())
            for (const Node* in : un->inputs)
                if (in == w) { used = true; break; }
        if (!used) g.remove_node(w);
    }
}

}  // namespace lf
