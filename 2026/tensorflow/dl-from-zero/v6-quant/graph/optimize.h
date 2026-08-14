#pragma once
#include <unordered_map>
#include <vector>
#include "graph.h"
#include "../kernels/kernels.h"   // forward(): 常量折叠复用运行时求值

namespace lf {

// ============================================================
// CSE (公共子表达式消除)
//
// 对应 TF tensorflow/core/graph/optimizer_cse.cc —— 初始 commit 就有。
// TF 原版算法 (注释里写得很清楚, 抄自 Cliff Click PLDI'95 的全局值编号):
//
//   std::unordered_map<size_t, Node*> available
//   for each node n in forward topological order:
//     h = NodeHash(n)
//     if available[h] exists and Equivalent(available[h], n)
//       redirect downstream uses of n to available[h]; remove n
//     else if available[h] does not exist: available[h] = n
//
// 哈希的是 (type, 输出类型, 输入节点+槽位, attrs); 等价性由 Equivalent() 判定:
//   1. is_stateful() → 永不合并          (有状态节点, 如 Variable/ApplyGradientDescent)
//   2. HasRefInput()  → 永不合并         (输入带 ref 句柄的节点, 值可能被并发改写)
//   3. attrs 相等                        (不同的常量 → 不同的值 → 不合并)
//   4. 输入 (src, src_output) 全等        (必须消费完全相同的输入)
//   5. 交换律 op: 输入按 Node* 排序后比较 (add(a,b) ≡ add(b,a), 对应 FillInputs 的 sort)
//   consider_fn 是 TF 留给调用方的过滤钩子 (master 用它跳过特定设备/控制流节点)。
//
// 我们的对应:
//   - 哈希 (type, 规范化输入 id, scalar, CONST 的 init 值), 再 Equivalent 复核
//   - 有状态 → VARIABLE / SGD_STEP        (对应 1)
//   - 我们没有 ref 句柄: VARIABLE 自带"读"语义, 它的直接消费者等价 TF 里 Read 之后的
//     纯节点 —— TF 允许合并它们 (Read 本身因为 HasRefInput 不合并), 我们也允许
//   - CONST 必须值相同才合并              (对应 3)
//   - 输入指针全等                        (对应 4, 我们只有单输出)
//   - ADD/MUL 交换律规范化                (对应 5)
//   - 额外规则: PLACEHOLDER 不参与合并 —— 我们的 feed 以 Node* 为键, 合并两个
//     placeholder 会破坏 feed 路由; TF 的 feed 以节点名为键, 合并后名字相同,
//     语义安全, 所以 TF 可以合并 placeholder (demo 级差异)
// ============================================================

namespace {

bool is_stateful(NodeType t) { return t == VARIABLE || t == SGD_STEP; }
bool is_commutative(NodeType t) { return t == ADD || t == MUL; }

size_t hash_combine(size_t h, size_t x) { return h * 31 + x; }

// 哈希: (type, 规范化输入 id 序列, scalar, qmin/qmax, CONST 的 init 值)
size_t node_hash(const Node* n) {
    size_t h = std::hash<int>{}(static_cast<int>(n->type));
    std::vector<Node*> ins = n->inputs;
    if (is_commutative(n->type)) {  // 交换律规范化 (对应 TF FillInputs 的 commutative sort)
        std::sort(ins.begin(), ins.end(),
                  [](const Node* a, const Node* b) { return a->id < b->id; });
    }
    for (const Node* in : ins) h = hash_combine(h, static_cast<size_t>(in->id));
    h = hash_combine(h, std::hash<float>{}(n->scalar));
    h = hash_combine(h, std::hash<float>{}(n->qmin));  // 量化范围不同 → 值不同
    h = hash_combine(h, std::hash<float>{}(n->qmax));
    if (n->type == CONST) {
        h = hash_combine(h, n->init.shape.size());
        for (int d : n->init.shape) h = hash_combine(h, static_cast<size_t>(d));
        for (float f : n->init.data) h = hash_combine(h, std::hash<float>{}(f));
    }
    return h;
}

// 等价判定 (对应 TF Equivalent): 哈希碰撞时复核, 防误合并
bool equivalent(const Node* a, const Node* b) {
    if (a->type != b->type) return false;
    if (a->scalar != b->scalar) return false;
    if (a->qmin != b->qmin || a->qmax != b->qmax) return false;
    if (is_commutative(a->type)) {
        auto ins_a = a->inputs, ins_b = b->inputs;
        auto by_id = [](const Node* p, const Node* q) { return p->id < q->id; };
        std::sort(ins_a.begin(), ins_a.end(), by_id);
        std::sort(ins_b.begin(), ins_b.end(), by_id);
        if (ins_a != ins_b) return false;
    } else if (a->inputs != b->inputs) {
        return false;
    }
    // init: CONST 值必须相同 (VARIABLE/PLACEHOLDER 不参与合并, 其余类型 init 为空)
    if (a->init.shape != b->init.shape) return false;
    if (a->init.data != b->init.data) return false;
    return true;
}

}  // namespace

inline void cse(Graph& g) {
    // 遍历序 = 创建序: 构建 API 保证输入先于消费者创建, 创建序即拓扑序
    // (对应 TF 的 GetReversePostOrder, 即"输入先于下游依赖")
    std::unordered_map<size_t, Node*> available;   // 哈希桶 → 首个候选 (对应 TF available)
    std::vector<Node*> removed;
    for (const auto& un : g.nodes()) {
        Node* n = un.get();
        if (is_stateful(n->type)) continue;    // 有状态节点永不合并 (对应 is_stateful)
        if (n->type == PLACEHOLDER) continue;  // 边界节点, feed 以 Node* 为键 (见头注释)
        size_t h = node_hash(n);
        auto it = available.find(h);
        if (it == available.end()) {
            available[h] = n;   // 只保留第一个候选 (对应 TF: 单候选, 永不更新)
        } else if (equivalent(it->second, n)) {
            // 命中: n 的全部消费边改指候选节点, 然后删 n (对应 TF AddEdge + RemoveNode)
            for (const auto& c : g.nodes()) {
                Node* consumer = c.get();
                if (consumer == n) continue;
                for (auto& in : consumer->inputs)
                    if (in == n) in = it->second;
            }
            removed.push_back(n);
        }
    }
    for (Node* n : removed) g.remove_node(n);
}

// ============================================================
// 常量折叠
//
// 对应 TF constant_folding.cc —— 2016-01-25 加入 (commit 71184628900),
// 提交说明原文: "Creates a local executor and executes a copy of the constant
// 'slice' of the original graph, and replaces nodes in original graph with
// constant nodes."
//
// TF 原版做法: 找出"输入全部可折叠"的节点集 (IsConstantFoldable: 非 stateful、
// 非控制流、非 Send/Recv), 拷一份子图, 用本地 executor + rendezvous 求值,
// 然后把原图里的节点删掉、换成新的同名 Const 节点 (ReplaceNodeWithConstant)。
//
// 我们的简化 (语义等价, 但省事且不破坏指针):
//   - 不拷子图、不开 executor: 直接在原图上, 输入全为 CONST 的纯节点用运行时
//     同一个 forward() 求值 (forward 是"这个 op 算什么"的唯一事实来源)
//   - 不删旧建新: 原地把节点改成 CONST (保持节点身份, 用户持有的指针仍有效)
//   - 创建序即拓扑序 → 单遍就能折叠整条链 (上游先折叠成 CONST, 下游当轮可见)
//   - 折叠后清掉"没有消费者"的孤儿常量 (TF 原版不清理, 靠后续 RemoveDeadNodes
//     pass; 我们顺手做掉, 让节点数可读 —— 效果一致)
// ============================================================

namespace {

// 可折叠 = 纯计算节点, 且输入全是 CONST (对应 IsConstantFoldable + 全父折叠检查)
bool foldable(const Node* n) {
    if (n->type == CONST || n->type == PLACEHOLDER || n->type == VARIABLE ||
        n->type == SGD_STEP)
        return false;
    for (const Node* in : n->inputs)
        if (in->type != CONST) return false;
    return true;
}

}  // namespace

inline void const_fold(Graph& g) {
    std::vector<Node*> folded;
    for (const auto& un : g.nodes()) {
        Node* n = un.get();
        if (!foldable(n)) continue;
        // 用运行时同一个 forward() 求值 (对应 TF "执行常量子图")
        RunState st;
        st.resize(g.id_count());
        for (const Node* in : n->inputs) st.out(in) = in->init;
        forward(n, st);
        // 原地转成 CONST (对应 TF ReplaceNodeWithConstant; 我们不换节点身份)
        n->type = CONST;
        n->init = st.out(n);
        n->inputs.clear();
        folded.push_back(n);
    }
    if (folded.empty()) return;

    // 清孤儿: 折叠把输入常量的消费边消掉了, 没人用的常量删掉 (对应 TF RemoveDeadNodes)
    std::vector<Node*> orphans;
    for (const auto& un : g.nodes()) {
        Node* n = un.get();
        if (n->type != CONST) continue;
        bool used = false;
        for (const auto& c : g.nodes()) {
            if (c.get() == n) continue;
            for (const Node* in : c->inputs) {
                if (in == n) { used = true; break; }
            }
            if (used) break;
        }
        if (!used) orphans.push_back(n);
    }
    for (Node* n : orphans) g.remove_node(n);
    g.touch();
}

// 图优化入口: CSE → 常量折叠 → CSE
//   - 折叠产生的新常量可能彼此重复 → 再跑一次 CSE 合并
//   - 对应论文 §5 master 在 step 前的优化阶段; TF 的优化器注册表也是多个 pass
//     反复跑到不动点
inline void optimize(Graph& g) {
    cse(g);
    const_fold(g);
    cse(g);
}

}  // namespace lf
