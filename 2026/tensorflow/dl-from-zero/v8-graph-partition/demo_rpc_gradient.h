#pragma once
#include "graph/graph.h"
#include "graph/gradients.h"
#include "framework/tensor.h"
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>

namespace lf {

// RPC Rendezvous: 模拟分布式训练中的跨机器通信
// 场景：两台机器各有一个 worker，通过网络传输 tensor
//
// 使用场景：
//   Machine 0: 运行前向子图 (数据并行的一个副本)
//   Machine 1: 运行后向子图 (参数服务器或另一个副本)
//   Send/Recv 跨机器传输，梯度需要反向的 Send/Recv
class RPCRendezvous {
public:
    RPCRendezvous() = default;

    // 发送 tensor（模拟网络序列化 + 传输）
    void Send(const std::string& key, const Tensor& tensor) {
        std::unique_lock<std::mutex> lock(mu_);

        // 模拟网络延迟
        std::this_thread::sleep_for(std::chrono::microseconds(100));

        // "序列化"并存储
        store_[key] = tensor;

        std::cout << "  [RPC Send] key=" << key
                  << " size=" << tensor.size()
                  << " (网络传输)" << std::endl;

        cv_.notify_all();
    }

    // 接收 tensor（阻塞直到数据到达）
    Tensor Recv(const std::string& key) {
        std::unique_lock<std::mutex> lock(mu_);

        // 等待数据到达
        cv_.wait(lock, [&]() { return store_.count(key) > 0; });

        Tensor result = store_[key];
        store_.erase(key);

        std::cout << "  [RPC Recv] key=" << key
                  << " size=" << result.size()
                  << " (网络接收)" << std::endl;

        return result;
    }

    // 检查 key 是否已经到达
    bool HasData(const std::string& key) const {
        std::lock_guard<std::mutex> lock(mu_);
        return store_.count(key) > 0;
    }

private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::unordered_map<std::string, Tensor> store_;
};

// 手动构建带 Send/Recv 的图（模拟已分区的跨机器图）
// 然后在这个图上构建梯度 → 触发 Send/Recv 的梯度处理
inline void demo_rpc_sendrecv_gradient() {
    std::cout << "\n=== RPC Send/Recv 梯度传播 Demo ===" << std::endl;
    std::cout << "场景：两台机器，Machine 0 发送激活到 Machine 1，需要反向传梯度\n" << std::endl;

    // Machine 0 的图（前向部分）
    Graph g0;
    Node* x = g0.placeholder("x", {100});
    Node* w1 = g0.variable("w1", 1.0f);
    Node* act = g0.mul(x, w1);  // 激活

    // 手动插入 Send（发送到 Machine 1）
    Node* send_act = g0.send("send_act", act);
    send_act->rendezvous_key = "act_0_to_1";

    std::cout << "[Machine 0] 前向图: x → mul(x, w1) → send_act" << std::endl;
    std::cout << "  节点数: " << g0.nodes().size() << std::endl;

    // Machine 1 的图（后向部分）
    Graph g1;
    Node* recv_act = g1.recv("recv_act");
    recv_act->rendezvous_key = "act_0_to_1";

    Node* w2 = g1.variable("w2", 2.0f);
    Node* output = g1.mul(recv_act, w2);
    Node* y_true = g1.placeholder("y", {100});
    Node* diff = g1.sub(output, y_true);
    Node* loss = g1.mean(g1.square(diff));

    std::cout << "\n[Machine 1] 前向图: recv_act → mul(recv_act, w2) → loss" << std::endl;
    std::cout << "  节点数: " << g1.nodes().size() << std::endl;

    // === 关键：在已有 Send/Recv 的图上构建梯度 ===
    std::cout << "\n[构建梯度] 在 Machine 1 的图上构建梯度..." << std::endl;
    auto grads1 = build_gradients(g1, loss);

    std::cout << "  梯度图节点数: " << g1.nodes().size() << std::endl;
    std::cout << "  recv_act 的梯度: "
              << (grads1.count(recv_act) ? grads1[recv_act]->name : "null") << std::endl;

    // 打印所有新增的 Send/Recv 节点（梯度相关）
    std::cout << "\n[检查] 梯度图中的 Send/Recv 节点:" << std::endl;
    bool found_grad_send = false;
    for (const auto& un : g1.nodes()) {
        Node* n = un.get();
        if (n->type == SEND || n->type == RECV) {
            const char* type_str = (n->type == SEND) ? "SEND" : "RECV";
            std::cout << "  " << n->name << " (type=" << type_str << ")";
            if (!n->rendezvous_key.empty()) {
                std::cout << " key=" << n->rendezvous_key;
            }
            if (!n->inputs.empty()) {
                std::cout << " <- " << n->inputs[0]->name;
            }
            std::cout << std::endl;

            if (n->type == SEND && n->rendezvous_key.find("_grad") != std::string::npos) {
                found_grad_send = true;
            }
        }
    }

    if (!found_grad_send) {
        std::cout << "  ❌ 没有找到梯度相关的 SEND 节点（key 应包含 '_grad'）" << std::endl;
    }

    // 检查 recv_act 的梯度节点是否是 SEND（反向传输）
    if (grads1.count(recv_act)) {
        Node* grad_recv_act = grads1[recv_act];
        std::cout << "\n[验证] recv_act 的梯度节点:" << std::endl;
        std::cout << "  节点名: " << grad_recv_act->name << std::endl;
        std::cout << "  节点类型: " << grad_recv_act->type
                  << (grad_recv_act->type == SEND ? " (SEND ✓)" : " (不是 SEND ✗)") << std::endl;

        if (grad_recv_act->type == SEND) {
            std::cout << "  rendezvous_key: " << grad_recv_act->rendezvous_key << std::endl;
            std::cout << "\n✓ 梯度正确生成了反向的 SEND 节点（从 Machine 1 发回 Machine 0）" << std::endl;
        } else {
            std::cout << "\n✗ 梯度没有生成 SEND 节点，无法反向传播到 Machine 0" << std::endl;
        }
    }

    std::cout << "\n=== Demo 说明 ===" << std::endl;
    std::cout << "当前实现：gradients.h 中的 RECV case 会被跳过（因为 inputs[0] 通常不是 SEND）" << std::endl;
    std::cout << "正确实现：RECV 的梯度应该是一个新的 SEND 节点，发送到反向的 rendezvous key" << std::endl;
    std::cout << "         例如：正向 key=\"act_0_to_1\"，反向 key=\"grad_act_1_to_0\"" << std::endl;
}

}  // namespace lf
