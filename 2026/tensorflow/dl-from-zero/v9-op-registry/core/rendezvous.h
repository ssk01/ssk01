#pragma once
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <string>
#include "../framework/tensor.h"

namespace lf {

// Rendezvous — 跨设备数据传输的配对机制 (v8)
// 对应 TF commit 1 的 IntraProcessRendezvous: Send/Recv 通过 key 配对,
// 可能以任意顺序到达 (Send 先到 → 缓存 tensor; Recv 先到 → 阻塞等 Send 唤醒)。
//
// 进程内实现: mutex + map + cv。跨进程实现 (未做) 会把 Send 序列化通过
// TCP/RDMA 发送, Recv 反序列化 —— 但接口相同, 这就是 v2 PS transport 的本质。
class Rendezvous {
public:
    // Send: 把 tensor 放进 rendezvous, key 配对。非阻塞。
    // 如果 Recv 已在等 → 直接交付 + 唤醒; 否则缓存进 pending_sends_。
    void Send(const std::string& key, const Tensor& tensor) {
        std::unique_lock<std::mutex> lock(mu_);

        // 检查是否有 Recv 在等待这个 key
        auto recv_it = pending_recvs_.find(key);
        if (recv_it != pending_recvs_.end()) {
            // Recv 先到 → 直接交付
            recv_it->second.tensor = tensor;
            recv_it->second.ready = true;
            recv_it->second.cv->notify_one();
        } else {
            // Recv 还没来 → 缓存
            pending_sends_[key] = tensor;
        }
    }

    // Recv: 从 rendezvous 取出 tensor, 阻塞等待 Send。
    // 如果 Send 已到 → 直接取; 否则在 pending_recvs_ 里等 Send 唤醒。
    Tensor Recv(const std::string& key) {
        std::unique_lock<std::mutex> lock(mu_);

        // 检查 Send 是否已到
        auto send_it = pending_sends_.find(key);
        if (send_it != pending_sends_.end()) {
            // Send 先到 → 直接取
            Tensor result = send_it->second;
            pending_sends_.erase(send_it);
            return result;
        }

        // Send 还没来 → 阻塞等待
        RecvWaiter waiter;
        std::condition_variable cv;
        waiter.cv = &cv;
        waiter.ready = false;
        pending_recvs_[key] = waiter;

        // 等待 Send 唤醒
        cv.wait(lock, [&waiter] { return waiter.ready; });

        Tensor result = waiter.tensor;
        pending_recvs_.erase(key);
        return result;
    }

private:
    struct RecvWaiter {
        std::condition_variable* cv;
        Tensor tensor;
        bool ready;
    };

    std::mutex mu_;
    std::unordered_map<std::string, Tensor> pending_sends_;      // Send 已到, Recv 未到
    std::unordered_map<std::string, RecvWaiter> pending_recvs_;  // Recv 已到, Send 未到
};

}  // namespace lf
