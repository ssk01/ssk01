#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <atomic>
#include "../framework/tensor.h"

namespace lf {

// FIFOQueue: 先进先出队列 (对应 TF FIFOQueue)
// 核心: enqueue 满则阻塞, dequeue 空则阻塞 → 生产者/消费者解耦
class FIFOQueue {
public:
    explicit FIFOQueue(int capacity) : capacity_(capacity), closed_(false) {}

    // Enqueue: 入队（满则阻塞）
    void Enqueue(const Tensor& t) {
        std::unique_lock<std::mutex> lock(mu_);

        // 等待队列不满或被关闭
        not_full_.wait(lock, [this] { return queue_.size() < capacity_ || closed_; });

        if (closed_) {
            throw std::runtime_error("Queue is closed");
        }

        queue_.push(t);
        not_empty_.notify_one();
    }

    // Dequeue: 出队（空则阻塞）
    Tensor Dequeue() {
        std::unique_lock<std::mutex> lock(mu_);

        // 等待队列非空或被关闭
        not_empty_.wait(lock, [this] { return !queue_.empty() || closed_; });

        if (closed_ && queue_.empty()) {
            throw std::runtime_error("Queue is closed and empty");
        }

        Tensor t = queue_.front();
        queue_.pop();
        not_full_.notify_one();
        return t;
    }

    // 尝试出队（非阻塞，空则返回 false）
    bool TryDequeue(Tensor& out) {
        std::lock_guard<std::mutex> lock(mu_);
        if (queue_.empty()) return false;
        out = queue_.front();
        queue_.pop();
        not_full_.notify_one();
        return true;
    }

    // 关闭队列
    void Close() {
        std::lock_guard<std::mutex> lock(mu_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    // 队列大小
    size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return queue_.size();
    }

    bool is_closed() const {
        std::lock_guard<std::mutex> lock(mu_);
        return closed_;
    }

private:
    size_t capacity_;
    std::queue<Tensor> queue_;
    bool closed_;

    mutable std::mutex mu_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;
};

}  // namespace lf
