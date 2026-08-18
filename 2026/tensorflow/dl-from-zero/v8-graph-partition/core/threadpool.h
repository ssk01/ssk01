#pragma once
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace lf {

// 固定大小 worker 池 —— 对应 commit 1 的 lib/core/threadpool.{h,cc}。
// commit 1 是手写的 (mutex + std::deque + waiter 栈 CV, nullptr sentinel 关闭),
// 这里同样手写, 不用 std::async: 池线程持久, 任务分派零线程创建开销。
class ThreadPool {
public:
    explicit ThreadPool(int n) {
        for (int i = 0; i < n; i++)
            workers_.emplace_back([this] { WorkerLoop(); });
    }
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> l(mu_);
            closing_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int size() const { return static_cast<int>(workers_.size()); }

    void Schedule(std::function<void()> fn) {
        {
            std::lock_guard<std::mutex> l(mu_);
            queue_.push_back(std::move(fn));
        }
        cv_.notify_one();
    }

private:
    void WorkerLoop() {
        while (true) {
            std::function<void()> fn;
            {
                std::unique_lock<std::mutex> l(mu_);
                cv_.wait(l, [&] { return closing_ || !queue_.empty(); });
                if (queue_.empty()) return;  // closing_
                fn = std::move(queue_.front());
                queue_.pop_front();
            }
            fn();
        }
    }

    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::function<void()>> queue_;
    std::vector<std::thread> workers_;
    bool closing_ = false;
};

}  // namespace lf
