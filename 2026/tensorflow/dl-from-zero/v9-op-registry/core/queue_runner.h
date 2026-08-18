#pragma once
#include <thread>
#include <vector>
#include <atomic>
#include <memory>
#include "../ops/queue_ops.h"
#include "../session.h"

namespace lf {

// QueueRunner: 后台线程不停 run enqueue 子图
// 对应 TF python/training/queue_runner.py
//
// 用法:
//   QueueRunner runner(sess, {enqueue_op1, enqueue_op2}, queue);
//   runner.start();  // 启动后台线程
//   // ... 训练循环消费 dequeue ...
//   runner.stop();   // 停止后台线程
class QueueRunner {
public:
    QueueRunner(Session* sess, const std::vector<Node*>& enqueue_ops,
                FIFOQueue* queue, int num_threads = 1)
        : sess_(sess), enqueue_ops_(enqueue_ops), queue_(queue),
          num_threads_(num_threads), should_stop_(false) {}

    ~QueueRunner() {
        stop();
    }

    // 启动后台线程
    void start() {
        if (!threads_.empty()) return;  // 已启动

        should_stop_ = false;
        for (int i = 0; i < num_threads_; ++i) {
            threads_.emplace_back([this, i]() {
                this->run_loop(i);
            });
        }
    }

    // 停止后台线程
    void stop() {
        should_stop_ = true;
        queue_->Close();  // 关闭队列，唤醒所有阻塞的线程

        for (auto& t : threads_) {
            if (t.joinable()) {
                t.join();
            }
        }
        threads_.clear();
    }

private:
    void run_loop(int thread_id) {
        while (!should_stop_) {
            try {
                // 轮询 enqueue ops
                for (Node* enqueue_op : enqueue_ops_) {
                    if (should_stop_) break;

                    // 简化: 假设 enqueue_op 不需要 feeds
                    // 实际应该传入数据生成逻辑
                    sess_->run(*enqueue_op->inputs[0]->inputs[0],  // 获取图
                              {enqueue_op}, {});
                }
            } catch (const std::exception& e) {
                if (!should_stop_) {
                    // 错误但未停止 → 继续重试
                }
                break;
            }
        }
    }

    Session* sess_;
    std::vector<Node*> enqueue_ops_;
    FIFOQueue* queue_;
    int num_threads_;
    std::atomic<bool> should_stop_;
    std::vector<std::thread> threads_;
};

// Coordinator: 管理多个 QueueRunner 的生命周期
// 对应 TF python/training/coordinator.py
class Coordinator {
public:
    Coordinator() : should_stop_(false) {}

    // 注册 QueueRunner
    void add_runner(std::shared_ptr<QueueRunner> runner) {
        runners_.push_back(runner);
    }

    // 启动所有 runner
    void start_all() {
        for (auto& runner : runners_) {
            runner->start();
        }
    }

    // 停止所有 runner
    void stop_all() {
        should_stop_ = true;
        for (auto& runner : runners_) {
            runner->stop();
        }
    }

    // 请求停止
    void request_stop() {
        should_stop_ = true;
    }

    bool should_stop() const {
        return should_stop_;
    }

private:
    std::vector<std::shared_ptr<QueueRunner>> runners_;
    std::atomic<bool> should_stop_;
};

}  // namespace lf
