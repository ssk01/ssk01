#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>
#include "../graph/graph.h"
#include "../kernels/kernels.h"
#include "../runtime.h"
#include "place.h"
#include "threadpool.h"

namespace lf {

// 一次编译的图执行计划 (Session 按 (图, 目标集) 缓存, 对应 commit 1 的
// Executor 编译产物: 就绪队列 + pending 计数表)。
struct GraphPlan {
    std::vector<Node*> order;                 // 拓扑序 (targets 可达集, 上游在前)
    std::vector<Node*> by_id;                 // 稠密 id → 节点 (空槽 nullptr)
    std::vector<std::vector<int>> consumers;  // 稠密 id → 图内消费它的节点 id 列表
    std::vector<int> in_degree;               // 图内入度 (pending 的初值)
    std::vector<Device> devices;              // 稠密 id → 放置的设备 (placer 产物)
};

// 一次 run 的执行状态 (对应 commit 1 的 ExecutorState)
struct ExecState {
    RunState* st = nullptr;                   // 值槽 (输出 + vars 指针)
    std::vector<int> pending;                 // pending[id] = 剩余未满足的入边数
    std::atomic<int> outstanding{0};          // 未完成 op 数 (commit 1 num_outstanding_ops_)
    std::mutex mu;                            // 保护 ready_ / finished_ / active_
    std::condition_variable cv;               // worker 的等待 (ready 非空 || finished)
    std::condition_variable done_cv;          // 主线程 (Run) 的等待 (finished && active==0)
    std::deque<int> ready;                    // 昂贵 op 的共享就绪队列
    bool finished = false;                    // outstanding 归零 → 全部 op 完成
    std::atomic<int> active{0};               // 仍在跑的 worker 任务数 (保证 es 存活)
};

// 执行队列 (对应 commit 1 的 executor.cc, 2118 行):
//   pending 计数就绪判定 → ScheduleReady: 便宜节点内联到当前 worker 的内联队列
//   (tail-call 语义, 零锁), 昂贵节点进共享就绪队列由 worker 池竞争; 每个 op 完成
//   时 outstanding 原子减一, 归零即整次 run 完成 (GPU 异步 op 的 done 回调也走
//   这条路径 —— ComputeAsync)。
class Executor {
public:
    explicit Executor(int workers) : pool_(workers) {}

    // 同步跑一次 (阻塞到全部 op 完成)。线程安全: 同一 executor 可被并发 Run
    // (各 run 独立 ExecState, worker 池共享 —— v3/v4 的并发推理语义不变)。
    // 主线程的完成等待在独立 done_cv 上: 若与 worker 共用 cv, GPU 完成回调的
    // notify_one 可能唤醒主线程 (谓词不满足又睡回) 而漏掉真正该醒的 worker →
    // worker 饿死死锁 (demo 6 接真 Metal 时复现过)。
    void Run(const GraphPlan& plan, RunState& st) {
        ExecState es;
        es.st = &st;
        es.pending.assign(plan.by_id.size(), 0);
        for (Node* n : plan.order) es.pending[n->id] = plan.in_degree[n->id];
        es.outstanding = static_cast<int>(plan.order.size());
        es.active = pool_.size();
        {
            std::lock_guard<std::mutex> l(es.mu);
            for (Node* n : plan.order)
                if (es.pending[n->id] == 0) es.ready.push_back(n->id);
        }
        for (int i = 0; i < pool_.size(); i++)
            pool_.Schedule([this, &plan, &es] { WorkerLoop(plan, es); });
        {
            std::unique_lock<std::mutex> l(es.mu);
            es.done_cv.wait(l, [&] { return es.finished && es.active == 0; });
        }
    }

    // 昂贵 op 判定 (对应 commit 1 的 ScheduleReady 按 op 开销分流; Eigen 的
    // CostModel 在此退化为按类型判定): O(N·F) 的矩阵乘族走共享队列, 其余
    // 逐元素/标量 op 内联。
    static bool IsCheap(const Node* n) {
        switch (n->type) {
        case MATMUL:
        case MATMUL_GRAD_A:
        case MATMUL_GRAD_B:
            return false;
        default:
            return true;
        }
    }

private:
    // 每 worker 的循环: 先清空自己的内联队列 (tail-call 语义), 再从共享队列取
    void WorkerLoop(const GraphPlan& plan, ExecState& es) {
        std::deque<int> inline_q;
        while (true) {
            while (!inline_q.empty()) {
                int id = inline_q.front();
                inline_q.pop_front();
                Process(plan, es, id, &inline_q);
            }
            int id;
            {
                std::unique_lock<std::mutex> l(es.mu);
                es.cv.wait(l, [&] { return es.finished || !es.ready.empty(); });
                if (es.finished && es.ready.empty()) break;
                id = es.ready.front();
                es.ready.pop_front();
            }
            Process(plan, es, id, &inline_q);
        }
        if (es.active.fetch_sub(1) == 1) {
            es.cv.notify_all();      // 唤醒可能仍等着的 worker (finished 后退出)
            es.done_cv.notify_all(); // 主线程: active 归零
        }
    }

    // 执行一个 op。CPU kernel 直接算完; GPU op 走 ComputeAsync (commit 1
    // gpu_device.cc): kernel 立即返回, done 回调传播输出 —— 真 Metal 异步下
    // done 从设备线程来, GPU op 与 CPU op 重叠执行。
    void Process(const GraphPlan& plan, ExecState& es, int id, std::deque<int>* inline_q) {
        Node* node = plan.by_id[id];
        if (plan.devices[id] == Device::GPU) {
            gpu_compute(node, *es.st, [this, &plan, &es, id] {
                PropagateOutputs(plan, es, id, nullptr);
            });
            return;
        }
        forward(node, *es.st);
        PropagateOutputs(plan, es, id, inline_q);
    }

    // 消费者计数递减 + 就绪调度 + 完成检测。同步 op 在这里跑; GPU 异步 op 的
    // done 回调也调它 (从 Metal 线程进来, inline_q 传 nullptr → 全部走共享队列)。
    void PropagateOutputs(const GraphPlan& plan, ExecState& es, int id,
                          std::deque<int>* inline_q) {
        std::lock_guard<std::mutex> l(es.mu);
        for (int c : plan.consumers[id])
            if (--es.pending[c] == 0) ScheduleReady(plan, es, c, inline_q);
        if (es.outstanding.fetch_sub(1) == 1) {
            es.finished = true;
            es.cv.notify_all();      // worker: 检查 finished 退出等待
            es.done_cv.notify_all(); // 主线程: Run 的完成等待
        }
    }

    // commit 1 的 ScheduleReady: 便宜 → 当前 worker 内联队列 (尾调用, 线程局部);
    // 昂贵 → 共享就绪队列 (worker 池竞争)。
    void ScheduleReady(const GraphPlan& plan, ExecState& es, int id,
                       std::deque<int>* inline_q) {
        if (inline_q && IsCheap(plan.by_id[id])) {
            inline_q->push_back(id);
        } else {
            es.ready.push_back(id);
            es.cv.notify_one();
        }
    }

    ThreadPool pool_;
};

}  // namespace lf
