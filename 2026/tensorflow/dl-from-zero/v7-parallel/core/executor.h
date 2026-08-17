#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
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

// 一次 run 的执行状态 (对应 commit 1 的 ExecutorState)。closure 模型下为
// 堆分配: 最后一个 op 完成的调用者负责 delete (对齐 SimpleExecutorState 的
// Finish(), executor.cc:1973-1982 —— 栈上对象在 closure 模型里会悬垂)。
// 主线程的完成等待在 Run 的栈上 Sync 对象上 (es 先被 delete, 等待点不能
// 是 es 的成员)。
struct ExecState {
    ExecState(const GraphPlan& plan, RunState* st_, std::function<void()> done_cb_)
        : st(st_), done_cb(std::move(done_cb_)) {
        pending.assign(plan.by_id.size(), 0);
        for (Node* n : plan.order) pending[n->id] = plan.in_degree[n->id];
        outstanding = static_cast<int>(plan.order.size());
    }
    RunState* st = nullptr;                   // 值槽 (输出 + vars 指针)
    std::vector<int> pending;                 // pending[id] = 剩余未满足的入边数
    std::atomic<int> outstanding{0};          // 未完成 op 数 (commit 1 num_outstanding_ops_)
    std::mutex mu;                            // 保护 pending 递减
    std::function<void()> done_cb;            // 完成通知 (捕获 Run 的 Sync, 不涉及 this)
};

// 执行队列 (对应 commit 1 的 executor.cc, 2118 行) —— closure 模型, 对齐
// SimpleExecutorState (executor.cc:1567-1982):
//   pending 计数就绪判定 → 批量 ScheduleReady: 便宜节点内联到当前线程的
//   inline 队列 (tail-call 语义, 零锁), 昂贵节点作为闭包提交线程池
//   (runner_ 语义); 每个 op 完成时 outstanding 原子减一, 归零的调用者
//   (最后一个完成者) 负责回收 ExecState + 通知主线程 (GPU 异步 op 的
//   done 回调也走这条路径 —— ComputeAsync)。
//
// 为什么是 closure 而不是 v7 初版的固定 worker 池: v7 的 WorkerLoop 在
// es.cv 上 park 等就绪队列, 一个 run 卡在 GPU 异步等待 (共享队列已空) 时
// 其全部 worker 独占池线程, 其他并发 run 的任务排队饿死 (convoy, 行为上
// 趋近一次只能跑一个 infer 请求)。closure 模型下没有 cv-wait: 没有就绪
// 节点时这条计算链就终止、线程归还池, 空闲 run 占用线程数为 0 —— 对齐
// TF 原版 (demo 6 的 done_cv 饿死问题也随之消失: 主线程等待点独立)。
class Executor {
public:
    explicit Executor(int workers) : pool_(workers) {}

    // 同步跑一次 (阻塞到全部 op 完成)。线程安全: 同一 executor 可被并发 Run
    // (各 run 独立 ExecState, worker 池共享 —— v3/v4 的并发推理语义不变)。
    void Run(const GraphPlan& plan, RunState& st) {
        // 完成通知必须独立于 ExecState: es 在完成者的 Finish 里被 delete,
        // 主线程的等待点与之无关才没有悬垂 (对齐 TF: done_cb 闭包不碰 this)。
        struct Sync {
            std::mutex mu;
            std::condition_variable cv;
            bool done = false;
        } sync;
        ExecState* es = new ExecState(plan, &st, [&sync] {
            std::lock_guard<std::mutex> l(sync.mu);
            sync.done = true;
            sync.cv.notify_all();
        });
        if (plan.order.empty()) {  // 空图: 直接完成 (对齐 RunAsync 的 ready.empty 分支)
            delete es;
            return;
        }
        // 根节点 (入度 0) 全部作为闭包提交池 (对齐 RunAsync 的
        // ScheduleReady(ready, nullptr), executor.cc:1768)。必须先持锁把
        // 根节点收集完再提交: 闭包一旦入池, 池线程会立即执行并递减 pending,
        // 边提交边扫 pending 会把「已被并发递减到 0 的非根节点」误判为根节点
        // 重复提交 → op 重复执行 → outstanding 提前归零 → Finish 在提交循环
        // 期间删除 es → 提交循环 use-after-free (v8 实测 SIGSEGV)。
        std::vector<int> roots;
        {
            std::lock_guard<std::mutex> l(es->mu);
            for (Node* n : plan.order)
                if (es->pending[n->id] == 0) roots.push_back(n->id);
        }
        for (int id : roots)
            pool_.Schedule([this, &plan, es, id] { Process(plan, es, id); });
        std::unique_lock<std::mutex> l(sync.mu);
        sync.cv.wait(l, [&] { return sync.done; });
    }

    // 昂贵 op 判定 (对应 commit 1 的 ScheduleReady 按 op 开销分流; Eigen 的
    // CostModel 在此退化为按类型判定): O(N·F) 的矩阵乘族走线程池, 其余
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
    // 执行一个就绪节点 (对齐 SimpleExecutorState::Process, executor.cc:1822-
    // 1924): 先清空自己的内联队列 (tail-call 语义), 队列空即返回、线程归还
    // 池。GPU 节点提交异步后立即返回 —— gpu_compute 的 CPU 兜底路径 (非
    // matmul / 形状不符 / Metal 不可用, kernels.h) 会同步调 done(), 可能
    // 在提交线程上直接触发 Finish (delete es), 循环不能再访问 es; 且 GPU
    // 节点 IsCheap=false 只会是循环的首个节点, return 不丢内联机会。
    void Process(const GraphPlan& plan, ExecState* es, int id) {
        std::deque<int> inline_q;
        inline_q.push_back(id);
        while (!inline_q.empty()) {
            int cur = inline_q.front();
            inline_q.pop_front();
            Node* node = plan.by_id[cur];
            if (plan.devices[cur] == Device::GPU) {
                gpu_compute(node, *es->st, [this, &plan, es, cur] {
                    if (PropagateOutputs(plan, es, cur, nullptr)) Finish(es);
                });
                return;
            }
            forward(node, *es->st);
            if (PropagateOutputs(plan, es, cur, &inline_q)) Finish(es);
        }
    }

    // 消费者计数递减 + 批量收集就绪 + 完成检测 (对齐 SimpleExecutorState 的
    // NodeDone, executor.cc:1926-1971)。同步 op 在这里跑; GPU 异步 op 的
    // done 回调也调它 (从 Metal 线程进来, inline_q 传 nullptr → 全部走池)。
    // 返回 completed: 仅 outstanding 归零的调用者为 true, 该调用者负责
    // Finish (delete es + 通知主线程); delete 发生在锁外 (返回之后)。
    bool PropagateOutputs(const GraphPlan& plan, ExecState* es, int id,
                          std::deque<int>* inline_q) {
        std::vector<int> ready;
        bool completed;
        {
            std::lock_guard<std::mutex> l(es->mu);
            for (int c : plan.consumers[id])
                if (--es->pending[c] == 0) ready.push_back(c);
            completed = (es->outstanding.fetch_sub(1) == 1);
        }
        ScheduleReady(plan, es, ready, inline_q);  // 锁外调度 (可能派发闭包)
        return completed;
    }

    // commit 1 的 ScheduleReady (批量版, 对齐 executor.cc:1701-1743):
    //   便宜 → 当前线程内联队列; 昂贵 → 派发线程池; 一批里的最后一个昂贵
    //   节点在 inline 队列为空时直接内联 (Tail recursion optimization,
    //   executor.cc:1732-1742 —— 省一次 入队→唤醒→出队 往返)。
    //   inline_q == nullptr (GPU 回调) → 全部派发线程池。
    void ScheduleReady(const GraphPlan& plan, ExecState* es,
                       const std::vector<int>& ready, std::deque<int>* inline_q) {
        if (ready.empty()) return;
        if (inline_q == nullptr) {
            for (int id : ready)
                pool_.Schedule([this, &plan, es, id] { Process(plan, es, id); });
            return;
        }
        int last_expensive = -1;
        for (int id : ready) {
            if (IsCheap(plan.by_id[id])) {
                inline_q->push_back(id);
            } else {
                if (last_expensive != -1)
                    pool_.Schedule([this, &plan, es, id = last_expensive] {
                        Process(plan, es, id);
                    });
                last_expensive = id;
            }
        }
        if (last_expensive != -1) {
            if (inline_q->empty()) {
                inline_q->push_back(last_expensive);  // tail recursion
            } else {
                pool_.Schedule([this, &plan, es, id = last_expensive] {
                    Process(plan, es, id);
                });
            }
        }
    }

    // 完成收尾: 先取出 done_cb 再回收 es (对齐 SimpleExecutorState::Finish
    // 的 delete this, executor.cc:1973-1982 —— done_cb 闭包只捕获 Run 的
    // Sync, 不涉及 es)。done_cb 直接调用, 不经线程池 (主线程只是在 cv 上
    // 等, 谁唤醒它都一样)。
    void Finish(ExecState* es) {
        auto done_cb = es->done_cb;
        delete es;
        done_cb();
    }

    ThreadPool pool_;
};

}  // namespace lf
