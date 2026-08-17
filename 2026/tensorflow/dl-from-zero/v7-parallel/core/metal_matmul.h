#pragma once
#include <functional>

namespace lf {
namespace metal {

// Metal 设备是否可用 (首次调用惰性初始化; 无 GPU 环境返回 false → placer
// 全部 CPU 放置)。线程安全。
bool Available();

// 异步 [N,F]×[F]→[N] matmul: 输入拷入共享存储 buffer (Apple 统一内存 →
// host/GPU 同一物理内存), dispatch 后立即返回; GPU 完成后 done() 从 Metal
// 内部线程调用 —— executor 用它传播输出 (GPU op 与 CPU op 重叠)。
// x/w 的指针在 done 被调用前必须保持有效。线程安全。
void MatmulAsync(const float* x, const float* w, float* out, int N, int F,
                 std::function<void()> done);

}  // namespace metal
}  // namespace lf
