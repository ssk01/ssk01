#pragma once
#include "../v9-op-registry/framework/op_registry.h"
#include "../v9-op-registry/session.h"
#include "primitive.h"
#include <Accelerate/Accelerate.h>
#include <chrono>
namespace lf10 {
using Clock = std::chrono::steady_clock;
inline double us(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}
enum class Backend { Scalar, Accelerate, Jit };
struct MatmulRuntime {
  Backend backend;
  primitive::PrimitiveCache cache;
  double prepare_us = 0, execute_us = 0;
  MatmulRuntime(Backend b, size_t cap) : backend(b), cache(cap) {
#ifndef LF_LIBXSMM
    if (b == Backend::Jit)
      throw std::runtime_error(
          "LIBXSMM support not built; JIT cannot fallback");
#endif
  }
  void reset() {
    cache.reset_stats();
    prepare_us = execute_us = 0;
  }
  lf::Tensor compute(const lf::Tensor &a, const lf::Tensor &b) {
    if (a.shape.size() != 2 || b.shape.size() != 2 ||
        a.shape[1] != b.shape[0] || a.shape[0] <= 0 || a.shape[1] <= 0 ||
        b.shape[1] <= 0 || a.data.size() != size_t(a.shape[0]) * a.shape[1] ||
        b.data.size() != size_t(b.shape[0]) * b.shape[1])
      throw std::invalid_argument(
          "MatMul2D requires dense row-major [M,K]*[K,N]");
    int m = a.shape[0], k = a.shape[1], n = b.shape[1];
    lf::Tensor c(std::vector<int>{m, n});
    auto begin = Clock::now();
    std::shared_ptr<primitive::Primitive> p;
    if (backend == Backend::Jit)
      p = cache.get({m, n, k, k, n, n, primitive::MatmulDesc::RowMajor});
    auto ready = Clock::now();
    if (p)
      p->execute(a.data.data(), b.data.data(), c.data.data());
    else if (backend == Backend::Accelerate)
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1,
                  a.data.data(), k, b.data.data(), n, 0, c.data.data(), n);
    else
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
          float sum = 0;
          for (int z = 0; z < k; ++z)
            sum += a.data[i * k + z] * b.data[z * n + j];
          c.data[i * n + j] = sum;
        }
    execute_us += us(ready, Clock::now());
    prepare_us += us(begin, ready);
    return c;
  }
};
// The v9 representation has a distinct appended NodeType; optimization and
// autodiff cannot confuse this operation with the original matrix-vector op.
inline lf::Node *matmul2d(lf::Graph &g, lf::Node *a, lf::Node *b) {
  return g.matmul2d(a, b);
}
inline void install(std::shared_ptr<MatmulRuntime> runtime) {
  if (!runtime)
    throw std::invalid_argument("null MatMul runtime");
  lf::OpDefBuilder("MatMul2D")
      .Input("a: float")
      .Input("b: float")
      .Output("c: float")
      .Finalize();
  lf::KernelDefBuilder("MatMul2D")
      .DeviceType(lf::Device::CPU)
      .Kernel([runtime = std::move(runtime)](lf::OpKernelContext *ctx) {
        ctx->set_output(runtime->compute(ctx->input(0), ctx->input(1)));
      })
      .Finalize();
}
// The registry owns runtime until another install. One runtime/one Session
// worker, serial requests only; configure before execution. Forward only.
} // namespace lf10
