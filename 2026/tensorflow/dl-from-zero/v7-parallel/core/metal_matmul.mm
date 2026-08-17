#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <mutex>
#include "metal_matmul.h"
#include "metallib_data.h"  // xxd -i 生成的嵌入 metallib (Makefile 产物, 全局符号)

// ObjC++ (Metal.hpp 未随 Xcode 提供 → Metal.h + ARC)。
// 统一内存架构: host 与 GPU 共享同一物理内存, MTLStorageModeShared buffer 直接
// 被两端读写 → 跨设备边界隐式化 (commit 1 的 PCIe GPU 需要 graph_partition +
// _Send/_Recv + rendezvous 显式搬运, 我们这里是共享内存, 等价物是 buffer 边界)。
namespace lf {
namespace metal {

namespace {
id<MTLDevice> g_device = nil;
id<MTLCommandQueue> g_queue = nil;
id<MTLComputePipelineState> g_pipeline = nil;
bool g_ok = false;
std::once_flag g_once;

// 首次调用惰性初始化: 拿默认设备 → 载入嵌入的 metallib (Makefile 用 xxd 生成
// build/metallib_data.h) → 建 matmul 的 compute pipeline。
bool Init() {
    NSError* err = nil;
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) return false;
    g_queue = [g_device newCommandQueue];

    dispatch_data_t data = dispatch_data_create(
        build_metal_matmul_metallib, build_metal_matmul_metallib_len, nil,
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    id<MTLLibrary> lib = [g_device newLibraryWithData:data error:&err];
    if (!lib) return false;
    id<MTLFunction> fn = [lib newFunctionWithName:@"matmul_nv"];
    if (!fn) return false;
    g_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&err];
    return g_pipeline != nil;
}

// CPU 兜底: Metal 不可用时的同步计算 (不应发生 —— placer 会全 CPU 放置,
// MatmulAsync 不会被调; 双保险, 与 shader 同累加顺序)。
void CpuFallback(const float* x, const float* w, float* out, int N, int F) {
    for (int n = 0; n < N; n++) {
        float acc = 0.0f;
        for (int f = 0; f < F; f++) acc += x[n * F + f] * w[f];
        out[n] = acc;
    }
}
}  // namespace

bool Available() {
    std::call_once(g_once, [] { g_ok = Init(); });
    return g_ok;
}

void MatmulAsync(const float* x, const float* w, float* out, int N, int F,
                 std::function<void()> done) {
    if (!Available()) {
        CpuFallback(x, w, out, N, F);
        done();
        return;
    }

    MTLResourceOptions opts = MTLResourceStorageModeShared;
    id<MTLBuffer> bx = [g_device newBufferWithBytes:x
                         length:(N * F * sizeof(float))
                        options:opts];
    id<MTLBuffer> bw = [g_device newBufferWithBytes:w
                         length:(F * sizeof(float))
                        options:opts];
    id<MTLBuffer> bo = [g_device newBufferWithLength:(N * sizeof(float))
                                             options:opts];
    id<MTLBuffer> bF = [g_device newBufferWithBytes:&F
                                             length:sizeof(int)
                                            options:opts];
    if (!bx || !bw || !bo || !bF) {
        CpuFallback(x, w, out, N, F);
        done();
        return;
    }

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_pipeline];
    [enc setBuffer:bx offset:0 atIndex:0];
    [enc setBuffer:bw offset:0 atIndex:1];
    [enc setBuffer:bo offset:0 atIndex:2];
    [enc setBuffer:bF offset:0 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(N, 1, 1)   // 每输出行一个 thread
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];

    // ComputeAsync 语义: commit 后立即返回, done 在 GPU 完成后从 Metal 内部线程
    // 调用。done 只做 enqueue + 计数 (executor 的 PropagateOutputs, 不进 CPU
    // kernel) → 不会等 worker 线程, 无死锁。
    [cb addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull cb) {
        (void)cb;
        std::memcpy(out, bo.contents, N * sizeof(float));  // 共享内存直读
        done();
    }];
    [cb commit];
}

}  // namespace metal
}  // namespace lf
