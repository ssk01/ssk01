// gemm.cpp — 单精度矩阵乘法 C = A * B (row-major, square N)
//
// 一个 naive baseline + 一个 OpenMP 竞品，外加三个逐步优化的版本：
//   naive   : 三重循环 i-j-k，单线程（基准）
//   openmp  : #pragma omp parallel for 并行化的 i-k-j 循环（要打败的竞品，14 线程）
//   v1      : 单线程，缓存分块 + 循环重排（i-k-j），依赖编译器自动向量化
//   v2      : 单线程，NEON 寄存器分块微内核(8x8) + A/B 打包
//   v3      : v2 + 多线程（按行条带切分），目标：快过 OpenMP
//
// 编译见 Makefile。运行: ./gemm [N]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <thread>
#include <algorithm>
#include <random>
#include <arm_neon.h>
#include <omp.h>
#include <Accelerate/Accelerate.h>   // Apple BLAS (cblas_sgemm)

using Clock = std::chrono::high_resolution_clock;

static float* alloc_f(size_t n) {
  void* p = nullptr;
  if (posix_memalign(&p, 64, n * sizeof(float)) != 0) { perror("posix_memalign"); exit(1); }
  return (float*)p;
}

// ---------------------------------------------------------------------------
// baseline: 三重循环 i-j-k，单线程
// ---------------------------------------------------------------------------
static void gemm_naive(int N, const float* A, const float* B, float* C) {
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      float s = 0.f;
      for (int k = 0; k < N; ++k) s += A[i * N + k] * B[k * N + j];
      C[i * N + j] = s;
    }
}

// ---------------------------------------------------------------------------
// naive_t: 行列缓存优化 —— 转置 B，使内层点积对 A、Bᵀ 均按行连续访问
// （保持 naive 的 i-j-k 点积结构，仅修复 B 的列访问 cache miss）
// ---------------------------------------------------------------------------
static void gemm_naive_t(int N, const float* A, const float* B, float* C) {
  float* Bt = alloc_f((size_t)N * N);
  for (int k = 0; k < N; ++k)
    for (int j = 0; j < N; ++j)
      Bt[(size_t)j * N + k] = B[(size_t)k * N + j];
  for (int i = 0; i < N; ++i) {
    const float* Ai = A + (size_t)i * N;
    for (int j = 0; j < N; ++j) {
      const float* Bj = Bt + (size_t)j * N;
      float s = 0.f;
      for (int k = 0; k < N; ++k) s += Ai[k] * Bj[k];
      C[(size_t)i * N + j] = s;
    }
  }
  free(Bt);
}

// ---------------------------------------------------------------------------
// ikj: 与 openmp 相同的 i-k-j 循环（改下标顺序），但单线程、不加 OpenMP
// 用于隔离「循环重排」本身的收益（不含并行）
// ---------------------------------------------------------------------------
static void gemm_ikj(int N, const float* A, const float* B, float* C) {
  for (int i = 0; i < N; ++i) {
    float* Ci = C + (size_t)i * N;
    for (int j = 0; j < N; ++j) Ci[j] = 0.f;
    for (int k = 0; k < N; ++k) {
      const float a = A[(size_t)i * N + k];
      const float* Bk = B + (size_t)k * N;
      for (int j = 0; j < N; ++j) Ci[j] += a * Bk[j];
    }
  }
}

// ---------------------------------------------------------------------------
// 竞品: OpenMP 并行化的 i-k-j 循环（缓存友好且自动向量化）
// ---------------------------------------------------------------------------
static void gemm_openmp(int N, const float* A, const float* B, float* C) {
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < N; ++i) {
    float* Ci = C + (size_t)i * N;
    for (int j = 0; j < N; ++j) Ci[j] = 0.f;
    for (int k = 0; k < N; ++k) {
      const float a = A[(size_t)i * N + k];
      const float* Bk = B + (size_t)k * N;
      for (int j = 0; j < N; ++j) Ci[j] += a * Bk[j];
    }
  }
}

// ---------------------------------------------------------------------------
// v1: 单线程，缓存分块 + i-k-j 循环重排
// ---------------------------------------------------------------------------
static void gemm_v1(int N, const float* A, const float* B, float* C) {
  std::memset(C, 0, (size_t)N * N * sizeof(float));
  const int BS = 64;
  for (int ii = 0; ii < N; ii += BS) {
    const int imax = std::min(ii + BS, N);
    for (int kk = 0; kk < N; kk += BS) {
      const int kmax = std::min(kk + BS, N);
      for (int jj = 0; jj < N; jj += BS) {
        const int jmax = std::min(jj + BS, N);
        for (int i = ii; i < imax; ++i) {
          float* Ci = C + (size_t)i * N;
          for (int k = kk; k < kmax; ++k) {
            const float a = A[(size_t)i * N + k];
            const float* Bk = B + (size_t)k * N;
            for (int j = jj; j < jmax; ++j) Ci[j] += a * Bk[j];
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// v2 / v3 共用: NEON 8x8 微内核 + 打包
// 要求 N % 8 == 0（benchmark 尺寸均为 8 的倍数）
// ---------------------------------------------------------------------------
static constexpr int MR = 8, NR = 8;

// 8x8 微内核: C[8][8] += A_panel(kc x 8) * B_panel(kc x 8)
// A_panel 布局: [p*8 + r]  (第 p 步的 8 个行值)
// B_panel 布局: [p*8 + c]  (第 p 步的 8 个列值)
static inline void micro_8x8(int kc, const float* __restrict Ap,
                             const float* __restrict Bp,
                             float* __restrict C, int ldc) {
  float32x4_t c0l = vdupq_n_f32(0), c0h = vdupq_n_f32(0);
  float32x4_t c1l = vdupq_n_f32(0), c1h = vdupq_n_f32(0);
  float32x4_t c2l = vdupq_n_f32(0), c2h = vdupq_n_f32(0);
  float32x4_t c3l = vdupq_n_f32(0), c3h = vdupq_n_f32(0);
  float32x4_t c4l = vdupq_n_f32(0), c4h = vdupq_n_f32(0);
  float32x4_t c5l = vdupq_n_f32(0), c5h = vdupq_n_f32(0);
  float32x4_t c6l = vdupq_n_f32(0), c6h = vdupq_n_f32(0);
  float32x4_t c7l = vdupq_n_f32(0), c7h = vdupq_n_f32(0);

  for (int p = 0; p < kc; ++p) {
    float32x4_t b0 = vld1q_f32(Bp + 0);   // cols 0..3
    float32x4_t b1 = vld1q_f32(Bp + 4);   // cols 4..7
    float32x4_t a0 = vld1q_f32(Ap + 0);   // rows 0..3
    float32x4_t a1 = vld1q_f32(Ap + 4);   // rows 4..7
    Bp += 8; Ap += 8;

    c0l = vfmaq_laneq_f32(c0l, b0, a0, 0); c0h = vfmaq_laneq_f32(c0h, b1, a0, 0);
    c1l = vfmaq_laneq_f32(c1l, b0, a0, 1); c1h = vfmaq_laneq_f32(c1h, b1, a0, 1);
    c2l = vfmaq_laneq_f32(c2l, b0, a0, 2); c2h = vfmaq_laneq_f32(c2h, b1, a0, 2);
    c3l = vfmaq_laneq_f32(c3l, b0, a0, 3); c3h = vfmaq_laneq_f32(c3h, b1, a0, 3);
    c4l = vfmaq_laneq_f32(c4l, b0, a1, 0); c4h = vfmaq_laneq_f32(c4h, b1, a1, 0);
    c5l = vfmaq_laneq_f32(c5l, b0, a1, 1); c5h = vfmaq_laneq_f32(c5h, b1, a1, 1);
    c6l = vfmaq_laneq_f32(c6l, b0, a1, 2); c6h = vfmaq_laneq_f32(c6h, b1, a1, 2);
    c7l = vfmaq_laneq_f32(c7l, b0, a1, 3); c7h = vfmaq_laneq_f32(c7h, b1, a1, 3);
  }

  #define STORE_ROW(r, cl, ch)                                            \
    vst1q_f32(C + (size_t)(r) * ldc + 0, vaddq_f32(vld1q_f32(C + (size_t)(r) * ldc + 0), cl)); \
    vst1q_f32(C + (size_t)(r) * ldc + 4, vaddq_f32(vld1q_f32(C + (size_t)(r) * ldc + 4), ch));
  STORE_ROW(0, c0l, c0h) STORE_ROW(1, c1l, c1h)
  STORE_ROW(2, c2l, c2h) STORE_ROW(3, c3l, c3h)
  STORE_ROW(4, c4l, c4h) STORE_ROW(5, c5l, c5h)
  STORE_ROW(6, c6l, c6h) STORE_ROW(7, c7l, c7h)
  #undef STORE_ROW
}

// 打包 B 的一个 K 条带 (行 pc..pc+kc, 全部 N 列) 成 NR 列的微面板
static void pack_B(int N, const float* B, int pc, int kc, float* Bpack) {
  const int npanel = N / NR;
  for (int jp = 0; jp < npanel; ++jp) {
    float* dst = Bpack + (size_t)jp * kc * NR;
    const int jcol = jp * NR;
    for (int p = 0; p < kc; ++p) {
      const float* src = B + (size_t)(pc + p) * N + jcol;
      vst1q_f32(dst + p * NR + 0, vld1q_f32(src + 0));
      vst1q_f32(dst + p * NR + 4, vld1q_f32(src + 4));
    }
  }
}

// 打包 A 的一个块 (行 ic..ic+mc, 列 pc..pc+kc) 成 MR 行的微面板
static void pack_A(int N, const float* A, int ic, int mc, int pc, int kc, float* Apack) {
  const int mpanel = mc / MR;
  for (int ip = 0; ip < mpanel; ++ip) {
    float* dst = Apack + (size_t)ip * kc * MR;
    const int irow = ic + ip * MR;
    for (int p = 0; p < kc; ++p) {
      const int col = pc + p;
      for (int r = 0; r < MR; ++r)
        dst[p * MR + r] = A[(size_t)(irow + r) * N + col];
    }
  }
}

// 计算 C 的行区间 [r0, r1)（分块 + 打包 + 微内核）。C 需事先清零。
static void gemm_blocked_rows(int N, const float* A, const float* B, float* C,
                              int r0, int r1, float* Apack, float* Bpack,
                              int MC, int KC) {
  for (int pc = 0; pc < N; pc += KC) {
    const int kc = std::min(KC, N - pc);
    pack_B(N, B, pc, kc, Bpack);
    for (int ic = r0; ic < r1; ic += MC) {
      const int mc = std::min(MC, r1 - ic);
      pack_A(N, A, ic, mc, pc, kc, Apack);
      const int mpanel = mc / MR;
      const int npanel = N / NR;
      for (int jp = 0; jp < npanel; ++jp) {
        const float* Bp = Bpack + (size_t)jp * kc * NR;
        for (int ip = 0; ip < mpanel; ++ip) {
          const float* Ap = Apack + (size_t)ip * kc * MR;
          micro_8x8(kc, Ap, Bp, C + (size_t)(ic + ip * MR) * N + jp * NR, N);
        }
      }
    }
  }
}

// v2: 单线程
static void gemm_v2(int N, const float* A, const float* B, float* C) {
  std::memset(C, 0, (size_t)N * N * sizeof(float));
  const int MC = 256, KC = 256;
  float* Apack = alloc_f((size_t)MC * KC);
  float* Bpack = alloc_f((size_t)KC * N);
  gemm_blocked_rows(N, A, B, C, 0, N, Apack, Bpack, MC, KC);
  free(Apack); free(Bpack);
}

// v3: 多线程（按行条带切分），每线程持有独立打包缓冲区
static void gemm_v3(int N, const float* A, const float* B, float* C) {
  std::memset(C, 0, (size_t)N * N * sizeof(float));
  const int MC = 256, KC = 256;
  int T = (int)std::thread::hardware_concurrency();
  if (T <= 0) T = 8;
  const int npanel = N / MR;              // 8 行为一个面板
  if (T > npanel) T = npanel;

  std::vector<std::thread> th;
  th.reserve(T);
  int base = npanel / T, rem = npanel % T, start = 0;
  for (int t = 0; t < T; ++t) {
    const int cnt = base + (t < rem ? 1 : 0);
    const int r0 = start * MR;
    const int r1 = (start + cnt) * MR;
    start += cnt;
    if (cnt == 0) continue;
    th.emplace_back([=]() {
      float* Apack = alloc_f((size_t)MC * KC);
      float* Bpack = alloc_f((size_t)KC * N);
      gemm_blocked_rows(N, A, B, C, r0, r1, Apack, Bpack, MC, KC);
      free(Apack); free(Bpack);
    });
  }
  for (auto& x : th) x.join();
}

// cblas: Apple Accelerate 库 cblas_sgemm 参考实现
static void gemm_cblas(int N, const float* A, const float* B, float* C) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              N, N, N, 1.f, A, N, B, N, 0.f, C, N);
}

// ---------------------------------------------------------------------------
// 测量与校验
// ---------------------------------------------------------------------------
using GemmFn = void (*)(int, const float*, const float*, float*);

static double bench(GemmFn fn, int N, const float* A, const float* B, float* C, int reps) {
  fn(N, A, B, C);  // warmup
  double best = 1e300;
  for (int r = 0; r < reps; ++r) {
    auto t0 = Clock::now();
    fn(N, A, B, C);
    auto t1 = Clock::now();
    double s = std::chrono::duration<double>(t1 - t0).count();
    best = std::min(best, s);
  }
  return best;
}

static double max_rel_err(int N, const float* ref, const float* got) {
  double maxdiff = 0, maxref = 0;
  for (size_t i = 0; i < (size_t)N * N; ++i) {
    maxdiff = std::max(maxdiff, (double)std::fabs(ref[i] - got[i]));
    maxref = std::max(maxref, (double)std::fabs(ref[i]));
  }
  return maxref > 0 ? maxdiff / maxref : maxdiff;
}

static double gflops(int N, double sec) { return 2.0 * N * N * N / sec / 1e9; }

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  // 1) 正确性校验 (N=512, 以 naive 为参考)
  {
    const int N = 512;
    float *A = alloc_f((size_t)N*N), *B = alloc_f((size_t)N*N);
    float *Cref = alloc_f((size_t)N*N), *Ct = alloc_f((size_t)N*N);
    for (size_t i = 0; i < (size_t)N*N; ++i) { A[i] = dist(rng); B[i] = dist(rng); }
    gemm_naive(N, A, B, Cref);
    struct { const char* name; GemmFn fn; } ks[] = {
      {"naive_t", gemm_naive_t}, {"ikj", gemm_ikj}, {"openmp", gemm_openmp}, {"v1", gemm_v1}, {"v2", gemm_v2}, {"v3", gemm_v3}, {"cblas", gemm_cblas},
    };
    printf("== 正确性校验 (N=%d, 参考=naive) ==\n", N);
    for (auto& k : ks) {
      k.fn(N, A, B, Ct);
      printf("  %-7s max_rel_err = %.2e  %s\n", k.name, max_rel_err(N, Cref, Ct),
             max_rel_err(N, Cref, Ct) < 1e-3 ? "OK" : "FAIL");
    }
    free(A); free(B); free(Cref); free(Ct);
  }

  // 2) 性能基准
  std::vector<int> sizes;
  if (argc > 1) sizes.push_back(std::atoi(argv[1]));
  else sizes = {1024, 2048};

  printf("\n线程数 (OpenMP/v3): %d\n", (int)std::thread::hardware_concurrency());
  for (int N : sizes) {
    if (N % 8 != 0) { printf("N=%d 需为 8 的倍数, 跳过\n", N); continue; }
    float *A = alloc_f((size_t)N*N), *B = alloc_f((size_t)N*N), *C = alloc_f((size_t)N*N);
    for (size_t i = 0; i < (size_t)N*N; ++i) { A[i] = dist(rng); B[i] = dist(rng); }
    const int reps = N <= 1024 ? 5 : (N <= 2048 ? 3 : 2);
    const bool big = N > 2048;   // 大尺寸跳过单线程标量版(太慢), 单线程 v2 只测 1 次

    printf("\n================ N = %d ================\n", N);
    printf("%-8s %10s %10s %10s\n", "kernel", "time(ms)", "GFLOPS", "vs openmp");

    double t_naive = (N <= 512) ? bench(gemm_naive, N, A, B, C, 1) : -1;
    double t_naive_t = big ? -1 : bench(gemm_naive_t, N, A, B, C, reps);
    double t_ikj = big ? -1 : bench(gemm_ikj, N, A, B, C, reps);
    double t_omp = bench(gemm_openmp, N, A, B, C, reps);
    double t_v1  = big ? -1 : bench(gemm_v1,  N, A, B, C, reps);
    double t_v2  = bench(gemm_v2,  N, A, B, C, big ? 1 : reps);
    double t_v3  = bench(gemm_v3,  N, A, B, C, reps);
    double t_cblas = bench(gemm_cblas, N, A, B, C, reps);

    auto row = [&](const char* n, double t) {
      if (t < 0) { printf("%-8s %10s %10s %10s\n", n, "-", "-", "-"); return; }
      printf("%-8s %10.2f %10.1f %9.2fx\n", n, t*1e3, gflops(N,t), t_omp/t);
    };
    row("naive", t_naive);
    row("naive_t", t_naive_t);
    row("ikj", t_ikj);
    row("openmp", t_omp);
    row("v1", t_v1);
    row("v2", t_v2);
    row("v3", t_v3);
    row("cblas", t_cblas);

    printf("-> v3 相对 openmp: %.2fx %s", t_omp/t_v3,
           t_v3 < t_omp ? "(更快 ✓)" : "(更慢 ✗)");
    printf("   |  v3 相对 cblas: %.2fx\n", t_cblas/t_v3);
    fflush(stdout);
    free(A); free(B); free(C);
  }
  return 0;
}
