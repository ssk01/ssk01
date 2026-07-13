// pack_A: strided-load + sequential-store  vs  sequential-load + strided-store
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <random>
#include <arm_neon.h>

using Clock = std::chrono::high_resolution_clock;

static float* alloc_f(size_t n){ void*p=nullptr; posix_memalign(&p,64,n*sizeof(float)); return (float*)p; }

static constexpr int MR = 8;

// 版本 A：跳读/顺写 (当前代码, gemm.cpp 第192-193行)
static void pack_A_strided_load(int N, const float* A, int ic, int mc, int pc, int kc, float* Apack) {
  int mpanel = mc / MR;
  for (int ip = 0; ip < mpanel; ++ip) {
    float* dst = Apack + (size_t)ip * kc * MR;
    int irow = ic + ip * MR;
    for (int p = 0; p < kc; ++p) {
      int col = pc + p;
      for (int r = 0; r < MR; ++r)
        dst[p * MR + r] = A[(size_t)(irow + r) * N + col];  // 读跳 N, 写连续
    }
  }
}

// 版本 B：顺读/跳写 (读 A 一行连续, 写到 dst 跳 8)
static void pack_A_strided_store(int N, const float* A, int ic, int mc, int pc, int kc, float* Apack) {
  int mpanel = mc / MR;
  for (int ip = 0; ip < mpanel; ++ip) {
    float* dst = Apack + (size_t)ip * kc * MR;
    int irow = ic + ip * MR;
    for (int r = 0; r < MR; ++r) {
      const float* src = A + (size_t)(irow + r) * N + pc;
      for (int p = 0; p < kc; ++p)
        dst[p * MR + r] = src[p];  // 读连续(一行 kc 个), 写跳 MR=8
    }
  }
}

static double best_in_ms(void (*f)(void*), void* ctx, int reps) {
  f(ctx); // warmup
  double b = 1e300;
  for (int r = 0; r < reps; ++r) {
    auto t0 = Clock::now();
    f(ctx);
    auto t1 = Clock::now();
    b = std::min(b, std::chrono::duration<double>(t1 - t0).count());
  }
  return b * 1e3;
}

// 用 struct + lambda 传给 best_in_ms，避免捕捉变量导致偏离
struct Ctx { int N; const float* A; int ic,mc,pc,kc; float* Abuf; int reps; };
void doit_A(void* v){ auto* c=(Ctx*)v; pack_A_strided_load(c->N,c->A,c->ic,c->mc,c->pc,c->kc,c->Abuf); }
void doit_B(void* v){ auto* c=(Ctx*)v; pack_A_strided_store(c->N,c->A,c->ic,c->mc,c->pc,c->kc,c->Abuf); }

int main(int argc, char**argv){
  setvbuf(stdout,nullptr,_IONBF,0);
  int Ns[] = {128, 256, 512, 1024, 2048, 4096};
  for (int N : Ns) {
    int MC = 256, KC = std::min(256,N);
    float *A = alloc_f((size_t)N*N), *Abuf = alloc_f((size_t)MC*KC);
    std::mt19937 rng(1);
    for (size_t i=0;i<(size_t)N*N;++i) A[i]=std::uniform_real_distribution<float>(-1,1)(rng);
    memset(Abuf, 0xcd, (size_t)MC*KC*sizeof(float));
    int reps = N<=1024 ? 200 : (N<=2048 ? 50 : 20);
    Ctx ctx = {N, A, 0, MC, 0, KC, Abuf, reps};
    double ta = best_in_ms(doit_A, &ctx, reps);
    double tb = best_in_ms(doit_B, &ctx, reps);
    printf("N=%5d  KC=%3d  strided_load=%7.3f us   strided_store=%7.3f us   %s faster\n",
           N, KC, ta*1000, tb*1000, ta < tb ? "load(now)" : "store(alt)");
    free(A); free(Abuf);
  }
  return 0;
}
