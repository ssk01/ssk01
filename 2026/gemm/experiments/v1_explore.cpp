// v1 探索：块大小调参 + 大矩阵（L2 放不下）场景
// 单线程。对比：
//   ikj              无分块
//   v1_bsX           原始 v1，i/k/j 三层都按 BS 分块（内层 j 被切短）
//   blk_k            只按 KC 分块 k、内层 j 跑满 N（修复短内层循环 + 减少 B 的 DRAM 重读）
//   blk_ik           按 MC 分块 i + KC 分块 k、内层 j 跑满
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <random>
using Clock = std::chrono::high_resolution_clock;

static float* alloc_f(size_t n){ void*p=nullptr; posix_memalign(&p,64,n*sizeof(float)); return (float*)p; }

static void ikj(int N,const float*A,const float*B,float*C){
  for(int i=0;i<N;++i){ float*Ci=C+(size_t)i*N;
    for(int j=0;j<N;++j)Ci[j]=0.f;
    for(int k=0;k<N;++k){ float a=A[(size_t)i*N+k]; const float*Bk=B+(size_t)k*N;
      for(int j=0;j<N;++j)Ci[j]+=a*Bk[j]; } } }

static void v1_bs(int N,const float*A,const float*B,float*C,int BS){
  memset(C,0,(size_t)N*N*sizeof(float));
  for(int ii=0;ii<N;ii+=BS){int im=std::min(ii+BS,N);
   for(int kk=0;kk<N;kk+=BS){int km=std::min(kk+BS,N);
    for(int jj=0;jj<N;jj+=BS){int jm=std::min(jj+BS,N);
     for(int i=ii;i<im;++i){float*Ci=C+(size_t)i*N;
      for(int k=kk;k<km;++k){float a=A[(size_t)i*N+k];const float*Bk=B+(size_t)k*N;
       for(int j=jj;j<jm;++j)Ci[j]+=a*Bk[j];}}}}} }

// 只分块 k：B 的 KC×N 面板留在 cache、被所有行 i 复用；内层 j 跑满整行
static void blk_k(int N,const float*A,const float*B,float*C,int KC){
  memset(C,0,(size_t)N*N*sizeof(float));
  for(int kk=0;kk<N;kk+=KC){int km=std::min(kk+KC,N);
   for(int i=0;i<N;++i){float*Ci=C+(size_t)i*N;
    for(int k=kk;k<km;++k){float a=A[(size_t)i*N+k];const float*Bk=B+(size_t)k*N;
     for(int j=0;j<N;++j)Ci[j]+=a*Bk[j];}}} }

// 分块 i + k：C 的 MC×N 面板常驻、B 的 KC×N 面板复用；内层 j 跑满
static void blk_ik(int N,const float*A,const float*B,float*C,int MC,int KC){
  memset(C,0,(size_t)N*N*sizeof(float));
  for(int ii=0;ii<N;ii+=MC){int im=std::min(ii+MC,N);
   for(int kk=0;kk<N;kk+=KC){int km=std::min(kk+KC,N);
    for(int i=ii;i<im;++i){float*Ci=C+(size_t)i*N;
     for(int k=kk;k<km;++k){float a=A[(size_t)i*N+k];const float*Bk=B+(size_t)k*N;
      for(int j=0;j<N;++j)Ci[j]+=a*Bk[j];}}}} }

static double gf(int N,double s){return 2.0*N*N*N/s/1e9;}
template<class F> static double best(F f,int reps){ f(); double b=1e300;
  for(int r=0;r<reps;++r){auto t0=Clock::now();f();auto t1=Clock::now();
    b=std::min(b,std::chrono::duration<double>(t1-t0).count());} return b; }

int main(int argc,char**argv){
  setvbuf(stdout,nullptr,_IONBF,0);
  std::mt19937 rng(1); std::uniform_real_distribution<float> d(-1,1);
  int Ns[]={1024,2048,4096,6144};
  double L2=16.0; // MB
  for(int N:Ns){
    float*A=alloc_f((size_t)N*N),*B=alloc_f((size_t)N*N),*C=alloc_f((size_t)N*N);
    for(size_t i=0;i<(size_t)N*N;++i){A[i]=d(rng);B[i]=d(rng);}
    int reps = N<=2048?3:1;
    double Bmb=4.0*N*N/1e6;
    printf("\n===== N=%d  (B=%.0f MB, L2=%.0f MB, B%s L2) =====\n",N,Bmb,L2,Bmb>L2?">":"<=");
    printf("%-12s %10s %10s\n","kernel","ms","GFLOPS");
    auto show=[&](const char*n,double s){printf("%-12s %10.1f %10.1f\n",n,s*1e3,gf(N,s));};
    show("ikj",       best([&]{ikj(N,A,B,C);},reps));
    show("v1_bs32",   best([&]{v1_bs(N,A,B,C,32);},reps));
    show("v1_bs64",   best([&]{v1_bs(N,A,B,C,64);},reps));
    show("v1_bs128",  best([&]{v1_bs(N,A,B,C,128);},reps));
    show("v1_bs256",  best([&]{v1_bs(N,A,B,C,256);},reps));
    show("blk_k256",  best([&]{blk_k(N,A,B,C,256);},reps));
    show("blk_ik",    best([&]{blk_ik(N,A,B,C,128,256);},reps));
    free(A);free(B);free(C);
  }
  return 0;
}
