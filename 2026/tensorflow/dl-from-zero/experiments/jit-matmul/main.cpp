#include <libxsmm.h>
#include <libxsmm_generator.h>
#include <cstring>
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <list>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
struct Data {
 int m,n,k;std::vector<float>a,b,c,ref;
 Data(int M,int N,int K):m(M),n(N),k(K),a(m*k),b(k*n),c(m*n),ref(m*n){
  std::mt19937 gen(73+m+n+k);std::uniform_real_distribution<float>d(-1,1);
  for(auto&v:a)v=d(gen);for(auto&v:b)v=d(gen);
  cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,m,n,k,1,a.data(),m,b.data(),k,0,ref.data(),m);
 }
 void blas(){cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,m,n,k,1,a.data(),m,b.data(),k,0,c.data(),m);}
 void run(libxsmm_gemmfunction f){libxsmm_gemm_param p{};p.a.primary=a.data();p.b.primary=b.data();p.c.primary=c.data();f(&p);}
 double check(){double err=0;for(size_t i=0;i<c.size();++i){err=std::max(err,double(std::abs(c[i]-ref[i])));if(!std::isfinite(c[i])||std::abs(c[i]-ref[i])>2e-4*(1+std::abs(ref[i])))throw std::runtime_error("numerical mismatch");}return err;}
};
libxsmm_gemmfunction dispatch(int m,int n,int k){
 auto shape=libxsmm_create_gemm_shape(m,n,k,m,k,m,LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32);
 auto fn=libxsmm_dispatch_gemm(shape,LIBXSMM_GEMM_FLAG_BETA_0,LIBXSMM_GEMM_PREFETCH_NONE);
 if(!fn)throw std::runtime_error("no generated kernel");return fn;
}
libxsmm_kernel_info info(libxsmm_gemmfunction fn){libxsmm_kernel_info i{};if(libxsmm_get_kernel_info(reinterpret_cast<const void*>(fn),&i)||i.is_reference_kernel)throw std::runtime_error("not a JIT kernel");return i;}
size_t registry_size(){libxsmm_registry_info i{};if(libxsmm_get_registry_info(&i))throw std::runtime_error("registry query failed");return i.size;}
void release(libxsmm_gemmfunction f){libxsmm_release_kernel(reinterpret_cast<const void*>(f));}
volatile double sink=0;
void exec_case(int m,int n,int k,int seed){
 Data d(m,n,k);auto t0=Clock::now();auto fn=dispatch(m,n,k);auto t1=Clock::now();auto i=info(fn);
 d.run(fn);double err=d.check();d.b[0]+=.125f;d.blas();d.ref=d.c;d.run(fn);err=std::max(err,d.check());
 std::vector<double> jit,blas;
 for(int r=0;r<15;++r)for(int j=0;j<2;++j){bool usejit=(j+r+seed)%2;
  for(int w=0;w<10;++w){if(usejit)d.run(fn);else d.blas();}
  auto start=Clock::now();for(int z=0;z<100;++z){if(usejit)d.run(fn);else d.blas();}
  (usejit?jit:blas).push_back(elapsed(start,Clock::now())/100);d.check();sink+=d.c[r%d.c.size()];
 }
 std::sort(jit.begin(),jit.end());std::sort(blas.begin(),blas.end());
 printf("{\"mode\":\"exec\",\"m\":%d,\"n\":%d,\"k\":%d,\"target\":\"%s\",\"code_bytes\":%zu,\"create_us\":%.6f,\"jit_us\":%.6f,\"blas_us\":%.6f,\"speedup\":%.6f,\"max_error\":%.9f}\n",m,n,k,libxsmm_get_target_arch(),i.code_size,elapsed(t0,t1),jit[7],blas[7],blas[7]/jit[7],err);
}
void cache_case(int cap,int rounds){
 if(cap<0 || rounds<1 || rounds>100000)throw std::invalid_argument("invalid capacity/rounds");
 std::vector<Data> ds;for(int m:{128,144,160,176,192,208,224,240})ds.emplace_back(m,128,512);
 std::list<std::pair<int,libxsmm_gemmfunction>> cache;
 double create=0,execute=0,total=0,checksum=0;size_t miss=0,hit=0,bytes=0;
 auto request=[&](int idx,bool measure){auto& d=ds[idx];auto start=Clock::now();
  auto found=std::find_if(cache.begin(),cache.end(),[&](const auto&x){return x.first==idx;});libxsmm_gemmfunction fn;
  if(found!=cache.end()){fn=found->second;cache.splice(cache.begin(),cache,found);if(measure)++hit;}
  else{auto before=registry_size();fn=dispatch(d.m,d.n,d.k);if(registry_size()!=before+1)throw std::runtime_error("outer miss hidden by backend cache");
   if(measure){++miss;bytes+=info(fn).code_size;}
   if(cap){cache.emplace_front(idx,fn);if(int(cache.size())>cap){release(cache.back().second);cache.pop_back();}}
  }
  auto ready=Clock::now();d.run(fn);auto done=Clock::now();if(!cap)release(fn);auto end=Clock::now();
  if(measure){create+=elapsed(start,ready);execute+=elapsed(ready,done);total+=elapsed(start,end);}
  d.check();checksum+=d.c[0];
 };
 for(int i=0;i<8;++i)request(i,false);checksum=0;
 for(int r=0;r<rounds;++r)for(int i=0;i<8;++i)request(i,true);
 size_t resident=registry_size();for(auto&x:cache)release(x.second);if(registry_size()!=0)throw std::runtime_error("kernel deregistration requires LIBXSMM_ENABLE_DEREG");
 printf("{\"mode\":\"cache\",\"capacity\":%d,\"requests\":%d,\"hits\":%zu,\"misses\":%zu,\"resident\":%zu,\"generated_bytes\":%zu,\"prepare_us\":%.6f,\"execute_us\":%.6f,\"total_us\":%.6f,\"checksum\":%.9f}\n",cap,rounds*8,hit,miss,resident,bytes,create,execute,total,checksum);
}
void generator_case(int m,int n,int k){
 auto shape=libxsmm_create_gemm_shape(m,n,k,m,k,m,LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32,LIBXSMM_DATATYPE_F32);
 libxsmm_descriptor_blob blob{};auto desc=libxsmm_gemm_descriptor_init_gemm(&blob,shape,LIBXSMM_GEMM_FLAG_BETA_0,LIBXSMM_GEMM_PREFETCH_NONE);
 std::vector<unsigned char> buffer(2*1024*1024);double total=0;unsigned bytes=0;
 for(int j=0;j<810;++j){libxsmm_generated_code code{};code.generated_code=buffer.data();code.buffer_size=buffer.size();code.arch=libxsmm_get_target_archid();code.code_type=2;
  auto start=Clock::now();libxsmm_generator_gemm_kernel(&code,desc);auto end=Clock::now();
  if(code.last_error)throw std::runtime_error("generator failed");if(j>=10)total+=elapsed(start,end);bytes=code.code_size+code.data_size;
 }
 auto fn=dispatch(m,n,k);auto i=info(fn);bool same=i.code_size==bytes && !std::memcmp(buffer.data(),reinterpret_cast<const void*>(fn),bytes);
 if(!same)throw std::runtime_error("generator bytes differ from dispatch");
 printf("{\"mode\":\"generator\",\"m\":%d,\"n\":%d,\"k\":%d,\"count\":800,\"code_bytes\":%u,\"same_bytes\":true,\"emit_us\":%.6f,\"per_emit_us\":%.6f}\n",m,n,k,bytes,total,total/800);
}
void verify_case(){
 double max_error=0;
 for(int size:{128,129,144,176,256}){
  Data d(size,size,512);auto fn=dispatch(d.m,d.n,d.k);info(fn);
  for(int pass=0;pass<2;++pass){
   d.run(fn);d.check();
   for(int j=0;j<d.n;++j)for(int i=0;i<d.m;++i){double ref=0;
    for(int q=0;q<d.k;++q)ref+=double(d.a[i+q*d.m])*d.b[q+j*d.k];
    double error=std::abs(d.c[i+j*d.m]-ref);max_error=std::max(error,max_error);
    if(error>2e-4*(1+std::abs(ref)))throw std::runtime_error("double scalar reference mismatch");
   }
   d.a[0]+=.31f;d.b[0]-=.27f;d.blas();d.ref=d.c;
  }
 }
 printf("PASS full double scalar reference and changing A/B: M=N 128,129,144,176,256 K512 max_abs_error %.9f\n",max_error);
}
int main(int argc,char**argv)try{
 libxsmm_init();if(argc==2 && std::string(argv[1])=="verify")verify_case();
 else if(argc>=5 && std::string(argv[1])=="exec")exec_case(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),argc>5?std::stoi(argv[5]):0);
 else if(argc==5 && std::string(argv[1])=="generator")generator_case(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]));
 else if(argc==4 && std::string(argv[1])=="cache")cache_case(std::stoi(argv[2]),std::stoi(argv[3]));
 else throw std::runtime_error("usage: exec M N K [order_seed] | cache capacity rounds");
}catch(const std::exception&e){fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
