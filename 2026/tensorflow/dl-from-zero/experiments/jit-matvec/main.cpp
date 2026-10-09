#include "jit_kernel.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <list>
#include <random>
#include <string>
extern "C" void neon_generic(const float*,const float*,float*,int,int);
extern "C" void accelerate(const float*,const float*,float*,int,int);
extern "C" void tf_existing(const float*,const float*,float*,int,int);
extern "C" int tf_has_sme2();
using namespace demo_jit;
extern "C" Entry aot_entry(int,int);
volatile double sink=0;
struct Data {
    int m,k;
    std::vector<float>a,b,c;
    Data(int rows,int cols):m(rows),k(cols),a(m*k),b(k),c(m+4,-9876) {
        std::mt19937 rng(m*1000+k);
        std::uniform_real_distribution<float> dist(-1,1);
        for(auto& x:a)x=dist(rng);for(auto& x:b)x=dist(rng);
    }
    void run(Entry fn){fn(a.data(),b.data(),c.data(),m,k);}
    void check() const {
        for(int i=0;i<m;++i){double ref=0;for(int j=0;j<k;++j)ref+=double(a[i*k+j])*b[j];
            if(!std::isfinite(c[i]) || std::abs(c[i]-ref)>2e-5*(1+std::abs(ref)))throw std::runtime_error("wrong output");}
        for(int i=m;i<m+4;++i)if(c[i]!=-9876)throw std::runtime_error("output overrun");
    }
};
void verify_all() {
    for(int m=4;m<=32;m+=4)for(int k=4;k<=256;k+=4){
        Data d(m,k);auto code=build(m,k);
        for(int pass=0;pass<2;++pass){d.run(code->entry());d.check();d.run(neon_generic);d.check();
            d.a[0]+=.137f;d.b[0]-=.291f;}
    }
    for(auto shape:std::vector<std::pair<int,int>>{{0,16},{3,16},{4,0},{4,257},{36,16}}){
        bool rejected=false;try{build(shape.first,shape.second);}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("invalid shape accepted");
    }
    puts("PASS 512 supported shapes, changed inputs/weights, output guards, invalid shapes");
}
void execution(int m,int k,int seed,const std::string& dump) {
    auto code=build(m,k);Data d(m,k);
    if(!dump.empty()){std::ofstream f(dump,std::ios::binary);f.write(reinterpret_cast<const char*>(code->getCode()),code->getSize());if(!f)throw std::runtime_error("dump failed");}
    std::vector<Entry> fns{neon_generic,code->entry(),accelerate,tf_existing};
    std::vector<const char*> names{"neon","jit","accelerate","tf_existing"};
    if(auto aot=aot_entry(m,k)){fns.push_back(aot);names.push_back("aot_specialized");}
    std::vector<std::vector<double>> timings(fns.size());
    for(auto fn:fns){d.run(fn);d.check();}
    for(int round=0;round<21;++round)for(size_t step=0;step<fns.size();++step){
        const size_t idx=(step+round+seed)%fns.size();const Entry fn=fns[idx];
        for(int i=0;i<2000;++i)d.run(fn);
        constexpr int count=200000;
        const auto begin=Clock::now();for(int i=0;i<count;++i)d.run(fn);
        timings[idx].push_back(us(begin,Clock::now())*1000/count);
        d.check();sink+=d.c[round%m];
        d.b[0]=float(round%7-3)/9;
    }
    printf("{\"mode\":\"exec\",\"m\":%d,\"k\":%d,\"code_bytes\":%zu,\"tf_sme2\":%d,\"first_allocate_other_us\":%.6f,\"first_emit_us\":%.6f,\"first_publish_us\":%.6f,\"ns\":{",m,k,code->getSize(),tf_has_sme2(),code->times.allocate_us,code->times.emit_us,code->times.publish_us);
    for(size_t i=0;i<fns.size();++i){auto& t=timings[i];std::sort(t.begin(),t.end());printf("%s\"%s\":%.6f",i?",":"",names[i],t[t.size()/2]);}puts("}}");
}
void lifecycle(int m,int k,int count) {
    Data d(m,k);BuildTimes times;double first=0,warm=0,destroy=0,total=0;
    for(int i=0;i<count+10;++i){
        auto t0=Clock::now();auto code=build(m,k);auto t1=Clock::now();
        d.run(code->entry());auto t2=Clock::now();
        d.run(code->entry());auto t3=Clock::now();
        auto bt=code->times;code.reset();auto t4=Clock::now();
        if(i>=10){times.allocate_us+=bt.allocate_us;times.emit_us+=bt.emit_us;times.publish_us+=bt.publish_us;
            first+=us(t1,t2);warm+=us(t2,t3);destroy+=us(t3,t4);total+=us(t0,t4);}
        d.check();d.b[0]=float(i%7)/11;
    }
    printf("{\"mode\":\"lifecycle\",\"m\":%d,\"k\":%d,\"count\":%d,\"allocate_other_us\":%.6f,\"emit_us\":%.6f,\"publish_us\":%.6f,\"first_execute_us\":%.6f,\"warm_execute_us\":%.6f,\"destroy_us\":%.6f,\"total_us\":%.6f}\n",m,k,count,times.allocate_us,times.emit_us,times.publish_us,first,warm,destroy,total);
}
// Experiment-only code retention. Stage 3 will supply the actual primitive API.
void cache_case(int capacity_,int rounds,bool fixed) {
    if(capacity_<0 || rounds<1)throw std::invalid_argument("bad cache arguments");
    std::vector<Data> data;
    for(int k:{8,12,16,20,24,28,32,36})data.emplace_back(8,k);
    std::list<std::pair<int,std::shared_ptr<Kernel>>> retained;
    size_t hits=0,misses=0,bytes_generated=0;BuildTimes build_times;
    double prepare=0,execute=0,total=0,checksum=0;
    auto request=[&](int idx,bool measure){
        auto& d=data[idx];d.b[0]=float((idx+int(checksum))%7)/11;
        const auto t0=Clock::now();
        auto it=std::find_if(retained.begin(),retained.end(),[&](const auto& item){return item.first==idx;});
        std::shared_ptr<Kernel> code;
        if(it!=retained.end()){
            code=it->second;retained.splice(retained.begin(),retained,it);if(measure)++hits;
        }else{
            code=build(d.m,d.k);
            if(measure){++misses;bytes_generated+=code->getSize();build_times.allocate_us+=code->times.allocate_us;build_times.emit_us+=code->times.emit_us;build_times.publish_us+=code->times.publish_us;}
            if(capacity_){retained.emplace_front(idx,code);if(retained.size()>size_t(capacity_))retained.pop_back();}
        }
        const auto t1=Clock::now();d.run(code->entry());const auto t2=Clock::now();
        code.reset();const auto t3=Clock::now();
        if(measure){prepare+=us(t0,t1);execute+=us(t1,t2);total+=us(t0,t3);}
        d.check();checksum+=d.c[0];
    };
    for(int i=0;i<(fixed?1:8);++i)request(i,false);
    checksum=0;
    for(int r=0;r<rounds;++r)for(int i=0;i<8;++i)request(fixed?0:i,true);
    size_t resident=0;for(const auto& item:retained)resident+=item.second->getSize();
    printf("{\"mode\":\"%s\",\"capacity\":%d,\"requests\":%d,\"hits\":%zu,\"misses\":%zu,\"generated_bytes\":%zu,\"resident_code_bytes\":%zu,\"allocate_other_us\":%.6f,\"emit_us\":%.6f,\"publish_us\":%.6f,\"prepare_us\":%.6f,\"execute_us\":%.6f,\"total_us\":%.6f,\"checksum\":%.9f}\n",fixed?"fixed":"mixed",capacity_,rounds*8,hits,misses,bytes_generated,resident,build_times.allocate_us,build_times.emit_us,build_times.publish_us,prepare,execute,total,checksum);
}
int main(int argc,char** argv) try {
    const std::string mode=argc>1?argv[1]:"verify";
    if(mode=="verify")verify_all();
    else if(mode=="exec" && argc>=4)execution(std::stoi(argv[2]),std::stoi(argv[3]),argc>4?std::stoi(argv[4]):0,argc>5?argv[5]:"");
    else if(mode=="lifecycle" && argc==5)lifecycle(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]));
    else if((mode=="mixed"||mode=="fixed") && argc==4)cache_case(std::stoi(argv[2]),std::stoi(argv[3]),mode=="fixed");
    else throw std::invalid_argument("usage: verify | exec M K [order_seed [dump.bin]] | mixed|fixed capacity rounds");
}catch(const std::exception& e){fprintf(stderr,"ERROR %s\n",e.what());return 1;}
