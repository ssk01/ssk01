#include "match.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace matching;
using Rows=std::vector<std::string>;
using Fn=bool(*)(View,const Pattern&);
bool scalar_p(View s,const Pattern& p){return scalar_prefix(s,p.bytes);}
bool scalar_s(View s,const Pattern& p){return scalar_suffix(s,p.bytes);}
bool scalar_c(View s,const Pattern& p){return scalar_contains(s,p.bytes);}
bool simd_c(View s,const Pattern& p){return simd_contains(s,p.bytes);}
bool libc_c(View s,const Pattern& p){return libc_contains(s,p.bytes);}
volatile uint64_t sink=0;

void trace(View s, View n) {
    std::cout<<"text=["<<s<<"] needle=["<<n<<"] backend="<<backend()<<"\n";
    std::cout<<"prefix: only start 0 -> "<<scalar_prefix(s,n)<<"\n";
    if(s.size()>=n.size()) std::cout<<"suffix: only start "<<s.size()-n.size()<<" -> "<<scalar_suffix(s,n)<<"\n";
    else std::cout<<"suffix: text too short -> 0\n";
    if(n.empty()){std::cout<<"empty needle matches at 0\n";return;}
    if(s.size()<n.size()){std::cout<<"no legal start\n";return;}
    size_t starts=s.size()-n.size()+1;
    std::cout<<"scalar contains: try j=0,1,...; compare n[k] at s[j+k]\n";
    for(size_t j=0;j<starts;++j){
        std::cout<<" j="<<j<<":";
        bool ok=true;
        for(size_t k=0;k<n.size();++k){std::cout<<" "<<s[j+k]<<"/"<<n[k];if(s[j+k]!=n[k]){ok=false;break;}}
        std::cout<<(ok?" MATCH\n":" fail\n");if(ok)break;
    }
    std::cout<<"SIMD contains: each lane is a DIFFERENT start; all lanes compare with n[0]='"<<n[0]<<"'\n";
    for(size_t j=0;j<starts;j+=16){
        // Trace uses padding for the final diagram; the timed kernel uses a scalar tail.
        alignas(16) char block[16]={};size_t valid=std::min(size_t(16),starts-j);
        std::memcpy(block,s.data()+j,valid);
        uint16_t mask=first_mask(block,n[0]);
        if(valid<16)mask&=static_cast<uint16_t>((1u<<valid)-1);
        std::cout<<" block j="<<j<<(valid<16?" [padded diagram; timed implementation uses scalar tail]":"")<<"\n";
        std::cout<<" lane: ";for(size_t k=0;k<valid;++k)std::cout<<std::setw(3)<<k;std::cout<<"\n byte: ";
        for(size_t k=0;k<valid;++k)std::cout<<std::setw(3)<<block[k];std::cout<<"\n hit:  ";
        for(size_t k=0;k<valid;++k)std::cout<<std::setw(3)<<((mask>>k)&1);std::cout<<"\n";
        std::cout<<" candidate mask=0x"<<std::hex<<mask<<std::dec<<" (bit 0 = lane 0)\n";
        while(mask){unsigned lane=__builtin_ctz(static_cast<unsigned>(mask));size_t pos=j+lane;
            bool ok=scalar_equal(s.substr(pos+1,n.size()-1),n.substr(1));
            std::cout<<" verify start=j+lane="<<j<<"+"<<lane<<"="<<pos<<", remaining ["<<s.substr(pos+1,n.size()-1)<<"] vs ["<<n.substr(1)<<"] -> "<<(ok?"MATCH":"fail")<<"\n";
            if(ok){std::cout<<" return true for this row; benchmark increments hit count and proceeds to NEXT ROW\n";return;}
            mask&=static_cast<uint16_t>(mask-1);
            std::cout<<" clear lowest candidate bit: mask &= mask - 1 -> 0x"<<std::hex<<mask<<std::dec<<"; try remaining candidates in THIS block\n";
        }
        std::cout<<" advance block j += 16\n";
    }
}
Rows lines(const std::string& path){std::ifstream f(path);if(!f)throw std::runtime_error("cannot open text file");Rows r;std::string s;while(std::getline(f,s))r.push_back(s);return r;}
Rows blob_rows(const std::string& blob,const std::string& lens,size_t cap){
    std::ifstream b(blob,std::ios::binary),l(lens,std::ios::binary);if(!b||!l)throw std::runtime_error("cannot open blob/lens");
    Rows r;size_t total=0;unsigned char raw[4];
    while(l.read(reinterpret_cast<char*>(raw),4)){
        uint32_t n=uint32_t(raw[0])|(uint32_t(raw[1])<<8)|(uint32_t(raw[2])<<16)|(uint32_t(raw[3])<<24);
        if(n>cap-total)break;
        std::string s(n,'\0');if(!b.read(s.data(),n))throw std::runtime_error("truncated blob");total+=n;r.push_back(std::move(s));
    }
    return r;
}
__attribute__((noinline)) uint64_t run(const Rows& rows,const Pattern& n,Fn fn){uint64_t h=0;for(const auto& s:rows)h+=fn(s,n);return h;}
__attribute__((noinline)) uint64_t read_all(const Rows& rows){
    uint64_t sum=0;
#if defined(__ARM_NEON)
    uint8x16_t acc=vdupq_n_u8(0);
#else
    __m128i acc=_mm_setzero_si128();
#endif
    for(const auto& s:rows){size_t j=0;
        for(;s.size()-j>=16;j+=16){
#if defined(__ARM_NEON)
            acc=veorq_u8(acc,vld1q_u8(reinterpret_cast<const uint8_t*>(s.data()+j)));
#else
            acc=_mm_xor_si128(acc,_mm_loadu_si128(reinterpret_cast<const __m128i*>(s.data()+j)));
#endif
        }
        for(;j<s.size();++j)sum+=static_cast<unsigned char>(s[j]);
    }
    alignas(16) uint8_t lane[16];
#if defined(__ARM_NEON)
    vst1q_u8(lane,acc);
#else
    _mm_storeu_si128(reinterpret_cast<__m128i*>(lane),acc);
#endif
    for(auto x:lane)sum+=x;return sum;
}
// Median of five batches; each batch lasts at least ~20 ms based on a pilot.
template<class F> double measure(F fn){
    using Clock=std::chrono::steady_clock;auto t=Clock::now();sink=fn();
    double pilot=std::chrono::duration<double>(Clock::now()-t).count();
    size_t repeats=std::min(size_t(1000),size_t(.02/std::max(pilot,1e-9))+1);
    std::vector<double> times;
    for(int k=0;k<5;++k){auto start=Clock::now();for(size_t q=0;q<repeats;++q)sink=fn();times.push_back(std::chrono::duration<double>(Clock::now()-start).count()/repeats);}
    std::sort(times.begin(),times.end());return times[2];
}
struct ScalarWork { uint64_t starts=0, compares=0, hits=0; };
// Count actual byte comparisons in a separate pass, never in timed kernels.
ScalarWork scalar_work(const Rows& rows, View n, int shape) {
    ScalarWork w;
    for (const auto& s: rows) {
        if (s.size()<n.size()) continue;
        auto equal_at=[&](size_t j) {
            ++w.starts;
            for(size_t k=0;k<n.size();++k) { ++w.compares; if(s[j+k]!=n[k]) return false; }
            return true;
        };
        if(shape==0) w.hits+=equal_at(0);
        else if(shape==1) w.hits+=equal_at(s.size()-n.size());
        else for(size_t j=0;j<=s.size()-n.size();++j) if(equal_at(j)) { ++w.hits; break; }
    }
    return w;
}
void bench(const Rows& rows,const std::vector<std::string>& needles){
    size_t bytes=0;for(const auto& s:rows)bytes+=s.size();
    if(rows.empty()||bytes==0)throw std::runtime_error("benchmark needs nonempty corpus");
    std::cout<<"# backend="<<backend()<<" rows="<<rows.size()<<" text_bytes="<<bytes<<" statistic=median_of_5_batches\n";
    double rd=measure([&]{return read_all(rows);});
    std::cout<<"# read-all checksum reference_ms="<<rd*1000<<" logical_GBps="<<bytes/rd/1e9<<" (not a hardware bandwidth ceiling)\n";
    std::cout<<"needle,shape,impl,hits,ms,logical_GBps,Mrows,blocks,active_blocks,candidates,tail_starts,scalar_starts,scalar_compares\n";
    const char* shapes[]={"prefix","suffix","contains"};
    Fn scalar[]={scalar_p,scalar_s,scalar_c};Fn simd[]={simd_prefix,simd_suffix,simd_c};
    for(const auto& n:needles)for(int shape=0;shape<3;++shape){
        Pattern prepared(n);
        uint64_t expect=0;for(const auto& s:rows){View v=s;expect+=shape==0?(v.size()>=n.size()&&v.substr(0,n.size())==n):shape==1?(v.size()>=n.size()&&v.substr(v.size()-n.size())==n):v.find(n)!=View::npos;}
        const auto work=scalar_work(rows,n,shape);
        if(work.hits!=expect)throw std::runtime_error("work counter mismatch");
        for(int impl=0;impl<(shape==2?3:2);++impl){Fn fn=impl==0?scalar[shape]:impl==1?simd[shape]:libc_c;
            uint64_t h=run(rows,prepared,fn);if(h!=expect)throw std::runtime_error("correctness mismatch");
            double sec=measure([&]{return run(rows,prepared,fn);});Stats st;
            if(shape==2&&impl==1)for(const auto& s:rows)simd_contains<true>(s,n,&st);
            std::cout<<n<<","<<shapes[shape]<<","<<(impl==0?"scalar":impl==1?"simd":"memchr")<<","<<h<<","<<std::fixed<<std::setprecision(4)<<sec*1000<<","<<bytes/sec/1e9<<","<<rows.size()/sec/1e6<<","<<st.blocks<<","<<st.active_blocks<<","<<st.candidates<<","<<st.tail_starts<<","<<work.starts<<","<<work.compares<<"\n";
        }
    }
}
int main(int argc,char**argv){try{
    std::string mode=argc>1?argv[1]:"help";
    if(mode=="trace"&&argc==4){trace(argv[2],argv[3]);return 0;}
    Rows rows;int first;
    if(mode=="lines"&&argc>=3){rows=lines(argv[2]);first=3;}
    else if(mode=="blob"&&argc>=4){rows=blob_rows(argv[2],argv[3],48*1024*1024);first=4;}
    else throw std::runtime_error("usage: demo trace text needle | lines file [needles...] | blob text.bin lengths.bin [needles...]; use python3 run.py for bundled real data");
    std::vector<std::string> needles;for(int i=first;i<argc;++i)needles.emplace_back(argv[i]);if(needles.empty())needles={"youtube"};bench(rows,needles);
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
