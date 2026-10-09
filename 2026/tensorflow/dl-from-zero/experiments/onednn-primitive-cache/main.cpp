#include <oneapi/dnnl/dnnl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace dnnl;
using Clock = std::chrono::steady_clock;
static double us(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}
static uint16_t to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fff + ((bits >> 16) & 1); // Round to nearest, ties to even.
    return uint16_t(bits >> 16);
}
static float from_bf16(uint16_t value) {
    uint32_t bits = uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
struct Work {
    int m, k, n;
    bool bf_input, bf_output;
    std::vector<float> a, b, c;
    std::vector<uint16_t> ab, bb, cb;
    memory::desc ad, bd, cd;
    memory am, bm, cm;
    std::unordered_map<int, memory> args;
    Work(int M, int K, int N, const engine& e, const std::string& dtype)
        : m(M), k(K), n(N), bf_input(dtype!="f32"), bf_output(dtype=="bf16"),
          a(M*K), b(K*N), c(M*N), ab(M*K), bb(K*N), cb(M*N),
          ad({M,K}, bf_input ? memory::data_type::bf16 : memory::data_type::f32, memory::format_tag::ab),
          bd({K,N}, bf_input ? memory::data_type::bf16 : memory::data_type::f32, memory::format_tag::ab),
          cd({M,N}, bf_output ? memory::data_type::bf16 : memory::data_type::f32, memory::format_tag::ab),
          am(ad,e,bf_input ? static_cast<void*>(ab.data()) : a.data()),
          bm(bd,e,bf_input ? static_cast<void*>(bb.data()) : b.data()),
          cm(cd,e,bf_output ? static_cast<void*>(cb.data()) : c.data()),
          args{{DNNL_ARG_SRC,am},{DNNL_ARG_WEIGHTS,bm},{DNNL_ARG_DST,cm}} {
        for (size_t i=0; i<a.size(); ++i) a[i]=float(int(i%17)-8)/16;
        for (size_t i=0; i<b.size(); ++i) b[i]=float(int(i%11)-5)/8;
        for (size_t i=0; i<a.size(); ++i) { ab[i]=to_bf16(a[i]); if(bf_input) a[i]=from_bf16(ab[i]); }
        for (size_t i=0; i<b.size(); ++i) { bb[i]=to_bf16(b[i]); if(bf_input) b[i]=from_bf16(bb[i]); }
    }
    float output(int cell) const { return bf_output ? from_bf16(cb[cell]) : c[cell]; }
    double verify(bool full) const {
        double max_error = 0;
        const int count = m*n;
        for (int idx=0; idx<(full ? count : 3); ++idx) {
            int cell = full ? idx : (idx==0 ? 0 : idx==1 ? count/2 : count-1);
            int row=cell/n, col=cell%n;
            double expected=0;
            for (int j=0; j<k; ++j) expected += double(a[row*k+j])*b[j*n+col];
            if(bf_output) expected=from_bf16(to_bf16(float(expected)));
            double err=std::abs(double(output(cell))-expected);
            if (!std::isfinite(output(cell)) || err > 1e-4+1e-4*std::abs(expected))
                throw std::runtime_error("incorrect result, M="+std::to_string(m));
            max_error=std::max(max_error,err);
        }
        return max_error;
    }
};
struct Sample { double pd, primitive, execute, total; };
static double percentile(std::vector<double> x, double q) {
    std::sort(x.begin(), x.end());
    return x[std::min(x.size()-1, size_t(std::ceil(q*x.size()))-1)];
}
int main(int argc, char** argv) try {
    // Positional: mode capacity rounds K N. Each process owns a fresh cache.
    if (argc != 6 && argc != 7) throw std::runtime_error("usage: cache_case fixed|mixed capacity rounds K N [f32|bf16|bf16-f32]");
    std::string dtype=argc==7 ? argv[6] : "f32";
    if(dtype!="f32" && dtype!="bf16" && dtype!="bf16-f32") throw std::runtime_error("invalid dtype");
    std::string mode=argv[1];
    int capacity=std::stoi(argv[2]), rounds=std::stoi(argv[3]);
    int k=std::stoi(argv[4]), n=std::stoi(argv[5]);
    if ((mode!="fixed" && mode!="mixed") || capacity<0 || rounds<1 || rounds>10000 || k<1 || n<1 || k>4096 || n>4096)
        throw std::runtime_error("invalid arguments");
    set_primitive_cache_capacity(capacity);
    engine e(engine::kind::cpu,0);
    stream s(e);
    std::vector<int> shapes = mode=="fixed" ? std::vector<int>{40} : std::vector<int>{16,24,32,40,48,56,64,72};
    std::vector<std::unique_ptr<Work>> works;
    for (int m: shapes) works.push_back(std::make_unique<Work>(m,k,n,e,dtype));
    auto version=dnnl_version();
    std::cout << std::unitbuf;
    std::cout << "CONFIG version=" << version->major << '.' << version->minor << '.' << version->patch
              << " hash=" << version->hash << " cpu_runtime=" << version->cpu_runtime
              << " capacity=" << get_primitive_cache_capacity() << " mode=" << mode << " dtype=" << dtype << '\n';
    std::vector<Sample> samples;
    double max_error=0, checksum=0;
    auto request = [&](Work& w, int sequence, bool measured) {
        w.b[0]=float(sequence%19)/16; // New data, same primitive description.
        w.bb[0]=to_bf16(w.b[0]);
        if(w.bf_input) w.b[0]=from_bf16(w.bb[0]);
        Sample x{};
        std::string impl;
        auto begin=Clock::now();
        {
            matmul::primitive_desc pd(e,w.ad,w.bd,w.cd);
            auto after_pd=Clock::now();
            matmul primitive(pd);
            auto prepared=Clock::now();
            primitive.execute(s,w.args);
            s.wait();
            auto executed=Clock::now();
            x.pd=us(begin,after_pd); x.primitive=us(after_pd,prepared); x.execute=us(prepared,executed);
            if (!measured) impl=pd.impl_info_str();
        }
        x.total=us(begin,Clock::now());
        max_error=std::max(max_error,w.verify(!measured));
        checksum+=w.output(0);
        if (measured) samples.push_back(x);
        else std::cout << "WARM M=" << w.m << " impl=" << impl << " prepare_us=" << x.pd+x.primitive << '\n';
    };
    for (size_t i=0;i<works.size();++i) request(*works[i],int(i),false);
    std::cout << "MEASURE_BEGIN\n";
    int sequence=0;
    for (int r=0;r<rounds;++r)
        for (auto& w:works) request(*w,sequence++,true);
    std::cout << "MEASURE_END\n";
    // Full verification again after changing weights repeatedly.
    for (auto& w:works) max_error=std::max(max_error,w->verify(true));
    double pd=0, primitive=0, execute=0, total=0;
    std::vector<double> latency;
    for(auto x:samples) { pd+=x.pd; primitive+=x.primitive; execute+=x.execute; total+=x.total; latency.push_back(x.total); }
    std::cout << std::fixed << std::setprecision(6)
        << "RESULT {\"requests\":" << samples.size() << ",\"pd_ms\":" << pd/1000
        << ",\"primitive_ms\":" << primitive/1000 << ",\"prepare_ms\":" << (pd+primitive)/1000
        << ",\"execute_ms\":" << execute/1000 << ",\"total_ms\":" << total/1000
        << ",\"p50_us\":" << percentile(latency,.5) << ",\"p95_us\":" << percentile(latency,.95)
        << ",\"max_error\":" << max_error << ",\"checksum\":" << checksum << "}\n";
    return 0;
} catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
