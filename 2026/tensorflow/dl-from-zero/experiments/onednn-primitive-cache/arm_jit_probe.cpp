// Probe real AArch64 JIT implementations, independently of the MatMul case.
#include <oneapi/dnnl/dnnl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
using namespace dnnl;
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double,std::milli>(b-a).count();
}
struct Work {
    int h;
    std::string op;
    memory::desc src_desc, dst_desc;
    memory src, dst;
    std::vector<float> mean, variance;
    memory mean_mem, variance_mem;
    std::unordered_map<int,memory> args;
    Work(const engine& e, std::string operation, int height)
        : h(height), op(operation),
          src_desc(op=="reorder" ? memory::desc({32,h*8},memory::data_type::f32,memory::format_tag::ab)
                                    : memory::desc({2,32,h,8},memory::data_type::f32,memory::format_tag::nChw8c)),
          dst_desc(op=="reorder" ? memory::desc({32,h*8},memory::data_type::f32,memory::format_tag::ba) : src_desc),
          src(src_desc,e),dst(dst_desc,e),mean(32,0),variance(32,1),
          mean_mem(memory::desc({32},memory::data_type::f32,memory::format_tag::x),e,mean.data()),
          variance_mem(memory::desc({32},memory::data_type::f32,memory::format_tag::x),e,variance.data()) {
        auto* values=static_cast<float*>(src.get_data_handle());
        for(size_t i=0;i<src_desc.get_size()/sizeof(float);++i) values[i]=float(i)/512;
        args = op=="reorder" ? std::unordered_map<int,memory>{{DNNL_ARG_FROM,src},{DNNL_ARG_TO,dst}}
                             : std::unordered_map<int,memory>{{DNNL_ARG_SRC,src},{DNNL_ARG_DST,dst}};
        if(op=="batchnorm") { args.emplace(DNNL_ARG_MEAN,mean_mem); args.emplace(DNNL_ARG_VARIANCE,variance_mem); }
    }
    size_t offset(int n,int c,int spatial) const {
        return ((n*4+c/8)*(h*8)+spatial)*8+c%8;
    }
    void verify() const {
        auto* input=static_cast<const float*>(src.get_data_handle());
        auto* output=static_cast<const float*>(dst.get_data_handle());
        auto check=[](float actual,float expected) {
            if(!std::isfinite(actual) || std::abs(actual-expected)>1e-4f+1e-4f*std::abs(expected))
                throw std::runtime_error("incorrect output");
        };
        if(op=="reorder") {
            for(int i=0;i<32;++i) for(int j=0;j<h*8;++j) check(output[j*32+i],input[i*h*8+j]);
        } else if(op=="batchnorm") {
            for(size_t i=0;i<src_desc.get_size()/sizeof(float);++i) check(output[i],input[i]/std::sqrt(1.00001f));
        } else {
            // oneDNN maps output c to input (c % (C/group)) * group + c/(C/group).
            for(int n=0;n<2;++n) for(int c=0;c<32;++c) for(int s=0;s<h*8;++s)
                check(output[offset(n,c,s)],input[offset(n,(c%8)*4+c/8,s)]);
        }
    }
};
int main(int argc,char** argv) try {
    if(argc!=3) throw std::runtime_error("usage: arm_jit_probe batchnorm|shuffle|reorder capacity");
    std::string op=argv[1];
    if(op!="batchnorm" && op!="shuffle" && op!="reorder") throw std::runtime_error("unknown op");
    set_primitive_cache_capacity(std::stoi(argv[2]));
    engine e(engine::kind::cpu,0); stream stream(e);
    std::vector<std::unique_ptr<Work>> work;
    for(int h=8;h<16;++h) work.push_back(std::make_unique<Work>(e,op,h));
    double pd_ms=0,create_ms=0,execute_ms=0,total_ms=0;
    std::cout << std::unitbuf;
    for(int round=0;round<=20;++round) {
        if(round==1) std::cout << "MEASURE_BEGIN\n";
        for(auto& w:work) {
            auto begin=Clock::now();
            double pd_time=0,create_time=0,exec_time=0;
            {
                primitive prim;
                auto run=[&](auto&& pd,auto make_primitive) {
                    auto descriptor_done=Clock::now();
                    prim=make_primitive(pd);
                    auto created=Clock::now();
                    if(round==0) std::cout << "SELECTED op=" << op << " impl=" << pd.impl_info_str() << '\n';
                    prim.execute(stream,w->args); stream.wait();
                    auto executed=Clock::now();
                    pd_time=ms(begin,descriptor_done); create_time=ms(descriptor_done,created); exec_time=ms(created,executed);
                    if(round==0) std::cout << "WARM op=" << op << " H=" << w->h << " impl=" << pd.impl_info_str()
                                          << " create_ms=" << create_time << '\n';
                };
                if(op=="batchnorm") run(batch_normalization_forward::primitive_desc(e,prop_kind::forward_inference,
                    w->src_desc,w->dst_desc,1e-5f,normalization_flags::use_global_stats),
                    [](const auto& pd){return batch_normalization_forward(pd);});
                else if(op=="shuffle") run(shuffle_forward::primitive_desc(e,prop_kind::forward_inference,w->src_desc,w->dst_desc,1,4),
                    [](const auto& pd){return shuffle_forward(pd);});
                else run(reorder::primitive_desc(e,w->src_desc,e,w->dst_desc),[](const auto& pd){return reorder(pd);});
            }
            double total=ms(begin,Clock::now());
            if(round>0) { pd_ms+=pd_time;create_ms+=create_time;execute_ms+=exec_time;total_ms+=total; }
            w->verify(); // Outside timing; also validates that generated code executes.
        }
    }
    std::cout << "MEASURE_END\nRESULT op=" << op << " capacity=" << get_primitive_cache_capacity()
              << " requests=160 pd_ms=" << pd_ms << " primitive_ms=" << create_ms << " execute_ms=" << execute_ms
              << " total_ms=" << total_ms << " correctness=PASS\n";
} catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
