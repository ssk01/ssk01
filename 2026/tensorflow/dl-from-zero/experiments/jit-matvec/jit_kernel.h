#pragma once
#include <xbyak_aarch64/xbyak_aarch64.h>
#include <chrono>
#include <memory>
#include <stdexcept>

namespace demo_jit {
using Clock = std::chrono::steady_clock;
inline double us(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b-a).count();
}
using Entry = void(*)(const float*, const float*, float*, int, int);
struct BuildTimes { double allocate_us=0, emit_us=0, publish_us=0; };

// The generator emits actual AArch64 instructions. Xbyak only encodes them;
// neither oneDNN primitives nor a compiler subprocess are used.
inline size_t capacity(int m,int k) {
    if (m < 4 || m > 32 || m % 4 || k < 4 || k > 256 || k % 4)
        throw std::invalid_argument("supported: M=4..32 step4, K=4..256 step4, FP32 contiguous GEMV");
    return 4*(m/4*(16+9*(k/4))+1)+64;
}
struct Kernel : Xbyak_aarch64::CodeGenerator {
    BuildTimes times;
    int m, k;
    Kernel(int rows, int cols) : CodeGenerator(capacity(rows,cols), Xbyak_aarch64::DontSetProtectRWE), m(rows), k(cols) {
        if (m < 4 || m > 32 || m % 4 || k < 4 || k > 256 || k % 4)
            throw std::invalid_argument("supported: M=4..32 step4, K=4..256 step4, FP32 contiguous GEMV");
        const auto start = Clock::now();
        for (int i=0; i<m; i+=4) {
            for (int r=0; r<4; ++r) {
                const Xbyak_aarch64::VReg v(16+r);
                movi(v.d2, 0);
            }
            for (int j=0; j<k; j+=4) {
                ldr(q0, ptr(x1, j*4));
                for (int r=0; r<4; ++r) {
                    ldr(Xbyak_aarch64::QReg(1+r), ptr(x0, ((i+r)*k+j)*4));
                    fmla(Xbyak_aarch64::VReg(16+r).s4, Xbyak_aarch64::VReg(1+r).s4, v0.s4);
                }
            }
            for (int r=0; r<4; ++r) {
                faddp(Xbyak_aarch64::VReg(16+r).s4, Xbyak_aarch64::VReg(16+r).s4, Xbyak_aarch64::VReg(16+r).s4);
                faddp(Xbyak_aarch64::SReg(16+r), Xbyak_aarch64::VReg(16+r).s2);
                str(Xbyak_aarch64::SReg(16+r), ptr(x2, (i+r)*4));
            }
        }
        ret();
        const auto emitted = Clock::now();
        readyRE();
        times.emit_us=us(start, emitted);
        times.publish_us=us(emitted, Clock::now());
    }
    Entry entry() const { return getCode<Entry>(); }
};
inline std::shared_ptr<Kernel> build(int m, int k) {
    const auto start=Clock::now();
    auto code=std::make_shared<Kernel>(m,k);
    code->times.allocate_us=us(start,Clock::now())-code->times.emit_us-code->times.publish_us;
    return code;
}
}
