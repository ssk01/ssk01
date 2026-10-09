#include <arm_neon.h>
#include <Accelerate/Accelerate.h>
#include "../../v9-op-registry/kernels/sme2_gemm.h"

// Separate translation unit; no LTO: runtime shape cannot be constant propagated.
extern "C" void neon_generic(const float* a, const float* b, float* c, int m, int k) {
    for (int i = 0; i < m; i += 4) {
        float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
        for (int j = 0; j < k; j += 4) {
            const auto w = vld1q_f32(b+j);
            s0 = vfmaq_f32(s0, vld1q_f32(a+(i+0)*k+j), w);
            s1 = vfmaq_f32(s1, vld1q_f32(a+(i+1)*k+j), w);
            s2 = vfmaq_f32(s2, vld1q_f32(a+(i+2)*k+j), w);
            s3 = vfmaq_f32(s3, vld1q_f32(a+(i+3)*k+j), w);
        }
        c[i] = vaddvq_f32(s0); c[i+1] = vaddvq_f32(s1);
        c[i+2] = vaddvq_f32(s2); c[i+3] = vaddvq_f32(s3);
    }
}
extern "C" void accelerate(const float* a, const float* b, float* c, int m, int k) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, k, 1, a, k, b, 1, 0, c, 1);
}
extern "C" void tf_existing(const float* a, const float* b, float* c, int m, int k) {
    lf::sme2_gemm(m, 1, k, a, b, c);
}
extern "C" int tf_has_sme2() { return LF_SME2; }

// AOT control: the same four-row arithmetic with compile-time M and K.
// This demonstrates that specialization can also be done before runtime.
template<int M,int K>
void aot(const float* a,const float* b,float* c,int,int) {
    for(int i=0;i<M;i+=4) {
        float32x4_t s0=vdupq_n_f32(0),s1=s0,s2=s0,s3=s0;
        for(int j=0;j<K;j+=4) {
            auto w=vld1q_f32(b+j);
            s0=vfmaq_f32(s0,vld1q_f32(a+(i+0)*K+j),w);
            s1=vfmaq_f32(s1,vld1q_f32(a+(i+1)*K+j),w);
            s2=vfmaq_f32(s2,vld1q_f32(a+(i+2)*K+j),w);
            s3=vfmaq_f32(s3,vld1q_f32(a+(i+3)*K+j),w);
        }
        c[i]=vaddvq_f32(s0);c[i+1]=vaddvq_f32(s1);
        c[i+2]=vaddvq_f32(s2);c[i+3]=vaddvq_f32(s3);
    }
}
using Entry=void(*)(const float*,const float*,float*,int,int);
extern "C" Entry aot_entry(int m,int k) {
    if(k!=16)return nullptr;
    switch(m){case 4:return aot<4,16>;case 8:return aot<8,16>;
        case 16:return aot<16,16>;case 32:return aot<32,16>;default:return nullptr;}
}
