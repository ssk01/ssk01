#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#else
#error "This demo requires ARM NEON or x86 SSE2"
#endif

namespace matching {
using View = std::string_view;
constexpr size_t width = 16;
inline const char* backend() {
#if defined(__ARM_NEON)
    return "NEON";
#else
    return "SSE2";
#endif
}
// Each output bit describes ONE byte position, not one whole substring.
inline uint16_t equal_mask(const char* a, const char* b) {
#if defined(__ARM_NEON)
    const auto eq = vceqq_u8(vld1q_u8(reinterpret_cast<const uint8_t*>(a)),
                              vld1q_u8(reinterpret_cast<const uint8_t*>(b)));
    alignas(16) uint8_t lanes[16];
    vst1q_u8(lanes, eq);
    uint16_t mask = 0;
    for (size_t i = 0; i < width; ++i) mask |= uint16_t(lanes[i] != 0) << i;
    return mask;
#else
    return static_cast<uint16_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(a)),
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(b)))));
#endif
}
inline uint16_t first_mask(const char* s, char first) {
#if defined(__ARM_NEON)
    const auto eq = vceqq_u8(vld1q_u8(reinterpret_cast<const uint8_t*>(s)),
                              vdupq_n_u8(static_cast<uint8_t>(first)));
    if (vmaxvq_u8(eq) == 0) return 0; // No candidates: skip all 16 starts.
    alignas(16) uint8_t lanes[16];
    vst1q_u8(lanes, eq);
    uint16_t mask = 0;
    for (size_t i = 0; i < width; ++i) mask |= uint16_t(lanes[i] != 0) << i;
    return mask;
#else
    return static_cast<uint16_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(s)), _mm_set1_epi8(first))));
#endif
}
inline bool scalar_equal(View a, View b) {
    if (a.size() != b.size()) return false;
    for (size_t k = 0; k < b.size(); ++k) if (a[k] != b[k]) return false;
    return true;
}
// Only complete, in-bounds vector loads; a short tail uses padded local copies.
inline bool vector_equal(View a, View b) {
    if (a.size() != b.size()) return false;
    size_t k = 0;
    for (; b.size() - k >= width; k += width)
        if (equal_mask(a.data() + k, b.data() + k) != 0xffff) return false;
    if (k == b.size()) return true;
    // A padded local copy keeps short-needle SIMD comparisons safe even at a page edge.
    alignas(16) char ap[16] = {}, bp[16] = {};
    std::memcpy(ap, a.data() + k, b.size() - k);
    std::memcpy(bp, b.data() + k, b.size() - k);
    return equal_mask(ap, bp) == 0xffff;
}
inline bool scalar_prefix(View s, View n) {
    return s.size() >= n.size() && scalar_equal(s.substr(0, n.size()), n);
}
inline bool scalar_suffix(View s, View n) {
    return s.size() >= n.size() && scalar_equal(s.substr(s.size() - n.size()), n);
}
// Prepare a constant query once, outside the row loop and timed region.
struct Pattern {
    View bytes;
    alignas(16) char left[16] = {}, right[16] = {};
    alignas(16) uint8_t left_mask[16] = {}, right_mask[16] = {};
    explicit Pattern(View n): bytes(n) {
        if (!n.empty() && n.size() <= width) {
            std::memcpy(left, n.data(), n.size());
            std::memcpy(right + width - n.size(), n.data(), n.size());
            for (size_t k=0;k<n.size();++k) { left_mask[k]=255; right_mask[width-n.size()+k]=255; }
        }
    }
};
inline bool selected_equal(const char* s, const char* n, const uint8_t* mask) {
#if defined(__ARM_NEON)
    auto eq=vceqq_u8(vld1q_u8(reinterpret_cast<const uint8_t*>(s)),vld1q_u8(reinterpret_cast<const uint8_t*>(n)));
    auto m=vld1q_u8(mask);
    return vmaxvq_u8(veorq_u8(vandq_u8(eq,m),m))==0;
#else
    uint16_t wanted=static_cast<uint16_t>(_mm_movemask_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(mask))));
    return (equal_mask(s,n)&wanted)==wanted;
#endif
}
inline bool simd_prefix(View s, const Pattern& p) {
    View n=p.bytes;
    if (n.empty()) return true;
    if (s.size() < n.size()) return false;
    if (n.size() <= width && s.size() >= width) return selected_equal(s.data(),p.left,p.left_mask);
    return vector_equal(s.substr(0, n.size()), n);
}
inline bool simd_suffix(View s, const Pattern& p) {
    View n=p.bytes;
    if (n.empty()) return true;
    if (s.size() < n.size()) return false;
    if (n.size() <= width && s.size() >= width) return selected_equal(s.data()+s.size()-width,p.right,p.right_mask);
    return vector_equal(s.substr(s.size()-n.size()), n);
}
inline bool simd_prefix(View s, View n) { return simd_prefix(s,Pattern(n)); }
inline bool simd_suffix(View s, View n) { return simd_suffix(s,Pattern(n)); }
inline bool scalar_contains(View s, View n) {
    if (n.empty()) return true;
    if (s.size() < n.size()) return false;
    for (size_t j = 0; j <= s.size() - n.size(); ++j)
        if (scalar_equal(s.substr(j, n.size()), n)) return true;
    return false;
}
struct Stats { uint64_t blocks=0, active_blocks=0, candidates=0, tail_starts=0; };
// Stats are compiled out in benchmark calls, collected in a separate pass.
template<bool Count = false>
inline bool simd_contains(View s, View n, Stats* stats = nullptr) {
    if (n.empty()) return true;
    if (s.size() < n.size()) return false;
    const size_t starts = s.size() - n.size() + 1;
    size_t j = 0;
    for (; starts - j >= width; j += width) {
        uint16_t mask = first_mask(s.data() + j, n[0]);
        if constexpr (Count) { ++stats->blocks; stats->active_blocks += mask != 0; }
        while (mask) {
            const unsigned lane = static_cast<unsigned>(__builtin_ctz(static_cast<unsigned>(mask)));
            const size_t pos = j + lane;
            if constexpr (Count) ++stats->candidates;
            if (scalar_equal(s.substr(pos + 1, n.size() - 1), n.substr(1))) return true;
            mask &= static_cast<uint16_t>(mask - 1); // Remove only the lowest candidate bit.
        }
    }
    for (; j < starts; ++j) {
        if constexpr (Count) ++stats->tail_starts;
        if (scalar_equal(s.substr(j, n.size()), n)) return true;
    }
    return false;
}
inline bool simd_contains_plain(View s, View n) { return simd_contains(s,n); }
inline bool libc_contains(View s, View n) {
    if (n.empty()) return true;
    if (s.size() < n.size()) return false;
    size_t j=0, starts=s.size()-n.size()+1;
    while (j<starts) {
        const auto* p=static_cast<const char*>(std::memchr(s.data()+j, static_cast<unsigned char>(n[0]), starts-j));
        if (!p) return false;
        j=static_cast<size_t>(p-s.data());
        if (scalar_equal(s.substr(j+1,n.size()-1),n.substr(1))) return true;
        ++j;
    }
    return false;
}
}
