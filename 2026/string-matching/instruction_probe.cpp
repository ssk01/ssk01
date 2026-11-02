// Stable entry points for inspecting the operations discussed in the tutorial.
#include "match.h"
extern "C" __attribute__((noinline)) unsigned filter16(const char* text, char first) {
    return matching::first_mask(text,first);
}
extern "C" __attribute__((noinline)) bool compare_selected(const char* text,const char* pattern,const uint8_t* mask) {
    return matching::selected_equal(text,pattern,mask);
}
extern "C" __attribute__((noinline)) unsigned first_candidate(unsigned nonzero_mask) {
    return __builtin_ctz(nonzero_mask); // Precondition: nonzero, as in while(mask).
}
extern "C" __attribute__((noinline)) unsigned remove_candidate(unsigned mask) {
    return mask & (mask-1);
}
