// 来源: cppreference tuple 推导指引 Example
// 适配: std::tuple->mytup::Tuple
// 预期: 挂 —— 未实现 CTAD 推导指引（tuple<Ts...> 从实参推断类型）
#include "tuple.h"

#include <utility>

int main()
{
    int a[2], b[3], c[4];
    mytup::Tuple t1{a, b, c};  // 需要推导指引 —— 未实现，挂
}
