// 来源: cppreference get(std::tuple) Example
// 适配: std::tuple->mytup::Tuple, std::get->mytup::get, std::tie->mytup::tie,
//       std::println->std::cout
// 预期: 挂 —— mytup::make_tuple / mytup::get<T>（按类型）/ mytup::tie 未实现;
//       按索引 get 与结构化绑定部分可过
#include "tuple.h"

#include <cassert>
#include <iostream>
#include <string>
#include <utility>

using namespace std;
using namespace mytup;

int main()
{
    using Tuple = mytup::Tuple<int, const char*, double>;
    auto x{mytup::make_tuple(42, "Foo", 3.14)};  // make_tuple 未实现，挂

    std::cout << "(" << mytup::get<0>(x) << ", "
              << mytup::get<1>(x) << ", "
              << mytup::get<2>(x) << ")\n";

    // 按类型取（C++14）—— 未实现，挂
    std::cout << "(" << mytup::get<int>(x) << ", "
              << mytup::get<const char*>(x) << ", "
              << mytup::get<double>(x) << ")\n";

    int a;
    const char* b;
    double c;
    mytup::tie(a, b, c) = x;  // tie 未实现，挂
    std::cout << "(" << a << ", " << b << ", " << c << ")\n";

    auto& [d, e, f] = x;
    std::cout << "(" << d << ", " << e << ", " << f << ")\n";
}
