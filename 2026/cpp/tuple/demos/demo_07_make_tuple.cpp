// 来源: cppreference std::make_tuple Example
// 适配: std::tuple->mytup::Tuple, std::get->mytup::get, std::tie->mytup::tie
// 预期: 挂 —— mytup::make_tuple / mytup::tie 未实现
//       参考元素 std::ref(n) 我们其实支持（Tuple<int&>），卡在 make_tuple 本身
#include "tuple.h"

#include <functional>
#include <iostream>
#include <utility>

using namespace std;
using namespace mytup;

mytup::Tuple<int, int> f()
{
    int x = 5;
    return mytup::make_tuple(x, 7);  // make_tuple 未实现，挂
}

int main()
{
    int n = 1;
    auto t = mytup::make_tuple(10, "Test", 3.14, std::ref(n), n);  // 挂
    n = 7;
    std::cout << "The value of t is ("
              << mytup::get<0>(t) << ", "
              << mytup::get<1>(t) << ", "
              << mytup::get<2>(t) << ", "
              << mytup::get<3>(t) << ", "
              << mytup::get<4>(t) << ")\n";

    int a, b;
    mytup::tie(a, b) = f();  // tie 未实现，挂
    std::cout << a << ' ' << b << '\n';
}
