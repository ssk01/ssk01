// 来源: cppreference tuple_size<std::tuple> Example
// 适配: std::tuple->mytup::Tuple, std::make_tuple->直接构造（我们的 make_tuple 未实现，非本页被测点）
// 预期: 过 —— std::tuple_size（我们对 mytup::Tuple 的特化）编译期/运行期都可用
#include "tuple.h"

#include <iostream>
#include <utility>

using namespace std;
using namespace mytup;

template <class T>
void test(T value)
{
    int a[std::tuple_size_v<T>];  // 编译期可用

    std::cout << std::tuple_size<T>{} << ' '  // 运行期可用
              << sizeof a << ' '
              << sizeof value << '\n';
}

int main()
{
    test(mytup::Tuple<int, int, double>(1, 2, 3.14));
}
