// 来源: cppreference std::make_from_tuple Example
// 适配: std::tuple->mytup::Tuple, std::make_tuple/make_from_tuple->mytup::xxx
// 预期: 挂 —— mytup::make_tuple / mytup::make_from_tuple 未实现
#include "tuple.h"

#include <iostream>
#include <utility>

using namespace std;
using namespace mytup;

struct Foo
{
    Foo(int first, float second, int third)
    {
        std::cout << first << ", " << second << ", " << third << '\n';
    }
};

int main()
{
    auto tuple = mytup::make_tuple(42, 3.14f, 0);           // 未实现，挂
    mytup::make_from_tuple<Foo>(std::move(tuple));          // 未实现，挂
}
