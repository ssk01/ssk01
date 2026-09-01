// 来源: cppreference std::apply Example
// 适配: std::tuple->mytup::Tuple, std::apply->mytup::apply;
//       std::pair 参数换成 mytup::Tuple（我们的 apply 只吃 mytup::Tuple），
//       CTAD 换成显式类型（deduction guide 未实现）
// 预期: 过 —— apply + 折叠表达式打印全可用
#include "tuple.h"

#include <iostream>
#include <utility>

using namespace std;
using namespace mytup;

int add(int first, int second) { return first + second; }

template <typename T>
T add_generic(T first, T second) { return first + second; }

auto add_lambda = [](auto first, auto second) { return first + second; };

template <typename... Ts>
std::ostream& operator<<(std::ostream& os, mytup::Tuple<Ts...> const& theTuple)
{
    mytup::apply
    (
        [&os](Ts const&... tupleArgs)
        {
            os << '[';
            std::size_t n{0};
            ((os << tupleArgs << (++n != sizeof...(Ts) ? ", " : "")), ...);
            os << ']';
        }, theTuple
    );
    return os;
}

int main()
{
    std::cout << mytup::apply(add, mytup::Tuple<int, int>(1, 2)) << '\n';

    // add_generic 无法从函数模板推导 —— cppreference 原文注释掉
    // std::cout << std::apply(add_generic, std::make_pair(2.0f, 3.0f)) << '\n';

    std::cout << mytup::apply(add_lambda, mytup::Tuple<float, float>(2.0f, 3.0f)) << '\n';

    mytup::Tuple<int, const char*, float, char> myTuple{25, "Hello", 9.31f, 'c'};
    std::cout << myTuple << '\n';
}
