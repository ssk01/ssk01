// 来源: cppreference std::tuple_cat Example
// 适配: std::tuple->mytup::Tuple, std::get->mytup::get,
//       std::tuple_cat/make_tuple/tie->mytup::xxx
// 预期: 挂 —— mytup::tuple_cat / mytup::make_tuple / mytup::tie 未实现
#include "tuple.h"

#include <iostream>
#include <string>
#include <utility>

using namespace std;
using namespace mytup;

template <class Tuple, std::size_t N>
struct TuplePrinter
{
    static void print(const Tuple& t)
    {
        TuplePrinter<Tuple, N - 1>::print(t);
        std::cout << ", " << mytup::get<N - 1>(t);
    }
};

template <class Tuple>
struct TuplePrinter<Tuple, 1>
{
    static void print(const Tuple& t)
    {
        std::cout << mytup::get<0>(t);
    }
};

template <typename... Args, std::enable_if_t<sizeof...(Args) == 0, int> = 0>
void print(const mytup::Tuple<Args...>&)
{
    std::cout << "()\n";
}

template <typename... Args, std::enable_if_t<sizeof...(Args) != 0, int> = 0>
void print(const mytup::Tuple<Args...>& t)
{
    std::cout << "(";
    TuplePrinter<decltype(t), sizeof...(Args)>::print(t);
    std::cout << ")\n";
}

int main()
{
    mytup::Tuple<int, std::string, float> t1(10, "Test", 3.14);
    int n = 7;
    auto t2 = mytup::tuple_cat(t1, mytup::make_tuple("Foo", "bar"), t1, mytup::tie(n));  // 均未实现，挂
    n = 42;
    print(t2);
}
