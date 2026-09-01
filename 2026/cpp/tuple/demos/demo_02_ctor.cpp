// 来源: cppreference tuple::tuple（构造函数）Example
// 适配: std::tuple->mytup::Tuple, std::get->mytup::get
// 预期: 挂 —— 转换构造 t3{t2}（不同元素类型的 tuple）与 pair 构造 t4 未实现;
//       默认/值构造、打印 helper 部分可过
#include "tuple.h"

#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace std;
using namespace mytup;

template <class Os, class T>
Os& operator<<(Os& os, const std::vector<T>& v)
{
    os << '{';
    for (auto i{v.size()}; const T& e : v)
        os << e << (--i ? "," : "");
    return os << '}';
}

template <class T>
void print_single(const T& v)
{
    if constexpr (std::is_same_v<T, std::decay_t<std::string>>)
        std::cout << std::quoted(v);
    else if constexpr (std::is_same_v<std::decay_t<T>, char>)
        std::cout << "'" << v << "'";
    else
        std::cout << v;
}

template <class Tuple, std::size_t N>
struct TuplePrinter
{
    static void print(const Tuple& t)
    {
        TuplePrinter<Tuple, N - 1>::print(t);
        std::cout << ", ";
        print_single(mytup::get<N - 1>(t));
    }
};

template <class Tuple>
struct TuplePrinter<Tuple, 1>
{
    static void print(const Tuple& t)
    {
        print_single(mytup::get<0>(t));
    }
};

template <class... Args>
void print(std::string_view message, const mytup::Tuple<Args...>& t)
{
    std::cout << message << " (";
    TuplePrinter<decltype(t), sizeof...(Args)>::print(t);
    std::cout << ")\n";
}

int main()
{
    mytup::Tuple<int, std::string, double> t1;
    print("Value-initialized, t1:", t1);

    mytup::Tuple<int, std::string, double> t2{42, "Test", -3.14};
    print("Initialized with values, t2:", t2);

    // 转换构造（从不同元素类型的 tuple）—— 未实现，挂
    mytup::Tuple<char, std::string, int> t3{t2};
    print("Implicitly converted, t3:", t3);

    // pair 构造 —— 未实现，挂
    mytup::Tuple<int, double> t4{std::make_pair(42, 3.14)};
    print("Constructed from a pair, t4:", t4);
}
