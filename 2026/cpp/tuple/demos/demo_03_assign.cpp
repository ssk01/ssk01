// 来源: cppreference tuple::operator= Example
// 适配: std::tuple->mytup::Tuple, std::apply->mytup::apply
// 预期: 挂 —— 同类型拷贝/移动赋值可过; 转换赋值 t1=t3 与 pair 赋值 t4=p1 未实现
//       （构造处把 {1,2,3} 改成 std::vector<int>{1,2,3}，隔离出真正的失败点）
#include "tuple.h"

#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace std;
using namespace mytup;

std::ostream& operator<<(std::ostream& os, std::vector<int> const& v)
{
    os << '{';
    for (std::size_t t = 0; t != v.size(); ++t)
        os << v[t] << (t + 1 < v.size() ? ", " : "");
    return os << '}';
}

template <class... Args>
void print_tuple(std::string_view name, const mytup::Tuple<Args...>& t)
{
    std::cout << name << " = {";
    mytup::apply([&](auto&& arg, auto&&... args)
    {
        std::cout << arg;
        ((std::cout << ", " << args), ...);
    }, t);
    std::cout << '}';
}

template <class Tuple1, class Tuple2>
void print_tuples(std::string_view name1, const Tuple1& t1,
                  std::string_view name2, const Tuple2& t2)
{
    print_tuple(name1, t1);
    std::cout << ", ";
    print_tuple(name2, mytup::Tuple(t2));  // 原版是 std::tuple(t2)：CTAD 从 pair 推导
    std::cout << "\n\n";
}

int main()
{
    mytup::Tuple<int, std::string, std::vector<int>>
        t1(1, "alpha", std::vector<int>{1, 2, 3}),
        t2(2, "beta", std::vector<int>{4, 5});
    print_tuples("1) t1", t1, "t2", t2);

    // 同类型拷贝赋值 —— 可过
    t1 = t2;
    print_tuples("2) t1 = t2;\n   t1", t1, "t2", t2);

    // 同类型移动赋值 —— 可过
    t1 = std::move(t2);
    print_tuples("3) t1 = std::move(t2);\n   t1", t1, "t2", t2);

    // 转换拷贝赋值（不同元素类型）—— 未实现，挂
    mytup::Tuple<short, const char*, std::vector<int>> t3(3, "gamma", std::vector<int>{6, 7, 8});
    t1 = t3;
    print_tuples("4) t1 = t3;\n   t1", t1, "t3", t3);

    // pair 赋值 —— 未实现，挂
    mytup::Tuple<std::string, std::vector<int>> t4("delta", std::vector<int>{10, 11, 12});
    std::pair<const char*, std::vector<int>> p1{"epsilon", {14, 15, 16}};
    print_tuples("6) t4", t4, "p1", p1);
    t4 = p1;
    print_tuples("7) t4 = p1;\n   t4", t4, "p1", p1);
}
