// 来源: cppreference tuple::swap（成员函数）Example
// 适配: std::tuple->mytup::Tuple, std::get->mytup::get
// 预期: 挂 —— mytup::make_tuple 未实现; p1.swap(p2) 成员 swap 未实现
#include "tuple.h"

#include <iostream>
#include <string>
#include <utility>

using namespace std;
using namespace mytup;

int main()
{
    mytup::Tuple<int, std::string, float> p1{42, "ABCD", 2.71}, p2;
    p2 = mytup::make_tuple(10, "1234", 3.14);   // make_tuple 未实现，挂

    auto print_p1_p2 = [&](auto rem)
    {
        std::cout << rem
                  << "p1 = {" << mytup::get<0>(p1)
                  << ", "     << mytup::get<1>(p1)
                  << ", "     << mytup::get<2>(p1) << "}, "
                  << "p2 = {" << mytup::get<0>(p2)
                  << ", "     << mytup::get<1>(p2)
                  << ", "     << mytup::get<2>(p2) << "}\n";
    };

    print_p1_p2("Before p1.swap(p2): ");
    p1.swap(p2);   // 成员 swap 未实现，挂
    print_p1_p2("After  p1.swap(p2): ");
    swap(p1, p2);
    print_p1_p2("After swap(p1, p2): ");
}
