// 来源: cppreference std::forward_as_tuple Example
// 适配: std::tuple->mytup::Tuple, std::forward_as_tuple->mytup::forward_as_tuple
// 预期: 挂 —— mytup::forward_as_tuple 未实现（依赖 Tuple<Types&&...> 引用元素，本体可支持）
#include "tuple.h"

#include <iostream>
#include <map>
#include <string>
#include <utility>

using namespace std;
using namespace mytup;

int main()
{
    std::map<int, std::string> m;

    m.emplace(std::piecewise_construct,
              mytup::forward_as_tuple(6),              // 未实现，挂
              mytup::forward_as_tuple(9, 'g'));
    std::cout << "m[6] = " << m[6] << '\n';
}
