// 来源: cppreference operator==,<,...(std::tuple) Example
// 适配: std::tuple->mytup::Tuple, std::get->mytup::get
// 预期: 挂 —— mytup::Tuple 未实现 operator<，std::sort 编译不过
//       （构造处改用显式 Tuple(...)，把失败点隔离到 operator<）
#include "tuple.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace mytup;

int main()
{
    std::vector<mytup::Tuple<int, std::string, float>> v;
    v.emplace_back(2, "baz", -0.1f);
    v.emplace_back(2, "bar", 3.14f);
    v.emplace_back(1, "foo", 10.1f);
    v.emplace_back(2, "baz", -1.1f);

    std::sort(v.begin(), v.end());  // 需要 operator< —— 未实现，挂

    for (const auto& p : v)
        std::cout << "{ " << mytup::get<0>(p)
                  << ", " << mytup::get<1>(p)
                  << ", " << mytup::get<2>(p)
                  << " }\n";
}
