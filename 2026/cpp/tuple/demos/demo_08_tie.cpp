// 来源: cppreference std::tie Example
// 适配: std::tuple->mytup::Tuple, std::tie->mytup::tie, std::ignore->mytup::ignore
// 预期: 挂 —— mytup::tie / mytup::ignore / operator<（tie 比较）/ CTAD 均未实现
#include "tuple.h"

#include <cassert>
#include <iostream>
#include <set>
#include <string>
#include <utility>

using namespace std;
using namespace mytup;

struct S
{
    int n;
    std::string s;
    float d;

    friend bool operator<(const S& lhs, const S& rhs) noexcept
    {
        return mytup::tie(lhs.n, lhs.s, lhs.d) < mytup::tie(rhs.n, rhs.s, rhs.d);  // tie+operator< 未实现，挂
    }
};

int main()
{
    std::set<S> set_of_s;

    S value{42, "Test", 3.14};
    std::set<S>::iterator iter;
    bool is_inserted;

    std::tie(iter, is_inserted) = set_of_s.insert(value);  // std::pair 的 tie 是 std 的，此处 OK
    assert(is_inserted);

    // CTAD 未实现，挂
    auto position = [](int w) { return mytup::Tuple(1 * w, 2 * w); };
    auto [x, y] = position(1);
    assert(x == 1 && y == 2);

    mytup::tie(x, y) = position(2);  // tie 未实现，挂
    assert(x == 2 && y == 4);

    mytup::Tuple<char, short> coordinates(6, 9);
    mytup::tie(x, y) = coordinates;  // 挂

    std::string z;
    mytup::tie(x, mytup::ignore, z) = mytup::Tuple(1, 2.0, "Test");  // ignore+CTAD 未实现，挂
    assert(x == 1 && z == "Test");
}
