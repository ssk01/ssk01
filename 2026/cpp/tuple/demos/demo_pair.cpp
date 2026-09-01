// 手写 Pair：与 Tuple 的实现对比 demo
#include "pair.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace mytup;

struct Empty1 {};

int main() {
  cout << "1) 构造 + first/second + get + 结构化绑定\n";
  {
    Pair<int, string> p(42, "hello");
    cout << p.first << ' ' << p.second << '\n';
    cout << get<0>(p) << ' ' << get<1>(p) << '\n';
    cout << get<int>(p) << ' ' << get<string>(p) << '\n';  // 按类型取
    auto [a, b] = p;
    cout << a << ' ' << b << '\n';
    static_assert(tuple_size<decltype(p)>::value == 2, "pair size is 2");
    get<0>(p) = 100;  // 写穿
    cout << p.first << '\n';
  }

  cout << "\n2) 拷贝 / 移动 / 转换构造\n";
  {
    Pair<int, string> a(1, "x");
    Pair<int, string> b(a);             // 拷贝
    Pair<int, string> c(std::move(b));  // 移动
    Pair<double, string> d(mytup::make_pair(2, "y"));  // 转换构造（double from int, string from char*）
    cout << a.first << a.second << ' ' << c.first << c.second << ' '
         << d.first << d.second << '\n';
  }

  cout << "\n3) 比较运算符 + sort 字典序\n";
  {
    Pair<int, int> x(1, 2), y(1, 3), z(2, 0);
    cout << (x < y) << (y < z) << (x == Pair<int, int>(1, 2)) << '\n';

    vector<Pair<string, int>> v = {{"b", 2}, {"a", 3}, {"a", 1}};
    sort(v.begin(), v.end());
    for (auto& p : v) cout << '{' << p.first << ',' << p.second << "} ";
    cout << '\n';
  }

  cout << "\n4) swap + make_pair（unwrap std::ref）\n";
  {
    int n = 7;
    auto m = mytup::make_pair(1, std::ref(n));
    get<1>(m) = 99;  // 写穿 n（int& 元素）
    cout << "n = " << n << '\n';
    Pair<int, string> p(1, "a"), q(2, "b");
    swap(p, q);
    cout << p.first << p.second << ' ' << q.first << q.second << '\n';
  }

  cout << "\n5) 与 2 元 tuple 互转\n";
  {
    Pair<int, string> from_tup(Tuple<int, string>(1, "x"));  // Pair  ← Tuple
    Tuple<int, string> from_pair(std::make_pair(2, "y"));    // Tuple ← std::pair
    cout << from_tup.first << from_tup.second << ' '
         << get<0>(from_pair) << get<1>(from_pair) << '\n';
  }

  cout << "\n6) EBO 对比：Pair 和 Tuple 原理一致，压缩效果一样\n";
  {
    cout << "sizeof(Pair<int, string>)        = " << sizeof(Pair<int, string>) << '\n';
    cout << "sizeof(Tuple<int, string>)       = " << sizeof(Tuple<int, string>) << '\n';
    cout << "sizeof(Pair<Empty1, int>)        = " << sizeof(Pair<Empty1, int>) << '\n';
    cout << "sizeof(Tuple<Empty1, int>)       = " << sizeof(Tuple<Empty1, int>) << '\n';
  }
}
