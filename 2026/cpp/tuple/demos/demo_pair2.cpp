// 继承式 EBO 版 Pair2：与 [[no_unique_address]] 成员式 Pair 对比
#include "pair.h"
#include "pair2.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace mytup;

struct Empty1 {};
struct Empty2 {};
struct EmptyFinal final {};

int main() {
  cout << "1) 功能：Pair2 与 Pair 等价（访问器是 first()/second() 而非成员）\n";
  {
    Pair2<int, string> p(42, "hello");
    cout << p.first() << ' ' << p.second() << '\n';
    cout << get<0>(p) << ' ' << get<1>(p) << '\n';
    auto [a, b] = p;  // 结构化绑定照常工作
    cout << a << ' ' << b << '\n';
    get<0>(p) = 100;
    cout << p.first() << '\n';

    vector<Pair2<string, int>> v = {{"b", 2}, {"a", 3}, {"a", 1}};
    sort(v.begin(), v.end());  // operator< 字典序
    for (auto& x : v) cout << '{' << x.first() << ',' << x.second() << "} ";
    cout << '\n';

    int n = 7;
    Pair2<int&, int> r(n, 3);  // 引用元素：继承式分派自动走"存成员"分支
    get<0>(r) = 99;
    cout << "n = " << n << '\n';
  }

  cout << "\n2) sizeof 对比：Pair（no_unique_address 成员式）vs Pair2（继承式 EBO）\n";
  {
    cout << "Empty1 + int      : Pair = " << sizeof(Pair<Empty1, int>)
         << "   Pair2 = " << sizeof(Pair2<Empty1, int>) << '\n';
    cout << "Empty1 + Empty2   : Pair = " << sizeof(Pair<Empty1, Empty2>)
         << "   Pair2 = " << sizeof(Pair2<Empty1, Empty2>) << '\n';
    cout << "Empty1 + Empty1   : Pair = " << sizeof(Pair<Empty1, Empty1>)
         << "   Pair2 = " << sizeof(Pair2<Empty1, Empty1>) << "   <- 同类型空类，两种都压不到 1\n";
    cout << "int + string      : Pair = " << sizeof(Pair<int, string>)
         << "   Pair2 = " << sizeof(Pair2<int, string>) << '\n';
    cout << "int& + string     : Pair = " << sizeof(Pair<int&, string>)
         << "   Pair2 = " << sizeof(Pair2<int&, string>) << '\n';
    cout << "EmptyFinal + int  : Pair = " << sizeof(Pair<EmptyFinal, int>)
         << "   Pair2 = " << sizeof(Pair2<EmptyFinal, int>) << "   <- final 空类，继承式输\n";
  }

  cout << "\n3) 函数指针元素：继承式分派也不崩（对比 smartPoint Uniqlo 的坑）\n";
  {
    void (*fp)(int) = [](int v) { cout << "fp(" << v << ")\n"; };
    Pair2<void (*)(int), int> f(fp, 42);
    get<0>(f)(get<1>(f));
  }
}
