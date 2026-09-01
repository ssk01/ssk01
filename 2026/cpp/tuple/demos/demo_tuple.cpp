#include "tuple.h"

#include <iostream>
#include <memory>
#include <string>
#include <utility>

using namespace std;
using namespace mytup;

struct Empty1 {};
struct Empty2 {};

// 朴素实现：普通成员，无 EBO，用来对比 sizeof
template <typename...>
struct NaiveTuple;

template <>
struct NaiveTuple<> {};

template <typename T, typename... Rest>
struct NaiveTuple<T, Rest...> {
  T head_;
  NaiveTuple<Rest...> tail_;
};

// 打印 helper：折叠表达式展开 get<0..n>
template <typename TupleLike, size_t... Is>
void print_tuple_impl(const TupleLike& t, index_sequence<Is...>) {
  cout << "{";
  ((cout << (Is ? ", " : "") << get<Is>(t)), ...);
  cout << "}";
}

template <typename... Types>
void print_tuple(const Tuple<Types...>& t) {
  print_tuple_impl(t, index_sequence_for<Types...>{});
}

int main() {
  cout << "1) 构造 + get + 类型工具\n";
  {
    Tuple<int, double, string> t(42, 3.14, "hello");
    static_assert(std::tuple_size<decltype(t)>::value == 3, "size must be 3");
    static_assert(std::is_same_v<std::tuple_element<1, decltype(t)>::type, double>,
                  "element 1 must be double");
    cout << get<0>(t) << ", " << get<1>(t) << ", " << get<2>(t) << '\n';
    get<0>(t) = 100;  // get 返回可写引用
    cout << "after write: " << get<0>(t) << '\n';
  }

  cout << "\n2) 拷贝 / 移动\n";
  {
    Tuple<int, string> a(1, "x");
    Tuple<int, string> b(a);            // 拷贝构造
    Tuple<int, string> c(std::move(b)); // 移动构造
    print_tuple(a);
    cout << '\n';
  }

  cout << "\n3) 结构化绑定（tuple_size + get 协议）\n";
  {
    Tuple<int, string, double> t(7, "seven", 7.5);
    auto [i, s, d] = t;       // 拷贝
    auto& [ri, rs, rd] = t;   // 引用
    rs = "SEVEN";             // 写穿到 t
    cout << i << ' ' << s << ' ' << d << '\n';
    print_tuple(t);
    cout << '\n';
  }

  cout << "\n4) apply\n";
  {
    Tuple<int, int, int> t(1, 2, 3);
    // 用 mytup::apply：非限定 apply 会和 std::apply（<memory> 传递引入）歧义
    cout << "sum = "
         << mytup::apply([](int x, int y, int z) { return x + y + z; }, t)
         << '\n';
    cout << "concat = "
         << mytup::apply([](int x, int y, int z) {
              return to_string(x) + "-" + to_string(y) + "-" + to_string(z);
            }, t)
         << '\n';
  }

  cout << "\n5) move-only 元素（unique_ptr）\n";
  {
    Tuple<unique_ptr<int>, string> t(make_unique<int>(5), "owned");
    cout << "*get<0>(t) = " << *get<0>(t) << '\n';
    Tuple<unique_ptr<int>, string> u(std::move(t));  // 移动构造，编译过 = 支持 move-only
    cout << "*get<0>(u) = " << *get<0>(u) << '\n';
  }

  cout << "\n6) 引用元素 / 函数指针元素（继承式 EBO 做不到，成员式可以）\n";
  {
    int x = 7;
    Tuple<int&> r(x);     // 引用元素：绑定外部变量
    get<0>(r) = 99;       // 写穿
    cout << "x = " << x << '\n';

    void (*fp)(int) = [](int v) { cout << "fp(" << v << ")\n"; };
    Tuple<void (*)(int)> ft(fp);
    get<0>(ft)(42);
  }

  cout << "\n7) EBO 压缩空元素\n";
  {
    cout << "sizeof(Empty1)                     = " << sizeof(Empty1) << '\n';
    cout << "sizeof(NaiveTuple<Empty1, Empty2, int>) = "
         << sizeof(NaiveTuple<Empty1, Empty2, int>) << '\n';
    cout << "sizeof(Tuple<Empty1, Empty2, int>)     = "
         << sizeof(Tuple<Empty1, Empty2, int>) << '\n';
    cout << "sizeof(Tuple<int, int, int>)       = "
         << sizeof(Tuple<int, int, int>) << '\n';
  }

  cout << "\n8) 空 tuple\n";
  {
    Tuple<> e;
    static_assert(tuple_size<decltype(e)>::value == 0, "empty");
    cout << "Tuple<> size = " << sizeof(e) << '\n';
  }
}
