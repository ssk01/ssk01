#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

#include "tuple.h"  // 复用 mytup::get / unwrap_ref_decay_t / Tuple（做 2 元互转）

namespace mytup {

// ===== Pair：2 元素异构容器 =====
// 概念上和 Tuple<T1,T2> 一样，但实现完全不同：
// 固定两个成员，不需要变参模板 / 递归继承 / 类型工具 / 条件 explicit。
// 值构造固定 2 参数，永远不会跟 1 参数的拷贝/移动构造抢（tuple 的转发偷拷贝坑在这里不存在）。

template <std::size_t I, typename T1, typename T2>
struct PairElement;

template <typename T1, typename T2>
struct PairElement<0, T1, T2> {
  using type = T1;
};

template <typename T1, typename T2>
struct PairElement<1, T1, T2> {
  using type = T2;
};

template <std::size_t I, typename T1, typename T2>
using pair_element_t = typename PairElement<I, T1, T2>::type;

template <typename T1, typename T2>
class Pair {
public:
  [[no_unique_address]] T1 first;
  [[no_unique_address]] T2 second;

  // 默认构造：值初始化所有元素（std::pair 语义）
  Pair() : first(), second() {}

  // 值构造：完美转发
  template <typename U1, typename U2>
  Pair(U1&& u1, U2&& u2)
      : first(std::forward<U1>(u1)), second(std::forward<U2>(u2)) {}

  // 拷贝/移动构造：隐式生成（值构造要 2 参数，不会来抢）

  // 转换构造：从 Pair<U1, U2>
  template <typename U1, typename U2,
            std::enable_if_t<std::is_constructible_v<T1, const U1&> &&
                                 std::is_constructible_v<T2, const U2&>,
                             int> = 0>
  Pair(const Pair<U1, U2>& other)
      : first(other.first), second(other.second) {}

  template <typename U1, typename U2,
            std::enable_if_t<std::is_constructible_v<T1, U1&&> &&
                                 std::is_constructible_v<T2, U2&&>,
                             int> = 0>
  Pair(Pair<U1, U2>&& other)
      : first(std::forward<U1>(other.first)),
        second(std::forward<U2>(other.second)) {}

  // 与 2 元 tuple 互转（std 里 pair 和 tuple<T1,T2> 可互相转换）
  template <typename U1, typename U2,
            std::enable_if_t<std::is_constructible_v<T1, const U1&> &&
                                 std::is_constructible_v<T2, const U2&>,
                             int> = 0>
  Pair(const Tuple<U1, U2>& t)
      : first(get<0>(t)), second(get<1>(t)) {}

  template <typename U1, typename U2,
            std::enable_if_t<std::is_constructible_v<T1, U1&&> &&
                                 std::is_constructible_v<T2, U2&&>,
                             int> = 0>
  Pair(Tuple<U1, U2>&& t)
      : first(get<0>(std::move(t))), second(get<1>(std::move(t))) {}

  void swap(Pair& other) noexcept(std::is_nothrow_swappable_v<T1> &&
                                  std::is_nothrow_swappable_v<T2>) {
    using std::swap;
    swap(first, other.first);
    swap(second, other.second);
  }
};

// ===== get<I>：直接按 0/1 硬编码，不用递归 =====
template <std::size_t I, typename T1, typename T2>
constexpr pair_element_t<I, T1, T2>& get(Pair<T1, T2>& p) noexcept {
  static_assert(I <= 1, "pair index out of range");
  if constexpr (I == 0)
    return p.first;
  else
    return p.second;
}

template <std::size_t I, typename T1, typename T2>
constexpr const pair_element_t<I, T1, T2>& get(
    const Pair<T1, T2>& p) noexcept {
  static_assert(I <= 1, "pair index out of range");
  if constexpr (I == 0)
    return p.first;
  else
    return p.second;
}

template <std::size_t I, typename T1, typename T2>
constexpr pair_element_t<I, T1, T2>&& get(Pair<T1, T2>&& p) noexcept {
  static_assert(I <= 1, "pair index out of range");
  if constexpr (I == 0)
    return std::forward<T1>(p.first);
  else
    return std::forward<T2>(p.second);
}

// ===== get<T>：按类型取（要求唯一）=====
template <typename T, typename T1, typename T2>
constexpr T& get(Pair<T1, T2>& p) noexcept {
  static_assert((std::is_same_v<T, T1> + std::is_same_v<T, T2>) == 1,
                "type T must occur exactly once in the pair");
  if constexpr (std::is_same_v<T, T1>)
    return p.first;
  else
    return p.second;
}

template <typename T, typename T1, typename T2>
constexpr const T& get(const Pair<T1, T2>& p) noexcept {
  static_assert((std::is_same_v<T, T1> + std::is_same_v<T, T2>) == 1,
                "type T must occur exactly once in the pair");
  if constexpr (std::is_same_v<T, T1>)
    return p.first;
  else
    return p.second;
}

template <typename T, typename T1, typename T2>
constexpr T&& get(Pair<T1, T2>&& p) noexcept {
  static_assert((std::is_same_v<T, T1> + std::is_same_v<T, T2>) == 1,
                "type T must occur exactly once in the pair");
  if constexpr (std::is_same_v<T, T1>)
    return std::forward<T1>(p.first);
  else
    return std::forward<T2>(p.second);
}

// ===== 比较运算符：先 first 后 second 字典序 =====
template <typename T1, typename T2>
constexpr bool operator==(const Pair<T1, T2>& a, const Pair<T1, T2>& b) {
  return a.first == b.first && a.second == b.second;
}

template <typename T1, typename T2>
constexpr bool operator!=(const Pair<T1, T2>& a, const Pair<T1, T2>& b) {
  return !(a == b);
}

template <typename T1, typename T2>
constexpr bool operator<(const Pair<T1, T2>& a, const Pair<T1, T2>& b) {
  if (a.first < b.first) return true;
  if (b.first < a.first) return false;
  return a.second < b.second;
}

template <typename T1, typename T2>
constexpr bool operator>(const Pair<T1, T2>& a, const Pair<T1, T2>& b) {
  return b < a;
}

template <typename T1, typename T2>
constexpr bool operator<=(const Pair<T1, T2>& a, const Pair<T1, T2>& b) {
  return !(b < a);
}

template <typename T1, typename T2>
constexpr bool operator>=(const Pair<T1, T2>& a, const Pair<T1, T2>& b) {
  return !(a < b);
}

// ===== swap / make_pair =====
template <typename T1, typename T2>
void swap(Pair<T1, T2>& a, Pair<T1, T2>& b) noexcept(noexcept(a.swap(b))) {
  a.swap(b);
}

template <typename T1, typename T2>
constexpr Pair<unwrap_ref_decay_t<T1>, unwrap_ref_decay_t<T2>> make_pair(
    T1&& a, T2&& b) {
  return Pair<unwrap_ref_decay_t<T1>, unwrap_ref_decay_t<T2>>(
      std::forward<T1>(a), std::forward<T2>(b));
}

}  // namespace mytup

// ===== std::tuple_size / tuple_element 特化（结构化绑定协议）=====
namespace std {

template <typename T1, typename T2>
struct tuple_size<::mytup::Pair<T1, T2>> : integral_constant<size_t, 2> {};

template <size_t I, typename T1, typename T2>
struct tuple_element<I, ::mytup::Pair<T1, T2>> {
  using type = ::mytup::pair_element_t<I, T1, T2>;
};

}  // namespace std
