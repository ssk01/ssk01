#include <cstddef>
#include <type_traits>
#include <utility>

// 继承式 EBO 版 pair（对比 pair.h 的 [[no_unique_address]] 成员式）
// 思路（老版 libc++ __compressed_pair_elem）：
//   空类元素（is_empty && !is_final）→ 继承存，空基类子对象压成 0 字节
//   其余（非空 / 引用 / 函数指针 / final 类）→ 存成员
// 实测（Clang）：除"final 空类"外，两种方案 sizeof 完全一样——
//   同类型空类 Empty1+Empty1 都是 2（继承式也无法压到 1：两个包装基类各自
//   内嵌同类型 Empty1 子对象，不能同址）；final 空类时继承式输（8 vs 4，
//   因为 final 不能继承只能存成员、又不能用 no_unique_address）。
// 真正的代价在 API：元素是基类子对象，没有 public 成员，只能暴露
//   first()/second() 访问器——做不到 std::pair 的 .first/.second 成员。

namespace mytup {

template <std::size_t I, typename T1, typename T2>
struct Pair2ElementType;

template <typename T1, typename T2>
struct Pair2ElementType<0, T1, T2> {
  using type = T1;
};

template <typename T1, typename T2>
struct Pair2ElementType<1, T1, T2> {
  using type = T2;
};

template <std::size_t I, typename T1, typename T2>
using pair2_element_t = typename Pair2ElementType<I, T1, T2>::type;

// 单个元素：空类走继承，否则存成员
template <std::size_t I, typename T,
          bool = std::is_empty_v<T> && !std::is_final_v<T>>
struct Pair2Elem;

// 空类 → 继承（EBO），value() 上转型取回
template <std::size_t I, typename T>
struct Pair2Elem<I, T, true> : private T {
  Pair2Elem() = default;
  template <typename U>
  Pair2Elem(U&& u) : T(std::forward<U>(u)) {}
  T& value() { return *this; }
  const T& value() const { return *this; }
};

// 非空 / 引用 / 函数指针 / final → 存成员
template <std::size_t I, typename T>
struct Pair2Elem<I, T, false> {
  Pair2Elem() : value_() {}
  template <typename U>
  Pair2Elem(U&& u) : value_(std::forward<U>(u)) {}
  T& value() { return value_; }
  const T& value() const { return value_; }
  T value_;
};

template <typename T1, typename T2>
class Pair2 : private Pair2Elem<0, T1>, private Pair2Elem<1, T2> {
  using E0 = Pair2Elem<0, T1>;
  using E1 = Pair2Elem<1, T2>;

public:
  Pair2() = default;

  // 值构造：完美转发到两个基类元素
  template <typename U1, typename U2>
  Pair2(U1&& a, U2&& b)
      : E0(std::forward<U1>(a)), E1(std::forward<U2>(b)) {}

  // 转换构造：从 Pair2<U1, U2>
  template <typename U1, typename U2,
            std::enable_if_t<std::is_constructible_v<T1, const U1&> &&
                                 std::is_constructible_v<T2, const U2&>,
                             int> = 0>
  Pair2(const Pair2<U1, U2>& other)
      : E0(get<0>(other)), E1(get<1>(other)) {}

  template <typename U1, typename U2,
            std::enable_if_t<std::is_constructible_v<T1, U1&&> &&
                                 std::is_constructible_v<T2, U2&&>,
                             int> = 0>
  Pair2(Pair2<U1, U2>&& other)
      : E0(get<0>(std::move(other))), E1(get<1>(std::move(other))) {}

  // 元素在基类子对象里，用访问器（没有 public 成员）
  T1& first() { return static_cast<E0&>(*this).value(); }
  const T1& first() const { return static_cast<const E0&>(*this).value(); }
  T2& second() { return static_cast<E1&>(*this).value(); }
  const T2& second() const { return static_cast<const E1&>(*this).value(); }

  void swap(Pair2& other) noexcept(std::is_nothrow_swappable_v<T1> &&
                                   std::is_nothrow_swappable_v<T2>) {
    using std::swap;
    swap(first(), other.first());
    swap(second(), other.second());
  }
};

// ===== get<I>：基类是私有的，走公开访问器 first()/second() =====
template <std::size_t I, typename T1, typename T2>
constexpr pair2_element_t<I, T1, T2>& get(Pair2<T1, T2>& p) noexcept {
  static_assert(I <= 1, "pair index out of range");
  if constexpr (I == 0)
    return p.first();
  else
    return p.second();
}

template <std::size_t I, typename T1, typename T2>
constexpr const pair2_element_t<I, T1, T2>& get(
    const Pair2<T1, T2>& p) noexcept {
  static_assert(I <= 1, "pair index out of range");
  if constexpr (I == 0)
    return p.first();
  else
    return p.second();
}

template <std::size_t I, typename T1, typename T2>
constexpr pair2_element_t<I, T1, T2>&& get(Pair2<T1, T2>&& p) noexcept {
  static_assert(I <= 1, "pair index out of range");
  // forward 而非 move：引用元素（int&）要保左值
  if constexpr (I == 0)
    return std::forward<T1>(p.first());
  else
    return std::forward<T2>(p.second());
}

// ===== 比较运算符 =====
template <typename T1, typename T2>
constexpr bool operator==(const Pair2<T1, T2>& a, const Pair2<T1, T2>& b) {
  return a.first() == b.first() && a.second() == b.second();
}

template <typename T1, typename T2>
constexpr bool operator!=(const Pair2<T1, T2>& a, const Pair2<T1, T2>& b) {
  return !(a == b);
}

template <typename T1, typename T2>
constexpr bool operator<(const Pair2<T1, T2>& a, const Pair2<T1, T2>& b) {
  if (a.first() < b.first()) return true;
  if (b.first() < a.first()) return false;
  return a.second() < b.second();
}

template <typename T1, typename T2>
constexpr bool operator>(const Pair2<T1, T2>& a, const Pair2<T1, T2>& b) {
  return b < a;
}

template <typename T1, typename T2>
constexpr bool operator<=(const Pair2<T1, T2>& a, const Pair2<T1, T2>& b) {
  return !(b < a);
}

template <typename T1, typename T2>
constexpr bool operator>=(const Pair2<T1, T2>& a, const Pair2<T1, T2>& b) {
  return !(a < b);
}

template <typename T1, typename T2>
void swap(Pair2<T1, T2>& a, Pair2<T1, T2>& b) noexcept(noexcept(a.swap(b))) {
  a.swap(b);
}

template <typename T1, typename T2>
constexpr Pair2<std::decay_t<T1>, std::decay_t<T2>> make_pair2(T1&& a, T2&& b) {
  return Pair2<std::decay_t<T1>, std::decay_t<T2>>(std::forward<T1>(a),
                                                   std::forward<T2>(b));
}

}  // namespace mytup

namespace std {

template <typename T1, typename T2>
struct tuple_size<::mytup::Pair2<T1, T2>> : integral_constant<size_t, 2> {};

template <size_t I, typename T1, typename T2>
struct tuple_element<I, ::mytup::Pair2<T1, T2>> {
  using type = ::mytup::pair2_element_t<I, T1, T2>;
};

}  // namespace std
