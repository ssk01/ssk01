#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

// 注意：实现放 namespace mytup，而不是全局。
// 全局放 `get`/`apply` 会和 std 撞名——libc++ 的 <memory> 会传递 include <tuple>，
// `using namespace std;` 之后 `std::apply` 可见且对 mytup::Tuple 可用（我们特化了
// tuple_size），非限定的 `apply(...)` 会歧义。进命名空间后无此问题。

namespace mytup {

// ===== 类型工具 =====

// std::ref 包装的类型要解包成引用（make_tuple 语义）
template <typename T>
struct UnwrapRef {
  using type = T;
};
template <typename T>
struct UnwrapRef<std::reference_wrapper<T>> {
  using type = T&;
};
template <typename T>
using unwrap_ref_decay_t = typename UnwrapRef<std::decay_t<T>>::type;

// 取 Types... 中第 I 个类型
template <std::size_t I, typename... Types>
struct TupleTypeAt;

template <std::size_t I, typename T, typename... Rest>
struct TupleTypeAt<I, T, Rest...> : TupleTypeAt<I - 1, Rest...> {};

template <typename T, typename... Rest>
struct TupleTypeAt<0, T, Rest...> {
  using type = T;
};

template <std::size_t I, typename... Types>
using tuple_type_at_t = typename TupleTypeAt<I, Types...>::type;

// 类型 T 在 Types... 中的下标（要求出现且唯一，由 get<T> 的 static_assert 保证）
template <typename T, typename... Types>
struct TupleTypeIndex;

template <typename T>
struct TupleTypeIndex<T> {
  static constexpr int value = -1;
};

template <typename T, typename... Rest>
struct TupleTypeIndex<T, T, Rest...> {
  static constexpr int value = 0;
};

template <typename T, typename Head, typename... Rest>
struct TupleTypeIndex<T, Head, Rest...> {
  static constexpr int value = TupleTypeIndex<T, Rest...>::value + 1;
};

template <typename... Types>
class Tuple;

// 逐元素检查"转换构造"可行性：From 的 const&（Rvalue=false）或 &&（Rvalue=true）能否构造 To
template <bool Rvalue, typename ToTuple, typename FromTuple>
struct TupleConstructible;

template <bool Rvalue>
struct TupleConstructible<Rvalue, Tuple<>, Tuple<>> : std::true_type {};

template <bool Rvalue, typename T, typename... ToRest, typename F,
          typename... FromRest>
struct TupleConstructible<Rvalue, Tuple<T, ToRest...>, Tuple<F, FromRest...>>
    : std::bool_constant<
          std::is_constructible_v<T, std::conditional_t<Rvalue, F&&, const F&>> &&
          TupleConstructible<Rvalue, Tuple<ToRest...>,
                             Tuple<FromRest...>>::value> {};

// 逐元素检查"赋值"可行性
template <bool Rvalue, typename ToTuple, typename FromTuple>
struct TupleAssignable;

template <bool Rvalue>
struct TupleAssignable<Rvalue, Tuple<>, Tuple<>> : std::true_type {};

template <bool Rvalue, typename T, typename... ToRest, typename F,
          typename... FromRest>
struct TupleAssignable<Rvalue, Tuple<T, ToRest...>, Tuple<F, FromRest...>>
    : std::bool_constant<
          std::is_assignable_v<T, std::conditional_t<Rvalue, F&&, const F&>> &&
          TupleAssignable<Rvalue, Tuple<ToRest...>,
                          Tuple<FromRest...>>::value> {};

// get<I> 前置声明：类内成员模板（转换构造/赋值）要用，定义在类后
template <std::size_t I, typename T, typename... Rest>
constexpr tuple_type_at_t<I, T, Rest...>& get(Tuple<T, Rest...>& t) noexcept;
template <std::size_t I, typename T, typename... Rest>
constexpr const tuple_type_at_t<I, T, Rest...>& get(
    const Tuple<T, Rest...>& t) noexcept;
template <std::size_t I, typename T, typename... Rest>
constexpr tuple_type_at_t<I, T, Rest...>&& get(Tuple<T, Rest...>&& t) noexcept;

// ===== Tuple 本体：递归存储 =====
// Tuple<T, Rest...> 继承 Tuple<Rest...>，自己只持有一个 head_。
// tail() 上转型就是 Rest... 部分；空元素靠 EBO（继承链上的空基类子对象压成 0 字节）
// 压缩；引用/函数指针元素用 [[no_unique_address]] 成员存（继承式做不了这两种）。
template <>
class Tuple<> {
public:
  void swap(Tuple&) noexcept {}
};

template <typename T, typename... Rest>
class Tuple<T, Rest...> : public Tuple<Rest...> {
  [[no_unique_address]] T head_;

public:
  // 默认构造要"值初始化"所有元素（std::tuple 语义，int→0、string→""）
  // 含引用元素的 tuple 该构造自然不可用（head_() 对引用非法）
  Tuple() : Tuple<Rest...>(), head_() {}
  Tuple(const Tuple&) = default;
  Tuple(Tuple&&) = default;
  // 声明了 move 构造后隐式拷贝赋值会被 delete，必须显式补回
  // （含引用元素的 tuple 这两个会自然被 delete，与 std 一致）
  Tuple& operator=(const Tuple&) = default;
  Tuple& operator=(Tuple&&) = default;

  // 值构造：head 存第一个元素，剩下的完美转发给基类 Tuple<Rest...>
  // ① enable_if 排除"首参是 Tuple"，否则转发模板（identity 绑定）比拷贝构造
  //    （identity+const 限定）更优，把拷贝/移动偷走
  // ② 条件 explicit：只有某个元素不可隐式转换时才 explicit（C++20 explicit(E)）
  template <typename U, typename... URest,
            std::enable_if_t<!std::is_same_v<std::decay_t<U>, Tuple>, int> = 0>
  explicit(!std::is_convertible_v<U, T> ||
           (... || !std::is_convertible_v<URest, Rest>))
  Tuple(U&& u, URest&&... rest)
      : Tuple<Rest...>(std::forward<URest>(rest)...),
        head_(std::forward<U>(u)) {}

  // 转换构造：从不同元素类型的 Tuple 逐元素构造（首元素进 head，其余转给基类）
  template <typename HeadFrom, typename... TailFrom,
            std::enable_if_t<(sizeof...(TailFrom) == sizeof...(Rest)) &&
                                 TupleConstructible<false, Tuple<T, Rest...>,
                                                    Tuple<HeadFrom, TailFrom...>>::value,
                             int> = 0>
  Tuple(const Tuple<HeadFrom, TailFrom...>& other)
      : Tuple<Rest...>(static_cast<const Tuple<TailFrom...>&>(other)),
        head_(get<0>(other)) {}

  template <typename HeadFrom, typename... TailFrom,
            std::enable_if_t<(sizeof...(TailFrom) == sizeof...(Rest)) &&
                                 TupleConstructible<true, Tuple<T, Rest...>,
                                                    Tuple<HeadFrom, TailFrom...>>::value,
                             int> = 0>
  Tuple(Tuple<HeadFrom, TailFrom...>&& other)
      : Tuple<Rest...>(static_cast<Tuple<TailFrom...>&&>(other)),
        head_(get<0>(std::move(other))) {}

  // pair 构造（2 元素 tuple）；尾元素构造由 Tuple<Rest...> 自己的值构造负责
  template <typename U1, typename U2,
            std::enable_if_t<(sizeof...(Rest) == 1) &&
                                 std::is_constructible_v<T, const U1&>,
                             int> = 0>
  Tuple(const std::pair<U1, U2>& p)
      : Tuple<Rest...>(p.second), head_(p.first) {}

  template <typename U1, typename U2,
            std::enable_if_t<(sizeof...(Rest) == 1) &&
                                 std::is_constructible_v<T, U1&&>,
                             int> = 0>
  Tuple(std::pair<U1, U2>&& p)
      : Tuple<Rest...>(std::forward<U2>(p.second)),
        head_(std::forward<U1>(p.first)) {}

  // 转换赋值
  template <typename HeadFrom, typename... TailFrom,
            std::enable_if_t<(sizeof...(TailFrom) == sizeof...(Rest)) &&
                                 TupleAssignable<false, Tuple<T, Rest...>,
                                                 Tuple<HeadFrom, TailFrom...>>::value,
                             int> = 0>
  Tuple& operator=(const Tuple<HeadFrom, TailFrom...>& other) {
    head_ = get<0>(other);
    static_cast<Tuple<Rest...>&>(*this) =
        static_cast<const Tuple<TailFrom...>&>(other);
    return *this;
  }

  template <typename HeadFrom, typename... TailFrom,
            std::enable_if_t<(sizeof...(TailFrom) == sizeof...(Rest)) &&
                                 TupleAssignable<true, Tuple<T, Rest...>,
                                                 Tuple<HeadFrom, TailFrom...>>::value,
                             int> = 0>
  Tuple& operator=(Tuple<HeadFrom, TailFrom...>&& other) {
    head_ = get<0>(std::move(other));
    static_cast<Tuple<Rest...>&>(*this) =
        static_cast<Tuple<TailFrom...>&&>(other);
    return *this;
  }

  // pair 赋值；尾元素赋值用 get<0>(tail) 完成
  template <typename U1, typename U2,
            std::enable_if_t<(sizeof...(Rest) == 1) &&
                                 std::is_assignable_v<T, const U1&>,
                             int> = 0>
  Tuple& operator=(const std::pair<U1, U2>& p) {
    head_ = p.first;
    get<0>(static_cast<Tuple<Rest...>&>(*this)) = p.second;
    return *this;
  }

  template <typename U1, typename U2,
            std::enable_if_t<(sizeof...(Rest) == 1) &&
                                 std::is_assignable_v<T, U1&&>,
                             int> = 0>
  Tuple& operator=(std::pair<U1, U2>&& p) {
    head_ = std::forward<U1>(p.first);
    get<0>(static_cast<Tuple<Rest...>&>(*this)) = std::forward<U2>(p.second);
    return *this;
  }

  void swap(Tuple& other) noexcept(
      (std::is_nothrow_swappable_v<T> && ... &&
       std::is_nothrow_swappable_v<Rest>)) {
    using std::swap;
    swap(head_, other.head_);
    static_cast<Tuple<Rest...>&>(*this).swap(
        static_cast<Tuple<Rest...>&>(other));
  }

  T& head() { return head_; }
  const T& head() const { return head_; }
  Tuple<Rest...>& tail() { return *this; }
  const Tuple<Rest...>& tail() const { return *this; }
};

// ===== get<I>：递归下钻，第 0 个取 head，其余取 tail 里的 I-1 =====
template <std::size_t I, typename T, typename... Rest>
constexpr tuple_type_at_t<I, T, Rest...>& get(Tuple<T, Rest...>& t) noexcept {
  if constexpr (I == 0)
    return t.head();
  else
    return get<I - 1>(t.tail());
}

template <std::size_t I, typename T, typename... Rest>
constexpr const tuple_type_at_t<I, T, Rest...>& get(
    const Tuple<T, Rest...>& t) noexcept {
  if constexpr (I == 0)
    return t.head();
  else
    return get<I - 1>(t.tail());
}

template <std::size_t I, typename T, typename... Rest>
constexpr tuple_type_at_t<I, T, Rest...>&& get(Tuple<T, Rest...>&& t) noexcept {
  // 用 forward<T> 而非 move：元素是引用（如 int&）时保左值，普通类型才移
  if constexpr (I == 0)
    return std::forward<T>(t.head());
  else
    return get<I - 1>(static_cast<Tuple<Rest...>&&>(t));
}

// ===== get<T>：按类型取（要求类型唯一）=====
template <typename T, typename... Types>
constexpr T& get(Tuple<Types...>& t) noexcept {
  static_assert((0 + ... + std::is_same_v<T, Types>) == 1,
                "type T must occur exactly once in the tuple");
  return get<TupleTypeIndex<T, Types...>::value>(t);
}

template <typename T, typename... Types>
constexpr const T& get(const Tuple<Types...>& t) noexcept {
  static_assert((0 + ... + std::is_same_v<T, Types>) == 1,
                "type T must occur exactly once in the tuple");
  return get<TupleTypeIndex<T, Types...>::value>(t);
}

template <typename T, typename... Types>
constexpr T&& get(Tuple<Types...>&& t) noexcept {
  static_assert((0 + ... + std::is_same_v<T, Types>) == 1,
                "type T must occur exactly once in the tuple");
  return get<TupleTypeIndex<T, Types...>::value>(std::move(t));
}

// ===== 工厂函数 =====
template <typename... Types>
constexpr Tuple<Types&...> tie(Types&... args) noexcept {
  return Tuple<Types&...>(args...);
}

template <typename... Types>
constexpr Tuple<unwrap_ref_decay_t<Types>...> make_tuple(Types&&... t) {
  return Tuple<unwrap_ref_decay_t<Types>...>(std::forward<Types>(t)...);
}

template <typename... Types>
constexpr Tuple<Types&&...> forward_as_tuple(Types&&... args) noexcept {
  return Tuple<Types&&...>(std::forward<Types>(args)...);
}

struct ignore_t {
  template <typename T>
  constexpr void operator=(const T&) const noexcept {}
};
inline constexpr ignore_t ignore{};

// ===== make_from_tuple =====
template <typename T, typename TupleLike, std::size_t... Is>
constexpr T make_from_tuple_impl(TupleLike&& t, std::index_sequence<Is...>) {
  return T(get<Is>(std::forward<TupleLike>(t))...);
}

template <typename T, typename TupleLike>
constexpr T make_from_tuple(TupleLike&& t) {
  return make_from_tuple_impl<T>(
      std::forward<TupleLike>(t),
      std::make_index_sequence<
          std::tuple_size<std::decay_t<TupleLike>>::value>{});
}

// ===== apply：展开成 f(get<0>(t), get<1>(t), ...) =====
template <typename F, typename TupleLike, std::size_t... Is>
constexpr decltype(auto) apply_impl(F&& f, TupleLike&& t,
                                    std::index_sequence<Is...>) {
  return std::invoke(std::forward<F>(f),
                     get<Is>(std::forward<TupleLike>(t))...);
}

template <typename F, typename TupleLike>
constexpr decltype(auto) apply(F&& f, TupleLike&& t) {
  return apply_impl(
      std::forward<F>(f), std::forward<TupleLike>(t),
      std::make_index_sequence<
          std::tuple_size<std::decay_t<TupleLike>>::value>{});
}

// ===== 比较运算符（同类型字典序）=====
template <typename Tuple, std::size_t I>
constexpr bool tuple_less_impl(const Tuple& lhs, const Tuple& rhs) {
  if constexpr (I == std::tuple_size_v<Tuple>) {
    return false;
  } else if (get<I>(lhs) < get<I>(rhs)) {
    return true;
  } else if (get<I>(rhs) < get<I>(lhs)) {
    return false;
  } else {
    return tuple_less_impl<Tuple, I + 1>(lhs, rhs);
  }
}

template <typename Tuple, std::size_t... Is>
constexpr bool tuple_eq_impl(const Tuple& lhs, const Tuple& rhs,
                             std::index_sequence<Is...>) {
  return ((get<Is>(lhs) == get<Is>(rhs)) && ...);
}

template <typename... Types>
constexpr bool operator==(const Tuple<Types...>& lhs,
                          const Tuple<Types...>& rhs) {
  return tuple_eq_impl(lhs, rhs, std::index_sequence_for<Types...>{});
}

template <typename... Types>
constexpr bool operator!=(const Tuple<Types...>& lhs,
                          const Tuple<Types...>& rhs) {
  return !(lhs == rhs);
}

template <typename... Types>
constexpr bool operator<(const Tuple<Types...>& lhs,
                         const Tuple<Types...>& rhs) {
  return tuple_less_impl<Tuple<Types...>, 0>(lhs, rhs);
}

template <typename... Types>
constexpr bool operator>(const Tuple<Types...>& lhs,
                         const Tuple<Types...>& rhs) {
  return rhs < lhs;
}

template <typename... Types>
constexpr bool operator<=(const Tuple<Types...>& lhs,
                          const Tuple<Types...>& rhs) {
  return !(rhs < lhs);
}

template <typename... Types>
constexpr bool operator>=(const Tuple<Types...>& lhs,
                          const Tuple<Types...>& rhs) {
  return !(lhs < rhs);
}

// ===== 自由 swap =====
template <typename... Types>
void swap(Tuple<Types...>& a, Tuple<Types...>& b) noexcept(noexcept(a.swap(b))) {
  a.swap(b);
}

// ===== tuple_cat：拼接任意个 tuple =====
namespace cat_detail {

// 结果类型：把所有 tuple 的元素类型按顺序拼成一个 Tuple
template <typename... Tuples>
struct TupleCatTypes;

template <typename... Acc>
struct TupleCatTypes<Tuple<Acc...>> {
  using type = Tuple<Acc...>;
};

template <typename... Acc, typename T, typename... Rest, typename... Tuples>
struct TupleCatTypes<Tuple<Acc...>, Tuple<T, Rest...>, Tuples...>
    : TupleCatTypes<Tuple<Acc..., T, Rest...>, Tuples...> {};

// 逐 tuple 消耗：把当前 tuple 的元素展开追加到已累积元素之后，
// 最后 Cat<Res>::build(args...) 用所有元素构造 Res
template <typename Res, typename... Tuples>
struct Cat;

template <typename Res>
struct Cat<Res> {
  template <typename... Args>
  static constexpr Res build(Args&&... args) {
    return Res(std::forward<Args>(args)...);
  }
};

template <typename Res, typename Tuple, typename... Rest>
struct Cat<Res, Tuple, Rest...> {
  template <typename T, typename... Tail, std::size_t... Is>
  static constexpr auto build_impl(std::index_sequence<Is...>, T&& t,
                                   Tail&&... tail)
      -> decltype(Cat<Res, Rest...>::build(std::forward<Tail>(tail)...,
                                           get<Is>(std::forward<T>(t))...)) {
    return Cat<Res, Rest...>::build(std::forward<Tail>(tail)...,
                                    get<Is>(std::forward<T>(t))...);
  }

  template <typename T, typename... Tail>
  static constexpr auto build(T&& t, Tail&&... tail)
      -> decltype(build_impl(
          std::make_index_sequence<std::tuple_size<std::decay_t<T>>::value>{},
          std::forward<T>(t), std::forward<Tail>(tail)...)) {
    return build_impl(
        std::make_index_sequence<std::tuple_size<std::decay_t<T>>::value>{},
        std::forward<T>(t), std::forward<Tail>(tail)...);
  }
};

}  // namespace cat_detail

template <typename... Tuples>
constexpr auto tuple_cat(Tuples&&... tpls) {
  using Res = cat_detail::TupleCatTypes<
      Tuple<>, std::decay_t<Tuples>...>::type;
  return cat_detail::Cat<Res, std::decay_t<Tuples>...>::build(
      std::forward<Tuples>(tpls)...);
}

// ===== 推导指引（CTAD）=====
template <typename... Types>
Tuple(Types...) -> Tuple<Types...>;

// pair → 2 元素 tuple 的推导指引（implicit guide 不会从 pair 构造生成，
// 因为其模板参数 U1/U2 推导不出类模板参数 T/Rest）
template <typename U1, typename U2>
Tuple(const std::pair<U1, U2>&) -> Tuple<U1, U2>;

template <typename U1, typename U2>
Tuple(std::pair<U1, U2>&&) -> Tuple<U1, U2>;

}  // namespace mytup

// ===== std::tuple_size / std::tuple_element 特化 =====
// 结构化绑定靠 std::tuple_size 完整 + get 协议，必须放进 namespace std
namespace std {

template <typename... Types>
struct tuple_size<::mytup::Tuple<Types...>>
    : integral_constant<size_t, sizeof...(Types)> {};

template <size_t I, typename... Types>
struct tuple_element<I, ::mytup::Tuple<Types...>> {
  using type = ::mytup::tuple_type_at_t<I, Types...>;
};

}  // namespace std
