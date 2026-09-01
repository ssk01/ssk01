#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

template <typename T> struct DefaultDeleter {
  DefaultDeleter() = default;
  // 多态转换: default_delete<D> → default_delete<B>（D* 可转 B*）
  template <typename U, std::enable_if_t<std::is_convertible_v<U *, T *>, int> = 0>
  DefaultDeleter(const DefaultDeleter<U> &) {}
  template <typename U, std::enable_if_t<std::is_convertible_v<U *, T *>, int> = 0>
  DefaultDeleter &operator=(const DefaultDeleter<U> &) { return *this; }
  void operator()(T *p) const { delete p; }
};
template <typename T> struct DefaultDeleter<T[]> {
  DefaultDeleter() = default;
  void operator()(T *p) const { delete[] p; }
};

// ============================================================
// deleter 存储层（分派式 EBO，老 libc++ __compressed_pair_elem 思路）
// - 类类型 && 空 && 非final → 继承（EBO，0 字节）
// - 否则（函数指针/引用/非空/final）→ 存成员
// ============================================================
template <typename D>
using CanEbo = std::bool_constant<std::is_class_v<D> && std::is_empty_v<D> && !std::is_final_v<D>>;

// 能当空基类 → 继承
template <typename D, bool = CanEbo<D>::value>
class DeleterHolder : private D {
public:
  DeleterHolder() noexcept = default;
  template <typename D2> explicit DeleterHolder(D2 &&d) noexcept : D(std::forward<D2>(d)) {}
  D &get_deleter() noexcept { return *this; }
  const D &get_deleter() const noexcept { return *this; }
};

// 不能 → 存成员（函数指针、引用、非空类）
template <typename D>
class DeleterHolder<D, false> {
  D d_;

public:
  DeleterHolder() noexcept : d_() {}
  template <typename D2> explicit DeleterHolder(D2 &&d) noexcept : d_(std::forward<D2>(d)) {}
  D &get_deleter() noexcept { return d_; }
  const D &get_deleter() const noexcept { return d_; }
};

// ============================================================
// 公共基类：存储（T* + deleter, EBO）+ 所有权操作
// 单对象和数组共用：两者都存"元素指针 T*"
// ============================================================
template <typename T, typename Deleter>
class UniqloBase : private DeleterHolder<Deleter> {
  using Holder = DeleterHolder<Deleter>;

protected:
  T *t_;

public:
  UniqloBase() noexcept : Holder(), t_(nullptr) {}
  explicit UniqloBase(T *t) noexcept : Holder(), t_(t) {}
  explicit UniqloBase(T *t, Deleter d) : Holder(std::forward<Deleter>(d)), t_(t) {}
  UniqloBase(const UniqloBase &) = delete;
  UniqloBase &operator=(const UniqloBase &) = delete;
  UniqloBase(UniqloBase &&u) noexcept
      : Holder(std::forward<Deleter>(u.get_deleter())), t_(u.release()) {}
  UniqloBase &operator=(UniqloBase &&u) noexcept {
    if (this != &u) {
      get_deleter()(t_);
      t_ = u.t_;
      u.t_ = nullptr;
      get_deleter() = std::forward<Deleter>(u.get_deleter());
    }
    return *this;
  }
  ~UniqloBase() {
    if (t_ != nullptr) {
      get_deleter()(t_);
    }
  }

  T *get() { return t_; }
  const T *get() const { return t_; }
  T *release() {
    T *tmp = t_;
    t_ = nullptr;
    return tmp;
  }
  void reset(T *t = nullptr) {
    if (t_ != nullptr) {
      get_deleter()(t_);
    }
    t_ = t;
  }
  void swap(UniqloBase &u) {
    std::swap(t_, u.t_);
    std::swap(get_deleter(), u.get_deleter());
  }
  explicit operator bool() { return t_ != nullptr; }
  explicit operator bool() const { return t_ != nullptr; }
  const Deleter &get_deleter() const { return Holder::get_deleter(); }
  Deleter &get_deleter() { return Holder::get_deleter(); }
};

// ============================================================
// 单对象 Uniqlo<T>：薄包装，加 operator* / -> 和 多态转换 move
// ============================================================
template <typename T, typename Deleter = DefaultDeleter<T>>
class Uniqlo : public UniqloBase<T, Deleter> {
  using Base = UniqloBase<T, Deleter>;

public:
  using Base::Base;
  using Base::operator=;

  T &operator*() { return *this->t_; }
  const T &operator*() const { return *this->t_; }
  T *operator->() { return this->t_; }
  const T *operator->() const { return this->t_; }

  // 多态转换 move 构造: Uniqlo<U, E> → Uniqlo<T, D>（U* 可转 T*）
  template <typename U, typename E,
            std::enable_if_t<std::is_convertible_v<U *, T *>, int> = 0>
  Uniqlo(Uniqlo<U, E> &&u) noexcept
      : Base(u.release(), std::forward<E>(u.get_deleter())) {}
  template <typename U, typename E,
            std::enable_if_t<std::is_convertible_v<U *, T *>, int> = 0>
  Uniqlo &operator=(Uniqlo<U, E> &&u) noexcept {
    this->reset(u.release());
    this->get_deleter() = std::forward<E>(u.get_deleter());
    return *this;
  }
};

// ============================================================
// 数组 Uniqlo<T[]>：薄包装，加 operator[]，没有 * / ->
// ============================================================
template <typename T, typename Deleter>
class Uniqlo<T[], Deleter> : public UniqloBase<T, Deleter> {
  using Base = UniqloBase<T, Deleter>;

public:
  using Base::Base;
  using Base::operator=;

  T &operator[](std::size_t i) { return this->t_[i]; }
  const T &operator[](std::size_t i) const { return this->t_[i]; }
};

// ============================================================
// make_uniqlo
// ============================================================
// 变参版用 enable_if 排除数组类型，否则 make_uniqlo<int[]>(5) 会精确匹配
// 变参版（int&& 比 int→size_t 转换更优）而走 new int[](5) 报错
template <class T, class... Args,
          std::enable_if_t<!std::is_array_v<T>, int> = 0>
Uniqlo<T> make_uniqlo(Args &&...args) {
  return Uniqlo<T>(new T(std::forward<Args>(args)...));
}
template <class T> Uniqlo<T> make_uniqlo(std::size_t n) {
  return Uniqlo<T>(new std::remove_extent_t<T>[n]);
}

// ============================================================
// 比较运算符（owner 语义: 比的是 get() 指针本身）
// ============================================================
template <typename T, typename D, typename U, typename E>
bool operator==(const Uniqlo<T, D> &a, const Uniqlo<U, E> &b) {
  return a.get() == b.get();
}
template <typename T, typename D>
bool operator==(const Uniqlo<T, D> &a, std::nullptr_t) { return !a; }
template <typename T, typename D>
bool operator==(std::nullptr_t, const Uniqlo<T, D> &a) { return !a; }
template <typename T, typename D, typename U, typename E>
bool operator!=(const Uniqlo<T, D> &a, const Uniqlo<U, E> &b) {
  return a.get() != b.get();
}
template <typename T, typename D>
bool operator!=(const Uniqlo<T, D> &a, std::nullptr_t) { return (bool)a; }
template <typename T, typename D>
bool operator!=(std::nullptr_t, const Uniqlo<T, D> &a) { return (bool)a; }
template <typename T, typename D, typename U, typename E>
bool operator<(const Uniqlo<T, D> &a, const Uniqlo<U, E> &b) {
  return a.get() < b.get();
}
template <typename T, typename D, typename U, typename E>
bool operator>(const Uniqlo<T, D> &a, const Uniqlo<U, E> &b) { return b < a; }
template <typename T, typename D, typename U, typename E>
bool operator<=(const Uniqlo<T, D> &a, const Uniqlo<U, E> &b) { return !(b < a); }
template <typename T, typename D, typename U, typename E>
bool operator>=(const Uniqlo<T, D> &a, const Uniqlo<U, E> &b) { return !(a < b); }

// ============================================================
// std::hash 特化（单一特化同时覆盖单对象和数组）
// ============================================================
namespace std {
template <typename T, typename D>
struct hash<::Uniqlo<T, D>> {
  size_t operator()(const ::Uniqlo<T, D> &u) const noexcept {
    return hash<decltype(u.get())>()(u.get());
  }
};
} // namespace std
