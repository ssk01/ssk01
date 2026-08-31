#include <cstddef>
#include <memory>
#include <utility>

using namespace std;

template <typename T> struct DefaultDeleter {
  void operator()(T *p) { delete p; }
};
template <typename T> struct DefaultDeleter<T[]> {
  void operator()(T *p) { delete[] p; }
};

template <typename T, typename Deleter = DefaultDeleter<T>>
class Uniqlo : private Deleter {
public:
  explicit Uniqlo(T *t) : t_(t) {};
  explicit Uniqlo(T *t, Deleter d) : Deleter(d), t_(t) {};
  ~Uniqlo() {
    if (t_ != nullptr) {
      get_deleter()(t_);
    }
  }
  Uniqlo(Uniqlo &t) = delete;
  Uniqlo &operator=(const Uniqlo &t) = delete;
  Uniqlo(Uniqlo &&u) : Deleter(std::forward<Deleter>(u.get_deleter())), t_(u.release()) {}
  Uniqlo &operator=(Uniqlo &&u) {
    if (this != &u) {
      get_deleter()(t_);
      t_ = u.t_;
      u.t_ = nullptr;
    }
    return *this;
  }
  T &operator*() { return *t_; }
  T *operator->() { return t_; }
  const T &operator*() const { return *t_; }
  const T *operator->() const { return t_; }

  // 3. get() / release() / reset() / swap() — 所有权查看与转移
  T *get() { return t_; }
  T *release() {
    T *tmp = t_;
    t_ = nullptr;
    return tmp;
  }

  void reset(T *t) {
    if (t_ != nullptr) {
        get_deleter()(t_);
    }
    t_ = t;
  }
  void swap(Uniqlo &u) { 
    std::swap(u.t_, t_);
    std::swap(u.get_deleter(), get_deleter());
 }
  explicit operator bool() { return t_ != nullptr; }
  const Deleter &get_deleter() const { return *(static_cast<const Deleter *>(this)); }
  Deleter &get_deleter() { return *(static_cast<Deleter *>(this)); }

private:
  T *t_;
};
template <class T, class... Args> Uniqlo<T> make_uniqlo(Args &&...args) {
  return Uniqlo<T>(new T(std::forward<Args>(args)...));
}