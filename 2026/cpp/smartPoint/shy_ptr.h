#include <atomic>
#include <type_traits>
#include <utility>
using namespace std;

// 内部默认 deleter（等价 std::default_delete，避免引 <memory>）
template<typename Y>
struct ShyDefaultDelete {
    void operator()(Y* p) const { delete p; }
};

// 控制块基类：只管 T* 指针 + 强/弱两个原子计数 + 虚函数钩子。
// 对 std::shared_ptr 的 __shared_count/__shared_weak_count（指针留在基类便于 get()，
// 官方是放派生里并在 shared_ptr 里另存一份指针，等做 alias 时再对齐）。
template<typename T>
class ShyPtrBase {
public:
    ShyPtrBase(T* ptr) : ptr_(ptr), count_(1), cb_cnt(0) {}
    virtual ~ShyPtrBase() = default;
    virtual void dispose() = 0;   // 强计数归零时删对象（官方 __on_zero_shared）
    int use_count() {
        return count_;
    }
    T* get() {
        return ptr_;
    }
    void reset() {
        
        if (count_.fetch_sub(1) == 1) {
            dispose();
        }
    };
    void increase_count() {
        count_++;
    }
    bool expired() {
        return cb_cnt == 0 && count_ == 0;
    }
    void releaseCb() {
        cb_cnt--;
    }
    void increaseCb() {
        cb_cnt++;
    }
 
private:
    T* ptr_;
    atomic_int count_;
    atomic_int cb_cnt;
};

// 分派式 deleter 持有者：空且非 final 的类型当私有基类（继承式 EBO，0 字节），
// 否则存成员。对 libc++ 旧版 __compressed_pair_elem 的思路。
template<typename D, bool = is_empty<D>::value && !is_final<D>::value>
class ShyDelHolder {
public:
    ShyDelHolder(D d) : del_(std::move(d)) {}
    D& deleter() { return del_; }
private:
    D del_;
};

template<typename D>
class ShyDelHolder<D, true> : private D {
public:
    ShyDelHolder(D d) : D(std::move(d)) {}
    D& deleter() { return *this; }
};

// 具体控制块：按 (T, Y, Deleter) 模板化，deleter 按值存。
// 对 std::shared_ptr 的 __shared_ptr_pointer<_Tp,_Dp,_Alloc>（去掉 allocator）。
template<typename T, typename Y, typename D>
class ShyPtrCtl : public ShyPtrBase<T>, private ShyDelHolder<D> {
public:
    ShyPtrCtl(Y* ptr, D d)
        : ShyPtrBase<T>(ptr), ShyDelHolder<D>(std::move(d)), raw_(ptr) {}
    void dispose() override {
        this->deleter()(raw_);
    }
private:
    Y* raw_;     // 构造时原始类型指针，dispose 时按它调 deleter
};


template<typename T>
class WeakPtr;

template<typename T>
class ShyPtr  {
    friend class WeakPtr<T>;
public:
    ShyPtr() :base_(nullptr) {}
    template<typename Y>
    ShyPtr(Y* ptr) : base_(new ShyPtrCtl<T, Y, ShyDefaultDelete<Y>>(ptr, ShyDefaultDelete<Y>())) {

    }
    template<typename Y, typename D>
    ShyPtr(Y* ptr, D d) : base_(new ShyPtrCtl<T, Y, D>(ptr, std::move(d))) {

    }
    ~ShyPtr() {
        release();
    }
    ShyPtr(const WeakPtr<T>& ptr) {
        base_ = ptr.base_;
        if (base_ != nullptr) {
            base_->increase_count();
        }
    }
    ShyPtr(const ShyPtr& ptr) {
        base_ = ptr.base_;
        if (base_ != nullptr) {
            base_->increase_count();
        }
    }
    ShyPtr(ShyPtr&& ptr) {
        base_ = ptr.base_;
        ptr.base_ = nullptr;
    }

    ShyPtr& operator=(const ShyPtr& ptr) {
        if (this == &ptr) return *this;
        release();
        base_ = ptr.base_;
        if (base_ != nullptr) {
            base_->increase_count();
        }
        return *this;
    }
    ShyPtr& operator=(ShyPtr&& ptr) {
        if (this == &ptr) return *this;
        release();
        base_ = ptr.base_;
        ptr.base_ = nullptr;
        return *this;
    }
    int use_count() const {
        if (!base_) return 0;
        return base_->use_count();
    }
    void reset() {
        release();
    }
    template<typename Y>
    void reset(Y* ptr) {
        release();
        base_ = new ShyPtrCtl<T, Y, ShyDefaultDelete<Y>>(ptr, ShyDefaultDelete<Y>());
    }

    operator bool() const {
        return get() != nullptr;
    }
    T* operator->() const {
        return get();
    }

    T* get() const {
        if (!base_)  return nullptr;
        return base_->get();
    }

    bool  unique() const {
        return use_count() == 1;
    }

    T& operator*() const {
        return *get();
    }

    void swap(ShyPtr& other) {
        std::swap(base_, other.base_);
    }

private:
    void release() {
        if (base_ != nullptr) {
            base_->reset();
            if (base_->expired()) {
                delete base_;
            }
            base_ = nullptr;
        }
    }
    ShyPtrBase<T> *base_;
};

template<typename T>
class WeakPtr  {
    friend class ShyPtr<T>;
public:
    WeakPtr() :base_(nullptr) {}
    template<typename Y>
    WeakPtr(Y* ptr) : base_(new ShyPtrCtl<T, Y, ShyDefaultDelete<Y>>(ptr, ShyDefaultDelete<Y>())){
        base_->increaseCb();
    }
    ~WeakPtr() {
        release();
    }
    WeakPtr(const WeakPtr& ptr) {
        base_ = ptr.base_;
        if (base_ != nullptr) {
            base_->increaseCb();
        }
    }
    WeakPtr(const ShyPtr<T>& ptr) {
        base_ = ptr.base_;
        if (base_ != nullptr) {
            base_->increaseCb();
        }
    }
    WeakPtr(WeakPtr&& ptr) {
        base_ = ptr.base_;
        ptr.base_ = nullptr;
    }
    WeakPtr& operator=(const ShyPtr<T>& ptr) {
        release();
        base_ = ptr.base_;
        if (base_ != nullptr) {
            base_->increaseCb();
        }
        return *this;
    }

    WeakPtr& operator=(const WeakPtr& ptr) {
        if (this == &ptr) return *this;
        release();
        base_ = ptr.base_;
        if (base_ != nullptr) {
            base_->increaseCb();
        }
        return *this;
    }
    WeakPtr& operator=(WeakPtr&& ptr) {
        if (this == &ptr) return *this;
        release();
        base_ = ptr.base_;
        ptr.base_ = nullptr;
        return *this;
    }
    int use_count() const {
        if (!base_) return 0;
        return base_->use_count();
    }
    void reset() {
        release();
    }
    bool expired() const {
        if (base_ == nullptr) {
            return true;
        } else {
            return base_->use_count() == 0;
        }
    }
    ShyPtr<T> lock() const {
        if (expired()) {
            return ShyPtr<T>{};
        } else {
            return ShyPtr<T>(*this);
        }
    }

    void swap(WeakPtr& other) {
        std::swap(base_, other.base_);
    }
    
private:
    void release() {
        if (base_ != nullptr) {
            base_->releaseCb();
            if (base_->expired()) {
                delete base_;
            }
            base_ = nullptr;
        }
    }
    ShyPtrBase<T> *base_;
};
template <typename T, typename... Args>
ShyPtr<T> make_shyptr(Args &&...args) {
    return ShyPtr<T>(new T(std::forward<Args>(args)...));
}

template<class T>
void swap(ShyPtr<T>& lhs, ShyPtr<T>& rhs) {
    lhs.swap(rhs);
}

template<class T>
void swap(WeakPtr<T>& lhs, WeakPtr<T>& rhs) {
    lhs.swap(rhs);
}