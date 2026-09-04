#include <atomic>
#include <utility>
using namespace std;

template<typename T>
class ShyPtrBase {
public:
    template<typename Y>
    ShyPtrBase(Y* ptr)
        : ptr_(ptr),
          deleter_([](T* p) { delete static_cast<Y*>(p); }),
          count_(1), cb_cnt(0) {

    }
    int use_count() {
        return count_;
    }
    T* get() {
        return ptr_;
    }
    void reset() {
        count_--;
        if (count_ == 0) {
            deleter_(ptr_);
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
    void (*deleter_)(T*);
    atomic_int count_;
    atomic_int cb_cnt;
};


template<typename T>
class WeakPtr;

template<typename T>
class ShyPtr  {
    friend class WeakPtr<T>;
public:
    ShyPtr() :base_(nullptr) {}
    template<typename Y>
    ShyPtr(Y* ptr) : base_(new ShyPtrBase<T>(ptr)){

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

    T* get() const {
        if (!base_)  return nullptr;
        return base_->get();
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
    WeakPtr(Y* ptr) : base_(new ShyPtrBase<T>(ptr)){
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