#include <atomic>
#include <utility>
using namespace std;

template<typename T>
class ShyPtrBase {
public:
    ShyPtrBase(T* ptr) : ptr_(ptr), count_(1) {

    }
    int use_count() {
        return count_;
    }
    T* get() {
        return ptr_;
    }
    void reset() {
        count_--;
        // cb_cnt--;
        if (count_ == 0) {
            delete ptr_;
        }
    };
    void increase_count() {
        count_++;
    }
    bool expired() {
        return cb_cnt == 0;
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


template<typename T>
class WeakPtr;

template<typename T>
class ShyPtr  {
public:
    ShyPtr() :base_(nullptr) {}
    ShyPtr(T* ptr) : base_(new ShyPtrBase<T>(ptr)){

    }
    ~ShyPtr() {
        if (base_ != nullptr) {
            base_->reset();
            if (base_->expired()) {
                delete base_;
            }
        }
    }
    ShyPtr(const WeakPtr<T>& ptr) {
        base_ = ptr.base_;
        if (base_ != nullptr) {
            base_->increase_count();
        }
    }
    ShyPtr(ShyPtr& ptr) {
        base_ = ptr.base_;
        base_->increase_count();
    }
    ShyPtr(ShyPtr&& ptr) {
        base_ = ptr.base_;
        ptr.base_ = nullptr;
    }

    ShyPtr& operator=(ShyPtr& ptr) {
        base_ = ptr.base_;
        base_->increase_count();
    }
    ShyPtr& operator=(ShyPtr&& ptr) {
        base_ = ptr.base_;
        ptr.base_ = nullptr;
    }
    int use_count() {
        if (!base_) return 0;
        return base_->use_count();
    }
    void reset() {
        base_->reset();
        base_ = nullptr;   
    }

    T* get() {
        if (!base_)  return nullptr;
        return base_->get();
    }

private:
    ShyPtrBase<T> *base_;
};

template<typename T>
class WeakPtr  {
public:
    WeakPtr() :base_(nullptr) {}
    WeakPtr(T* ptr) : base_(new ShyPtrBase<T>(ptr)){
        base_->increaseCb();
    }
    ~WeakPtr() {
        if (base_ != nullptr) {
            base_->releaseCb();
            if (base_->expired()) {
                delete base_;
            }
        }
    }
    WeakPtr(WeakPtr& ptr) {
        base_ = ptr.base_;
        base_->increaseCb();
    }
    WeakPtr(WeakPtr&& ptr) {
        base_ = ptr.base_;
        ptr.base_ = nullptr;
    }
    WeakPtr& operator=(const ShyPtr<T>& ptr) {
        base_ = ptr.base_;
        if (base_ != nullptr) {
            base_->increaseCb();
        }
    }

    WeakPtr& operator=(WeakPtr& ptr) {
        base_ = ptr.base_;
        if (base_ != nullptr) {
            base_->increaseCb();
        }
    }
    WeakPtr& operator=(WeakPtr&& ptr) {
        base_ = ptr.base_;
        ptr.base_ = nullptr;
    }
    int use_count() {
        if (!base_) return 0;
        return base_->use_count();
    }
    void reset() {
        base_->releaseCb();
        base_ = nullptr;   
    }
    bool expired() {
        if (base_ == nullptr) {
            return true;
        } else {
            return base_->use_count() == 0;
        }
    }
    ShyPtr<T>& lock() {
        if (expired()) {
            return ShyPtr<T>{};
        } else {
            return ShyPtr<T>(*this);
        }
    }
    
private:
    ShyPtrBase<T> *base_;
};

// todo. const shyptr 怎么解决常量引用的问题?
//