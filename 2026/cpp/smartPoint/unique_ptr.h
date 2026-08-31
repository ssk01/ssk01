#include <cstddef>
#include <memory>
#include <utility>

using namespace std;

template<typename T>
struct DefaultDeleter {
    void operator()(T* p) {
        delete p;
    }  
};
template <typename T>
struct DefaultDeleter<T[]>{
        void operator()(T* p) {
        delete[] p;
    }  
};

template <typename T, typename  Deleter=DefaultDeleter<T>>
class Uniqlo {
public:
    explicit Uniqlo(T* t): t_(t) {};
    explicit Uniqlo(T* t, Deleter d): t_(t), d_(d) {};
    ~Uniqlo() {
        if (t_ != nullptr) {
            d_(t_);
        }
    }
    Uniqlo(Uniqlo& t) = delete;
    Uniqlo& operator=(const Uniqlo& t) = delete;
    Uniqlo(Uniqlo&& u): t_(u.t_){
        u.t_ = nullptr;
    }
    Uniqlo& operator=(Uniqlo&& u) {
        if (this != &u) {
            d_(t_);
            t_ = u.t_;
            u.t_ = nullptr;
        }
        return *this;
    }
    T& operator*() {
        return *t_;
    }
    T* operator->() {
        return t_;
    }    
    const T& operator*() const {
        return *t_;
    }
    const T* operator->() const {
        return t_;
    }    

    // 3. get() / release() / reset() / swap() — 所有权查看与转移    
    T* get() {
        return t_;
    }
    T* release() {
        T* tmp = t_;
        t_ = nullptr;
        return tmp;
    }

    void reset(T* t) {
        d_(t_);
        t_ = t;
    }
    void swap(Uniqlo& u) {
        std::swap(u.t_, t_);
    }
    explicit operator bool() {
        return t_ != nullptr;
    }
    const Deleter& get_deleter() const {
        return d_;
    }
    Deleter& get_deleter() {
        return d_;
    }
private:

T* t_;
Deleter d_;
};
template< class T, class... Args >
Uniqlo<T> make_uniqlo(Args&&... args ) {
    return Uniqlo<T>(new T(std::forward<Args>(args)...));
}