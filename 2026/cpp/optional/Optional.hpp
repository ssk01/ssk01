#ifndef OPTIONAL_HPP
#define OPTIONAL_HPP

#include <new>
#include <exception>

class bad_optional_access : public std::exception {
public:
    const char* what() const noexcept override { return "bad_optional_access"; }
};

template <typename T>
class Optional {
private:
    union Storage {
        T value;
        Storage() {}
        ~Storage() {}
    };

    Storage storage_;
    bool has_value_ = false;

    T* get() noexcept { return &storage_.value; }
    const T* get() const noexcept { return &storage_.value; }

public:
    Optional() noexcept : has_value_(false) {}

    Optional(const T& v) : has_value_(true) {
        new (get()) T(v);
    }

    Optional(const Optional& other) : has_value_(other.has_value_) {
        if (other.has_value_) {
            new (get()) T(*other.get());
        }
    }

    Optional& operator=(const Optional& other) {
        if (this != &other) {
            reset();
            if (other.has_value_) {
                new (get()) T(*other.get());
                has_value_ = true;
            }
        }
        return *this;
    }

    ~Optional() { reset(); }

    bool has_value() const noexcept { return has_value_; }
    explicit operator bool() const noexcept { return has_value_; }

    T& value() {
        if (!has_value_) throw bad_optional_access();
        return *get();
    }

    const T& value() const {
        if (!has_value_) throw bad_optional_access();
        return *get();
    }

    T value_or(const T& fallback) const {
        return has_value_ ? *get() : fallback;
    }

    void reset() {
        if (has_value_) {
            get()->~T();
            has_value_ = false;
        }
    }
};

#endif
