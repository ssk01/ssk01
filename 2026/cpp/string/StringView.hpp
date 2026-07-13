#ifndef STRING_VIEW_HPP
#define STRING_VIEW_HPP

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <ostream>

class StringView {
public:
    using size_type = std::size_t;
    static constexpr size_type npos = static_cast<size_type>(-1);

    constexpr StringView() noexcept : data_(nullptr), size_(0) {}

    constexpr StringView(const char* s) : data_(s), size_(s ? std::char_traits<char>::length(s) : 0) {}

    StringView(const std::string& s) : data_(s.data()), size_(s.size()) {}

    constexpr StringView(const char* s, size_type count) : data_(s), size_(count) {}

    ~StringView() = default;

    constexpr StringView(const StringView& other) noexcept : data_(other.data_), size_(other.size_) {}

    constexpr StringView& operator=(const StringView& other) noexcept {
        if (this != &other) {
            data_ = other.data_;
            size_ = other.size_;
        }
        return *this;
    }

    constexpr StringView(StringView&& other) noexcept : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    constexpr StringView& operator=(StringView&& other) noexcept {
        if (this != &other) {
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    constexpr const char* data() const noexcept { return data_; }
    constexpr size_type size() const noexcept { return size_; }
    constexpr bool empty() const noexcept { return size_ == 0; }

    constexpr const char* begin() const noexcept { return data_; }
    constexpr const char* end() const noexcept { return data_ + size_; }

    constexpr const char& operator[](size_type pos) const { return data_[pos]; }

    constexpr const char& at(size_type pos) const {
        if (pos >= size_) throw std::out_of_range("StringView::at");
        return data_[pos];
    }

    constexpr const char& front() const { return data_[0]; }
    constexpr const char& back() const { return data_[size_ - 1]; }

    constexpr void remove_prefix(size_type n) { data_ += n; size_ -= n; }
    constexpr void remove_suffix(size_type n) { size_ -= n; }

    constexpr StringView substr(size_type pos = 0, size_type count = npos) const {
        if (pos > size_) throw std::out_of_range("StringView::substr");
        size_type rcount = std::min(count, size_ - pos);
        return StringView(data_ + pos, rcount);
    }

    constexpr size_type find(char ch, size_type pos = 0) const noexcept {
        for (size_type i = pos; i < size_; ++i) {
            if (data_[i] == ch) return i;
        }
        return npos;
    }

    constexpr size_type find(const StringView& sv, size_type pos = 0) const noexcept {
        if (sv.size_ == 0) return pos <= size_ ? pos : npos;
        if (sv.size_ > size_ || pos > size_ - sv.size_) return npos;
        for (size_type i = pos; i <= size_ - sv.size_; ++i) {
            if (std::char_traits<char>::compare(data_ + i, sv.data_, sv.size_) == 0) return i;
        }
        return npos;
    }

    constexpr size_type find_first_not_of(char ch, size_type pos = 0) const noexcept {
        for (size_type i = pos; i < size_; ++i) {
            if (data_[i] != ch) return i;
        }
        return npos;
    }

    constexpr size_type find_last_not_of(char ch, size_type pos = npos) const noexcept {
        if (empty()) return npos;
        size_type start = std::min(pos, size_ - 1);
        for (size_type i = start; ; --i) {
            if (data_[i] != ch) return i;
            if (i == 0) break;
        }
        return npos;
    }

    void swap(StringView& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }

    int compare(const StringView& other) const noexcept {
        size_type minlen = std::min(size_, other.size_);
        int cmp = std::char_traits<char>::compare(data_, other.data_, minlen);
        if (cmp != 0) return cmp;
        if (size_ < other.size_) return -1;
        if (size_ > other.size_) return 1;
        return 0;
    }

    friend bool operator==(const StringView& a, const StringView& b) noexcept { return a.compare(b) == 0; }
    friend bool operator!=(const StringView& a, const StringView& b) noexcept { return a.compare(b) != 0; }
    friend bool operator<(const StringView& a, const StringView& b) noexcept { return a.compare(b) < 0; }
    friend bool operator<=(const StringView& a, const StringView& b) noexcept { return a.compare(b) <= 0; }
    friend bool operator>(const StringView& a, const StringView& b) noexcept { return a.compare(b) > 0; }
    friend bool operator>=(const StringView& a, const StringView& b) noexcept { return a.compare(b) >= 0; }

    friend std::ostream& operator<<(std::ostream& os, const StringView& sv) {
        return os.write(sv.data_, static_cast<std::streamsize>(sv.size_));
    }

private:
    friend class PinnableView;
    const char* data_;
    size_type size_;
};

#endif
