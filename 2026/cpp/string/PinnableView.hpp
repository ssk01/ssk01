#ifndef PINNABLE_VIEW_HPP
#define PINNABLE_VIEW_HPP

#include "StringView.hpp"
#include "Cleanable.hpp"
#include <string>
#include <cassert>

class PinnableView : public StringView, public Cleanable {
public:
    PinnableView() { buf_ = &self_space_; }

    explicit PinnableView(std::string* buf) { buf_ = buf; }

    PinnableView(PinnableView&& other) noexcept
        : StringView(std::move(static_cast<StringView&>(other)))
        , Cleanable(std::move(static_cast<Cleanable&>(other)))
        , self_space_(std::move(other.self_space_))
        , buf_((other.buf_ == &other.self_space_) ? &self_space_ : other.buf_)
        , pinned_(other.pinned_) {
        other.buf_ = &other.self_space_;
        other.pinned_ = false;
    }

    PinnableView& operator=(PinnableView&& other) noexcept {
        if (this != &other) {
            Reset();
            static_cast<StringView&>(*this) = std::move(static_cast<StringView&>(other));
            static_cast<Cleanable&>(*this) = std::move(static_cast<Cleanable&>(other));
            self_space_ = std::move(other.self_space_);
            buf_ = (other.buf_ == &other.self_space_) ? &self_space_ : other.buf_;
            pinned_ = other.pinned_;
            other.buf_ = &other.self_space_;
            other.pinned_ = false;
        }
        return *this;
    }

    PinnableView(const PinnableView&) = delete;
    PinnableView& operator=(const PinnableView&) = delete;

    void PinSlice(const StringView& sv, Cleanable* cleanable) {
        assert(!pinned_);
        pinned_ = true;
        data_ = sv.data();
        size_ = sv.size();
        if (cleanable) {
            cleanable->DelegateCleanupsTo(this);
        }
    }

    void PinSlice(const StringView& sv, CleanupFunction fn, void* arg1, void* arg2) {
        assert(!pinned_);
        pinned_ = true;
        data_ = sv.data();
        size_ = sv.size();
        RegisterCleanup(fn, arg1, arg2);
    }

    void PinSelf(const StringView& sv) {
        assert(!pinned_);
        buf_->assign(sv.data(), sv.size());
        data_ = buf_->data();
        size_ = buf_->size();
    }

    void PinSelf() {
        assert(!pinned_);
        data_ = buf_->data();
        size_ = buf_->size();
    }

    void remove_prefix(size_t n) {
        assert(n <= size());
        if (pinned_) {
            data_ += n;
            size_ -= n;
        } else {
            buf_->erase(0, n);
            PinSelf();
        }
    }

    void remove_suffix(size_t n) {
        assert(n <= size());
        if (pinned_) {
            size_ -= n;
        } else {
            buf_->erase(size() - n, n);
            PinSelf();
        }
    }

    void Reset() {
        Cleanable::Reset();
        pinned_ = false;
        size_ = 0;
    }

    std::string* GetSelf() { return buf_; }
    bool IsPinned() const { return pinned_; }

private:
    std::string self_space_;
    std::string* buf_;
    bool pinned_ = false;
};

#endif
