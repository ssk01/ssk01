#ifndef CLEANABLE_HPP
#define CLEANABLE_HPP

#include <utility>
#include <cassert>

class Cleanable {
public:
    using CleanupFunction = void (*)(void* arg1, void* arg2);

    Cleanable() = default;

    ~Cleanable() { DoCleanup(); }

    Cleanable(const Cleanable&) = delete;
    Cleanable& operator=(const Cleanable&) = delete;

    Cleanable(Cleanable&& other) noexcept
        : cleanup_(other.cleanup_) {
        other.cleanup_ = {nullptr, nullptr, nullptr, nullptr};
    }

    Cleanable& operator=(Cleanable&& other) noexcept {
        if (this != &other) {
            DoCleanup();
            cleanup_ = other.cleanup_;
            other.cleanup_ = {nullptr, nullptr, nullptr, nullptr};
        }
        return *this;
    }

    void RegisterCleanup(CleanupFunction fn, void* arg1, void* arg2) {
        if (cleanup_.function == nullptr) {
            cleanup_.function = fn;
            cleanup_.arg1 = arg1;
            cleanup_.arg2 = arg2;
        } else {
            cleanup_.next = new CleanupNode{cleanup_.function, cleanup_.arg1, cleanup_.arg2, cleanup_.next};
            cleanup_.function = fn;
            cleanup_.arg1 = arg1;
            cleanup_.arg2 = arg2;
        }
    }

    void DelegateCleanupsTo(Cleanable* other) {
        if (cleanup_.function == nullptr) return;
        other->RegisterCleanup(cleanup_.function, cleanup_.arg1, cleanup_.arg2);
        for (CleanupNode* node = cleanup_.next; node != nullptr; ) {
            other->RegisterCleanup(node->function, node->arg1, node->arg2);
            CleanupNode* next = node->next;
            delete node;
            node = next;
        }
        cleanup_.function = nullptr;
        cleanup_.next = nullptr;
    }

    void Reset() {
        DoCleanup();
        cleanup_.function = nullptr;
        cleanup_.next = nullptr;
    }

    bool HasCleanups() const { return cleanup_.function != nullptr; }

private:
    struct CleanupNode {
        CleanupFunction function;
        void* arg1;
        void* arg2;
        CleanupNode* next;
    };

    CleanupNode cleanup_{nullptr, nullptr, nullptr, nullptr};

    void DoCleanup() {
        if (cleanup_.function) {
            cleanup_.function(cleanup_.arg1, cleanup_.arg2);
            for (CleanupNode* node = cleanup_.next; node != nullptr; ) {
                node->function(node->arg1, node->arg2);
                CleanupNode* next = node->next;
                delete node;
                node = next;
            }
        }
    }
};

#endif
