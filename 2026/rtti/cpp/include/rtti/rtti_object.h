#pragma once

#include "type_info.h"
#include "type_registry.h"

#include <stdexcept>
#include <type_traits>

namespace rtti {

class Object {
public:
    explicit Object(const TypeInfo* type) : rtti_type_id_(type->id) {}

    Object() : rtti_type_id_(0) {}

    [[nodiscard]] int type_id() const noexcept { return rtti_type_id_; }

    [[nodiscard]] const TypeInfo& type_of() const {
        const TypeInfo* ti = type_registry().lookup(rtti_type_id_);
        if (ti == nullptr) {
            throw std::runtime_error("Object::type_of: invalid type id");
        }
        return *ti;
    }

    [[nodiscard]] bool is_instance(const TypeInfo& type) const noexcept {
        return type.is_assignable_from(rtti_type_id_);
    }

    [[nodiscard]] bool is_instance(std::string_view type_name) const noexcept {
        const TypeInfo* ti = type_registry().lookup(type_name);
        return ti != nullptr && is_instance(*ti);
    }

    template <typename T>
    [[nodiscard]] T* as() {
        static_assert(std::is_base_of_v<Object, T>,
                      "T must derive from rtti::Object");
        const TypeInfo& target = T::rtti_type_info();
        if (!is_instance(target)) {
            throw std::bad_cast();
        }
        return static_cast<T*>(this);
    }

    template <typename T>
    [[nodiscard]] const T* as() const {
        static_assert(std::is_base_of_v<Object, T>,
                      "T must derive from rtti::Object");
        const TypeInfo& target = T::rtti_type_info();
        if (!is_instance(target)) {
            throw std::bad_cast();
        }
        return static_cast<const T*>(this);
    }

    template <typename T>
    [[nodiscard]] bool is() const noexcept {
        static_assert(std::is_base_of_v<Object, T>,
                      "T must derive from rtti::Object");
        const TypeInfo& target = T::rtti_type_info();
        return is_instance(target);
    }

protected:
    int rtti_type_id_;
};

} // namespace rtti

#define RTTI_ENABLE(ClassName)                                  \
    static const ::rtti::TypeInfo& rtti_type_info() {           \
        static const ::rtti::TypeInfo* ti =                     \
            ::rtti::type_registry().lookup(#ClassName);         \
        return *ti;                                             \
    }                                                           \
    static int rtti_type_id() { return rtti_type_info().id; }   \
    ClassName() { rtti_type_id_ = rtti_type_info().id; }
