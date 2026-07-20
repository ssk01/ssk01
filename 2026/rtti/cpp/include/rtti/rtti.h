#pragma once

#include "rtti/type_info.h"
#include "rtti/type_registry.h"
#include "rtti/rtti_object.h"

namespace rtti {

inline const TypeInfo* type_of(const Object* obj) {
    return obj != nullptr ? &obj->type_of() : nullptr;
}

inline bool is_instance(const Object* obj, const TypeInfo& type) {
    return obj != nullptr && obj->is_instance(type);
}

inline bool is_instance(const Object* obj, std::string_view type_name) {
    return obj != nullptr && obj->is_instance(type_name);
}

inline bool is_assignable_from(const TypeInfo* super, const TypeInfo* sub) {
    return super != nullptr && sub != nullptr && super->is_assignable_from(*sub);
}

template <typename T>
T* cast(Object* obj) {
    if (obj == nullptr) return nullptr;
    return obj->as<T>();
}

template <typename T>
const T* cast(const Object* obj) {
    if (obj == nullptr) return nullptr;
    return obj->as<T>();
}

} // namespace rtti
