#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace rtti {

struct TypeInfo {
    int id = 0;
    int high = 0;
    std::string name;
    const TypeInfo* parent = nullptr;

    bool is_assignable_from(int type_id) const noexcept {
        return type_id >= id && type_id <= high;
    }

    bool is_assignable_from(const TypeInfo& other) const noexcept {
        return is_assignable_from(other.id);
    }

    bool is_assignable_to(const TypeInfo& other) const noexcept {
        return other.is_assignable_from(id);
    }

    std::string hierarchy_path() const {
        std::string path = name;
        for (const TypeInfo* p = parent; p != nullptr; p = p->parent) {
            path = p->name + " -> " + path;
        }
        return path;
    }
};

} // namespace rtti
