#pragma once

#include "type_info.h"

#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rtti {

class TypeRegistry {
public:
    static TypeRegistry& instance();

    TypeInfo* define(std::string_view name, TypeInfo* parent = nullptr);

    TypeInfo* define(std::string_view name, TypeInfo* parent,
                     std::size_t object_size);

    void seal();

    [[nodiscard]] bool is_sealed() const noexcept { return sealed_; }

    const TypeInfo* lookup(int type_id) const noexcept;

    const TypeInfo* lookup(std::string_view name) const noexcept;

    [[nodiscard]] bool is_assignable_from(const TypeInfo* super,
                                          int type_id) const noexcept;

    [[nodiscard]] const std::vector<TypeInfo*>& all_types() const noexcept {
        return ordered_;
    }

private:
public:
    TypeRegistry() = default;

private:

    int dfs_assign(TypeInfo* node, int& next_id);

    void collect_children(const TypeInfo* parent,
                           std::vector<TypeInfo*>& out);

    std::unordered_map<std::string, TypeInfo*> by_name_;
    std::unordered_map<std::size_t, TypeInfo*> by_size_;
    std::vector<TypeInfo*> roots_;
    std::vector<TypeInfo*> ordered_;
    std::vector<TypeInfo*> by_id_; // 1-indexed
    bool sealed_ = false;
};

inline TypeRegistry& type_registry() { return TypeRegistry::instance(); }

} // namespace rtti
