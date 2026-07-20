#include "rtti/type_registry.h"

#include <algorithm>
#include <cassert>

namespace rtti {

TypeRegistry& TypeRegistry::instance() {
    static TypeRegistry reg;
    return reg;
}

TypeInfo* TypeRegistry::define(std::string_view name, TypeInfo* parent) {
    return define(name, parent, 0);
}

TypeInfo* TypeRegistry::define(std::string_view name, TypeInfo* parent,
                                std::size_t object_size) {
    if (sealed_) {
        throw std::logic_error("TypeRegistry: already sealed");
    }

    std::string key(name);
    if (by_name_.contains(key)) {
        throw std::logic_error("TypeRegistry: duplicate type '" + key + "'");
    }

    auto* info = new TypeInfo();
    info->name = key;
    info->parent = parent;
    info->id = 0;
    info->high = 0;
    by_name_[key] = info;

    if (parent == nullptr) {
        roots_.push_back(info);
    }

    if (object_size > 0) {
        by_size_[object_size] = info;
    }

    return info;
}

void TypeRegistry::seal() {
    if (sealed_) return;

    ordered_.clear();
    int next_id = 1;

    for (auto* root : roots_) {
        dfs_assign(root, next_id);
    }

    int max_id = next_id - 1;
    by_id_.resize(static_cast<std::size_t>(max_id) + 1);
    for (auto* info : ordered_) {
        by_id_[static_cast<std::size_t>(info->id)] = info;
    }

    sealed_ = true;
}

int TypeRegistry::dfs_assign(TypeInfo* node, int& next_id) {
    int id = next_id++;
    int high = id;

    node->id = id;
    node->high = high;

    std::vector<TypeInfo*> children;
    collect_children(node, children);
    std::sort(children.begin(), children.end(),
              [](const TypeInfo* a, const TypeInfo* b) {
                  return a->name < b->name;
              });

    for (auto* child : children) {
        int child_high = dfs_assign(child, next_id);
        high = child_high;
        node->high = high;
    }

    ordered_.push_back(node);
    return high;
}

void TypeRegistry::collect_children(const TypeInfo* parent,
                                     std::vector<TypeInfo*>& out) {
    for (auto& [name, info] : by_name_) {
        if (info->parent == parent) {
            out.push_back(info);
        }
    }
}

const TypeInfo* TypeRegistry::lookup(int type_id) const noexcept {
    if (type_id < 0 || static_cast<std::size_t>(type_id) >= by_id_.size()) {
        return nullptr;
    }
    return by_id_[static_cast<std::size_t>(type_id)];
}

const TypeInfo* TypeRegistry::lookup(std::string_view name) const noexcept {
    std::string key(name);
    auto it = by_name_.find(key);
    return (it != by_name_.end()) ? it->second : nullptr;
}

bool TypeRegistry::is_assignable_from(const TypeInfo* super,
                                       int type_id) const noexcept {
    return super != nullptr && super->is_assignable_from(type_id);
}

} // namespace rtti
