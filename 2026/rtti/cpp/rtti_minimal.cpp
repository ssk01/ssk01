/**
 * rtti_minimal.cpp — 面试级丐版 RTTI，单文件，< 150 行
 *
 * 核心思路：预序遍历类型树分配整数 ID，子类型 ID 落在父类型区间内。
 * instanceof = 两次 int 比较，O(1)，无间接寻址，每对象 4 字节开销。
 *
 * 编译: c++ -std=c++17 -O2 rtti_minimal.cpp -o rtti_minimal
 */

#include <cassert>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

// ─── 1. 类型描述符 ───
struct TypeInfo {
    int id = 0;
    int high = 0;
    std::string name;
    TypeInfo* parent = nullptr;

    bool isAssignableFrom(int tid) const { return tid >= id && tid <= high; }
};

// ─── 2. 类型注册表 ───
class TypeRegistry {
    std::unordered_map<std::string, TypeInfo*> byName;
    std::vector<TypeInfo*> byId;    // byId[0] 空着，ID 从 1 开始
    bool sealed = false;

    int dfs(TypeInfo* node, int& nextId) {
        int cur = nextId++;
        int hi  = cur;
        node->id = cur;
        node->high = hi;

        for (auto& [name, info] : byName) {
            if (info->parent == node) {
                hi = dfs(info, nextId);
                node->high = hi;
            }
        }
        return hi;
    }

public:
    TypeInfo* define(const std::string& name, TypeInfo* parent = nullptr) {
        if (sealed) throw std::logic_error("sealed");
        auto* t = new TypeInfo{0, 0, name, parent};
        byName[name] = t;
        return t;
    }

    void seal() {
        int nextId = 1;
        for (auto& [name, info] : byName)
            if (info->parent == nullptr) dfs(info, nextId);
        int maxId = nextId - 1;
        byId.resize(maxId + 1);
        for (auto& [name, info] : byName) byId[info->id] = info;
        sealed = true;
    }

    const TypeInfo* lookup(int id) const {
        return (id > 0 && id < (int)byId.size()) ? byId[id] : nullptr;
    }
    const TypeInfo* lookup(const std::string& name) const {
        auto it = byName.find(name);
        return it != byName.end() ? it->second : nullptr;
    }
};

// 全局单例
TypeRegistry& registry() { static TypeRegistry r; return r; }

// ─── 3. 基类（每对象 4 字节 int） ───
struct Object {
    int _tid = 0;
    Object() = default;
    explicit Object(const TypeInfo* t) : _tid(t->id) {}

    bool isA(const TypeInfo& t) const { return t.isAssignableFrom(_tid); }

    template<typename T>
    T* as() {
        if (!isA(T::typeInfo())) throw std::bad_cast();
        return static_cast<T*>(this);
    }
};

// ─── 4. 示例：游戏实体层级 ───
struct Entity : Object {
    Entity() = default;
    Entity(const TypeInfo* t) : Object(t) {}
};

struct Player : Entity {
    Player() { _tid = registry().lookup("Player")->id; }
    static const TypeInfo& typeInfo() { return *registry().lookup("Player"); }
};

struct Enemy : Entity {
    Enemy() { _tid = registry().lookup("Enemy")->id; }
    static const TypeInfo& typeInfo() { return *registry().lookup("Enemy"); }
};

struct Orc : Enemy {
    Orc() { _tid = registry().lookup("Orc")->id; }
    static const TypeInfo& typeInfo() { return *registry().lookup("Orc"); }
};

struct Dragon : Enemy {
    Dragon() { _tid = registry().lookup("Dragon")->id; }
    static const TypeInfo& typeInfo() { return *registry().lookup("Dragon"); }
};

// ─── 5. 测试 ───
int main() {
    auto& reg = registry();
    auto* entity = reg.define("Entity");
    auto* player = reg.define("Player", entity);
    auto* enemy  = reg.define("Enemy",  entity);
    auto* orc    = reg.define("Orc",    enemy);
    auto* dragon = reg.define("Dragon", enemy);
    reg.seal();

    // 区间验证
    assert(entity->id == 1 && entity->high == 5);
    assert(enemy->id  == 2 && enemy->high  == 4);
    assert(dragon->id == 3 && dragon->high == 3);
    assert(orc->id    == 4 && orc->high    == 4);
    assert(player->id == 5 && player->high == 5);

    // isAssignableFrom
    assert( entity->isAssignableFrom(orc->id));
    assert( enemy->isAssignableFrom(orc->id));
    assert(!player->isAssignableFrom(orc->id));
    assert(!orc->isAssignableFrom(player->id));

    // 对象类型检查
    Orc o;
    assert(o.isA(*entity));
    assert(o.isA(*enemy));
    assert(o.isA(*orc));
    assert(!o.isA(*player));

    // 安全转型
    Entity* e = &o;
    Orc* casted = e->as<Orc>();
    assert(casted != nullptr);

    bool caught = false;
    try { e->as<Player>(); } catch (std::bad_cast&) { caught = true; }
    assert(caught);

    std::cout << "All tests passed.\n";
    return 0;
}
