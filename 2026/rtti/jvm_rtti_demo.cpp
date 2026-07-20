/**
 * jvm_rtti_demo.cpp — 模拟 JVM HotSpot 的 RTTI 实现
 *
 * JVM 对象头：
 *   ┌──────────────┬──────────────┐
 *   │  Mark Word   │  klass ptr   │  ← 压缩后 4 字节
 *   └──────────────┴──────────────┘
 *
 * instanceof 流程 (HotSpot InstanceKlass::is_subtype_of)：
 *   1) walk _super 链                        (主继承, O(depth))
 *   2) 查 _secondary_super_cache              (热路径缓存, O(1))
 *   3) scan _secondary_supers[]               (接口列表, O(#interfaces))
 *
 * 编译: c++ -std=c++17 -O2 jvm_rtti_demo.cpp -o jvm_rtti_demo && ./jvm_rtti_demo
 */

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

// ═══════════════════════════════════════════════════════════════
// 1. Klass — 类型元数据 (对应 HotSpot 的 InstanceKlass)
// ═══════════════════════════════════════════════════════════════

struct Klass {
    std::string  _name;
    Klass*       _super   = nullptr;
    int          _depth   = 0;

    std::vector<Klass*> _secondary_supers;   // 实现的接口列表

    // 热路径缓存 — HotSpot 的真字段
    mutable Klass* _secondary_super_cache = nullptr;

    Klass(std::string name, Klass* super = nullptr)
        : _name(std::move(name)), _super(super) {
        if (super) _depth = super->_depth + 1;
    }

    void add_interface(Klass* iface) {
        _secondary_supers.push_back(iface);
    }

    void inherit_interfaces_from(Klass* parent) {
        _secondary_supers = parent->_secondary_supers;
    }

    // ── 核心：判断 this 是不是 target 的子类型 ──
    bool is_subtype_of(Klass* target) const {
        // 1) 走主继承链 _super
        for (const Klass* k = this; k != nullptr; k = k->_super) {
            if (k == target) return true;
        }

        // 2) 查缓存
        if (_secondary_super_cache == target) return true;

        // 3) 扫描接口列表
        for (auto* iface : _secondary_supers) {
            if (iface == target) {
                _secondary_super_cache = target;  // ← 写入热路径缓存
                return true;
            }
        }
        return false;
    }
};

// ═══════════════════════════════════════════════════════════════
// 2. Object — 模拟 JVM 对象 (带 klass ptr)
// ═══════════════════════════════════════════════════════════════

struct Object {
    Klass* _klass;

    Object(Klass* k) : _klass(k) {}

    bool instanceof(Klass* target) const {
        return _klass->is_subtype_of(target);
    }

    template <typename T>
    T* dyn_cast() {
        if (!instanceof(T::klass_ptr())) throw std::bad_cast();
        return static_cast<T*>(this);
    }
};

// ═══════════════════════════════════════════════════════════════
// 3. 类型定义 (对应 JVM 类加载时生成的 InstanceKlass)
// ═══════════════════════════════════════════════════════════════

// ── 接口 ──
struct IDrawableKlass {
    inline static Klass k{"IDrawable"};
};
struct ICollidableKlass {
    inline static Klass k{"ICollidable"};
};

// ── 类 ──
struct EntityKlass {
    inline static Klass k{"Entity"};
};

struct PlayerKlass {
    inline static Klass k{"Player", &EntityKlass::k};
};

struct EnemyKlass {
    inline static Klass k{"Enemy", &EntityKlass::k};
};

struct OrcKlass {
    inline static Klass k{"Orc", &EnemyKlass::k};
};

struct DragonKlass {
    inline static Klass k{"Dragon", &EnemyKlass::k};
};

// ── 接口实现关系 ──
static void init_interfaces() {
    PlayerKlass::k.add_interface(&IDrawableKlass::k);
    EnemyKlass::k.add_interface(&ICollidableKlass::k);
    OrcKlass::k.inherit_interfaces_from(&EnemyKlass::k);
    DragonKlass::k.inherit_interfaces_from(&EnemyKlass::k);
}

// ── 业务类 ──
struct Entity : Object {
    Entity() : Object(&EntityKlass::k) {}
    static Klass* klass_ptr() { return &EntityKlass::k; }
};

struct Player : Entity {
    Player() { _klass = &PlayerKlass::k; }
    static Klass* klass_ptr() { return &PlayerKlass::k; }
};

struct Enemy : Entity {
    Enemy() { _klass = &EnemyKlass::k; }
    static Klass* klass_ptr() { return &EnemyKlass::k; }
};

struct Orc : Enemy {
    Orc() { _klass = &OrcKlass::k; }
    static Klass* klass_ptr() { return &OrcKlass::k; }
};

struct Dragon : Enemy {
    Dragon() { _klass = &DragonKlass::k; }
    static Klass* klass_ptr() { return &DragonKlass::k; }
};

// ═══════════════════════════════════════════════════════════════
// 4. 测试
// ═══════════════════════════════════════════════════════════════

int main() {
    init_interfaces();

    auto check = [](const Klass& a, const Klass& b, bool expected) {
        bool result = a.is_subtype_of(const_cast<Klass*>(&b));
        const char* st = (result == expected) ? "PASS" : "FAIL";
        std::cout << "  [" << st << "] " << a._name
                  << " instanceof " << b._name
                  << " = " << (result ? "true" : "false") << "\n";
    };

    std::cout << "── 主继承链 (_super) ──\n";
    check(OrcKlass::k,    EnemyKlass::k,   true);
    check(OrcKlass::k,    EntityKlass::k,  true);
    check(PlayerKlass::k, EnemyKlass::k,   false);
    check(EnemyKlass::k,  PlayerKlass::k,  false);
    check(EntityKlass::k, OrcKlass::k,     false);

    std::cout << "\n── 接口 (_secondary_supers) ──\n";
    check(OrcKlass::k,    ICollidableKlass::k, true);
    check(DragonKlass::k, ICollidableKlass::k, true);
    check(PlayerKlass::k, ICollidableKlass::k, false);
    check(PlayerKlass::k, IDrawableKlass::k,   true);

    std::cout << "\n── 运行时对象 instanceof ──\n";
    Orc o;
    assert(o.instanceof(&OrcKlass::k));
    assert(o.instanceof(&EnemyKlass::k));
    assert(o.instanceof(&EntityKlass::k));
    assert(o.instanceof(&ICollidableKlass::k));
    assert(!o.instanceof(&PlayerKlass::k));
    assert(!o.instanceof(&IDrawableKlass::k));
    std::cout << "  [PASS] Orc instanceof all checks\n";

    std::cout << "\n── 安全转型 ──\n";
    Entity* e = &o;
    Orc* c = e->dyn_cast<Orc>();
    assert(c != nullptr);
    std::cout << "  [PASS] Entity → Orc\n";

    bool caught = false;
    try { e->dyn_cast<Player>(); } catch (std::bad_cast&) { caught = true; }
    assert(caught);
    std::cout << "  [PASS] Entity → Player throws\n";

    std::cout << "\n── _secondary_super_cache 热路径缓存 ──\n";
    std::cout << "  cache 前: " << OrcKlass::k._secondary_super_cache << "\n";
    OrcKlass::k.is_subtype_of(&ICollidableKlass::k);
    std::cout << "  cache 后: " << OrcKlass::k._secondary_super_cache
              << "  → 命中 ICollidable(" << &ICollidableKlass::k << ")\n";

    std::cout << "\n✓ All tests passed.\n";
    return 0;
}
