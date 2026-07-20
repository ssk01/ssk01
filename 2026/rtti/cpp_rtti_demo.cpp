/**
 * cpp_rtti_demo.cpp — 模拟 C++ 原生 RTTI 实现 (vtable + type_info + dynamic_cast)
 *
 * C++ 多态对象的内存布局:
 *
 *   ┌──────────┬──────────────┐
 *   │  vptr    │  member data │    ← vptr 指向 vtable (8 字节)
 *   └────┬─────┴──────────────┘
 *        │
 *        ▼
 *   ┌──────────────────────────────────┐
 *   │  vtable                          │
 *   │  [0]: &type_info (RTTI 元数据)    │  ← 第一个槽是 type_info 指针
 *   │  [1]: &virtual_func_1             │
 *   │  [2]: &virtual_func_2             │
 *   │  ...                             │
 *   └──────────┬───────────────────────┘
 *              │
 *              ▼
 *   ┌──────────────────────────────────┐
 *   │  type_info                       │
 *   │  name: "Orc"                     │
 *   │  __base_list: [Enemy, Entity]    │  ← 继承链, dynamic_cast 沿此遍历
 *   └──────────────────────────────────┘
 *
 * dynamic_cast<Orc*>(entity_ptr):
 *   1. 读 entity_ptr 的 vptr → vtable
 *   2. 读 vtable[0] → 对象的 type_info
 *   3. 沿 type_info.__base_list 遍历，找是否 == Orc 的 type_info
 *   4. 找到 → 调整指针偏移 + 返回，找不到 → nullptr
 *
 * 编译: c++ -std=c++17 -O2 cpp_rtti_demo.cpp -o cpp_rtti_demo && ./cpp_rtti_demo
 */

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

// ═══════════════════════════════════════════════════════════════
// 1. type_info — C++ 标准库 <typeinfo> 的模拟实现
// ═══════════════════════════════════════════════════════════════

struct TypeInfo {
    const char*            name;         // 类型名 (mangled name)
    std::vector<TypeInfo*> base_list;    // 直接基类的 type_info 列表

    TypeInfo(const char* n) : name(n) {}
    void add_base(TypeInfo* base) { base_list.push_back(base); }

    // 递归沿继承链查找 target — 模拟 __do_dynamic_cast
    bool is_base_of(const TypeInfo* target) const {
        if (this == target) return true;
        for (auto* base : base_list) {
            if (base->is_base_of(target)) return true;
        }
        return false;
    }
};

// ═══════════════════════════════════════════════════════════════
// 2. vtable — 虚函数表 (第一个槽 = type_info 指针)
// ═══════════════════════════════════════════════════════════════

struct VTable {
    TypeInfo* rtti;   // slot [0]: 指向 type_info
    // 后面是虚函数指针, 这里省略

    VTable(TypeInfo* t) : rtti(t) {}
};

// ═══════════════════════════════════════════════════════════════
// 3. 对象基类 — 带 vptr
// ═══════════════════════════════════════════════════════════════

struct Object {
    VTable* vptr;           // 8 字节 — 每个多态对象的开销
    int     data = 0;

    Object(VTable* vt) : vptr(vt) {}

    // 模拟 dynamic_cast
    template<typename T>
    T* dyn_cast() {
        TypeInfo* my_type    = this->vptr->rtti;     // (1) vptr → vtable → type_info
        TypeInfo* target_type = T::type_info_ptr();   // (2) 目标 type_info
        if (my_type->is_base_of(target_type)) {       // (3) 沿 base_list 遍历
            return static_cast<T*>(this);              // (4) 转换
        }
        return nullptr;
    }

    TypeInfo* type_of() const { return vptr->rtti; }
};

// ═══════════════════════════════════════════════════════════════
// 4. 具体类型 — 每个类型有自己的 type_info + vtable
// ═══════════════════════════════════════════════════════════════

// Entity
inline TypeInfo  ti_Entity{"Entity"};
inline VTable    vt_Entity{&ti_Entity};

struct Entity : Object {
    Entity() : Object(&vt_Entity) {}
    static TypeInfo* type_info_ptr() { return &ti_Entity; }
};

// Player
inline TypeInfo  ti_Player{"Player"};
inline VTable    vt_Player{&ti_Player};

struct Player : Entity {
    Player() { vptr = &vt_Player; }
    static TypeInfo* type_info_ptr() { return &ti_Player; }
};

// Enemy
inline TypeInfo  ti_Enemy{"Enemy"};
inline VTable    vt_Enemy{&ti_Enemy};

struct Enemy : Entity {
    Enemy() { vptr = &vt_Enemy; }
    static TypeInfo* type_info_ptr() { return &ti_Enemy; }
};

// Orc
inline TypeInfo  ti_Orc{"Orc"};
inline VTable    vt_Orc{&ti_Orc};

struct Orc : Enemy {
    Orc() { vptr = &vt_Orc; }
    static TypeInfo* type_info_ptr() { return &ti_Orc; }
};

// Dragon
inline TypeInfo  ti_Dragon{"Dragon"};
inline VTable    vt_Dragon{&ti_Dragon};

struct Dragon : Enemy {
    Dragon() { vptr = &vt_Dragon; }
    static TypeInfo* type_info_ptr() { return &ti_Dragon; }
};

// 初始化继承关系 (编译器在编译时完成, 这里手动模拟)
void init_type_hierarchy() {
    ti_Player.add_base(&ti_Entity);
    ti_Enemy.add_base(&ti_Entity);
    ti_Orc.add_base(&ti_Enemy);
    ti_Dragon.add_base(&ti_Enemy);
}

// ═══════════════════════════════════════════════════════════════
// 5. 实际 C++ dynamic_cast (对比参考)
// ═══════════════════════════════════════════════════════════════

struct NativeEntity  { virtual ~NativeEntity() = default; int data = 0; };
struct NativePlayer : NativeEntity {};
struct NativeEnemy  : NativeEntity {};
struct NativeOrc    : NativeEnemy {};

// ═══════════════════════════════════════════════════════════════
// 6. 测试
// ═══════════════════════════════════════════════════════════════

int main() {
    init_type_hierarchy();

    // ── 内存布局 ──
    std::cout << "sizeof(VTable*)    = " << sizeof(VTable*) << " bytes (vptr)\n";
    std::cout << "sizeof(Object)     = " << sizeof(Object)  << " bytes\n\n";

    // ── 打印 type_info 继承链 ──
    auto print_ti = [](TypeInfo* ti, auto&& self, int indent = 0) -> void {
        printf("%*s%s\n", indent * 2, "", ti->name);
        for (auto* b : ti->base_list) self(b, self, indent + 1);
    };
    std::cout << "── type_info 继承关系 ──\n";
    print_ti(&ti_Orc, print_ti);
    std::cout << "\n";

    // ── 模拟 dynamic_cast (模拟版) ──
    std::cout << "── 模拟 dynamic_cast ──\n";
    Orc orc;
    Entity* e = &orc;

    auto* cast_to_orc = e->dyn_cast<Orc>();
    std::cout << "  Entity* → Orc*:   " << (cast_to_orc ? "✓" : "✗") << "\n";

    auto* cast_to_enemy = e->dyn_cast<Enemy>();
    std::cout << "  Entity* → Enemy*: " << (cast_to_enemy ? "✓" : "✗") << "\n";

    auto* cast_to_player = e->dyn_cast<Player>();
    std::cout << "  Entity* → Player*:" << (cast_to_player ? "✓" : "✗") << "\n";

    auto* cast_to_dragon = e->dyn_cast<Dragon>();
    std::cout << "  Entity* → Dragon*:" << (cast_to_dragon ? "✓" : "✗") << "\n";

    // ── 实际 C++ dynamic_cast ──
    std::cout << "\n── 实际 C++ dynamic_cast ──\n";
    NativeOrc no;
    NativeEntity* ne = &no;

    std::cout << "  NativeEntity* → NativeOrc*:   "
              << (dynamic_cast<NativeOrc*>(ne)    ? "✓" : "✗") << "\n";
    std::cout << "  NativeEntity* → NativeEnemy*: "
              << (dynamic_cast<NativeEnemy*>(ne)  ? "✓" : "✗") << "\n";
    std::cout << "  NativeEntity* → NativePlayer*:"
              << (dynamic_cast<NativePlayer*>(ne) ? "✓" : "✗") << "\n";

    // ── 模拟版 vptr 遍历 trace ──
    std::cout << "\n── dynamic_cast 执行 trace (Orc → find Entity) ──\n";
    std::cout << "  (1) 读 orc.vptr → vtable (addr=" << orc.vptr << ")\n";
    std::cout << "  (2) 读 vtable[0] → type_info: " << orc.vptr->rtti->name << "\n";
    std::cout << "  (3) 沿 base_list 遍历:\n";
    for (auto* ti = orc.type_of(); ti; ) {
        std::cout << "        " << ti->name;
        if (ti == &ti_Entity) { std::cout << " ← 命中!\n"; break; }
        if (ti->base_list.empty()) { std::cout << " → 无更多基类\n"; break; }
        std::cout << " → ";
        ti = ti->base_list[0];
    }

    std::cout << "\n✓ Done.\n";
    return 0;
}
