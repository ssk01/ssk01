#include "rtti/rtti.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

using namespace rtti;

// ---- Test model classes ----

struct Entity : Object {
    RTTI_ENABLE(Entity)
};

struct Player : Entity {
    RTTI_ENABLE(Player)
};

struct Enemy : Entity {
    RTTI_ENABLE(Enemy)
};

struct Orc : Enemy {
    RTTI_ENABLE(Orc)
};

struct Dragon : Enemy {
    RTTI_ENABLE(Dragon)
};

// ---- Registration ----

static void register_types() {
    auto& reg = type_registry();

    auto* entity  = reg.define("Entity");
    auto* player  = reg.define("Player", entity);
    auto* enemy   = reg.define("Enemy", entity);
    auto* orc     = reg.define("Orc", enemy);
    auto* dragon  = reg.define("Dragon", enemy);

    reg.seal();
}

// ---- Tests ----

static void test_interval_assignment() {
    auto& reg = type_registry();

    const TypeInfo* entity = reg.lookup("Entity");
    const TypeInfo* player = reg.lookup("Player");
    const TypeInfo* enemy  = reg.lookup("Enemy");
    const TypeInfo* orc    = reg.lookup("Orc");
    const TypeInfo* dragon = reg.lookup("Dragon");

    assert(entity->id == 1 && entity->high == 5);
    assert(enemy->id == 2 && enemy->high == 4);
    assert(dragon->id == 3 && dragon->high == 3);
    assert(orc->id == 4 && orc->high == 4);
    assert(player->id == 5 && player->high == 5);

    std::cout << "  [PASS] test_interval_assignment\n";
}

static void test_is_assignable_from() {
    const TypeInfo* entity = type_registry().lookup("Entity");
    const TypeInfo* player = type_registry().lookup("Player");
    const TypeInfo* enemy  = type_registry().lookup("Enemy");
    const TypeInfo* orc    = type_registry().lookup("Orc");
    const TypeInfo* dragon = type_registry().lookup("Dragon");

    assert(entity->is_assignable_from(*player));
    assert(entity->is_assignable_from(*enemy));
    assert(entity->is_assignable_from(*orc));
    assert(entity->is_assignable_from(*dragon));

    assert(!player->is_assignable_from(*entity));
    assert(!player->is_assignable_from(*enemy));
    assert(!player->is_assignable_from(*orc));

    assert(enemy->is_assignable_from(*orc));
    assert(enemy->is_assignable_from(*dragon));
    assert(!enemy->is_assignable_from(*entity));
    assert(!enemy->is_assignable_from(*player));

    std::cout << "  [PASS] test_is_assignable_from\n";
}

static void test_object_type_check() {
    Orc orc;
    const TypeInfo* entity = type_registry().lookup("Entity");
    const TypeInfo* player = type_registry().lookup("Player");
    const TypeInfo* enemy  = type_registry().lookup("Enemy");
    const TypeInfo* orc_t  = type_registry().lookup("Orc");
    const TypeInfo* dragon = type_registry().lookup("Dragon");

    assert(orc.is_instance(*entity));
    assert(!orc.is_instance(*player));
    assert(orc.is_instance(*enemy));
    assert(orc.is_instance(*orc_t));
    assert(!orc.is_instance(*dragon));

    assert(orc.type_of().name == "Orc");

    std::cout << "  [PASS] test_object_type_check\n";
}

static void test_cast_success() {
    Orc orc;
    Entity* e = &orc;

    Orc* o = rtti::cast<Orc>(e);
    assert(o != nullptr);

    std::cout << "  [PASS] test_cast_success\n";
}

static void test_cast_fail() {
    Dragon dragon;
    Entity* e = &dragon;

    bool caught = false;
    try {
        rtti::cast<Orc>(e);
    } catch (const std::bad_cast&) {
        caught = true;
    }
    assert(caught);

    std::cout << "  [PASS] test_cast_fail\n";
}

static void test_is_template() {
    Orc orc;
    assert(orc.is<Orc>());
    assert(orc.is<Enemy>());
    assert(orc.is<Entity>());
    assert(!orc.is<Player>());
    assert(!orc.is<Dragon>());

    std::cout << "  [PASS] test_is_template\n";
}

static void test_huge_hierarchy() {
    TypeRegistry reg;

    TypeInfo* root = reg.define("Root");
    for (int i = 0; i < 1000; i++) {
        reg.define("Child" + std::to_string(i), root);
    }
    reg.seal();

    const TypeInfo* child0 = reg.lookup("Child0");
    assert(child0->id == 2);
    assert(child0->high == 2);
    assert(root->is_assignable_from(*child0));
    assert(!child0->is_assignable_from(*root));

    std::cout << "  [PASS] test_huge_hierarchy\n";
}

static void test_rtti_utility() {
    Player player;
    const TypeInfo* entity = type_registry().lookup("Entity");
    const TypeInfo* enemy  = type_registry().lookup("Enemy");

    assert(rtti::is_instance(&player, *entity));
    assert(!rtti::is_instance(&player, *enemy));
    assert(rtti::type_of(&player)->name == "Player");

    std::cout << "  [PASS] test_rtti_utility\n";
}

int main() {
    register_types();

    std::cout << "Running C++ RTTI tests...\n";
    test_interval_assignment();
    test_is_assignable_from();
    test_object_type_check();
    test_cast_success();
    test_cast_fail();
    test_is_template();
    test_huge_hierarchy();
    test_rtti_utility();
    std::cout << "All tests passed.\n";

    return 0;
}
