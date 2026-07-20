#include "rtti/rtti.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using namespace rtti;

// ---- Model classes ----

struct Entity : Object {
    Entity() : Object() {}
    virtual ~Entity() = default;
};

struct Player : Entity {
    RTTI_ENABLE(Player)
    ~Player() override = default;
};

struct Enemy : Entity {
    RTTI_ENABLE(Enemy)
    ~Enemy() override = default;
};

struct Orc : Enemy {
    RTTI_ENABLE(Orc)
    ~Orc() override = default;
};

struct Dragon : Enemy {
    RTTI_ENABLE(Dragon)
    ~Dragon() override = default;
};

static void register_types() {
    auto& reg = type_registry();
    auto* entity  = reg.define("Entity");
    auto* player  = reg.define("Player", entity);
    auto* enemy   = reg.define("Enemy", entity);
    auto* orc     = reg.define("Orc", enemy);
    auto* dragon  = reg.define("Dragon", enemy);
    reg.seal();
}

constexpr int N_OBJECTS = 100000;
constexpr int N_ITERS   = 100;

static int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
}

int main() {
    register_types();

    const TypeInfo* enemy_type = type_registry().lookup("Enemy");

    // Create objects
    std::vector<std::unique_ptr<Entity>> objects;
    objects.reserve(N_OBJECTS);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 2);

    for (int i = 0; i < N_OBJECTS; i++) {
        switch (dist(rng)) {
            case 0: objects.push_back(std::make_unique<Player>()); break;
            case 1: objects.push_back(std::make_unique<Orc>());    break;
            case 2: objects.push_back(std::make_unique<Dragon>()); break;
        }
    }

    // ---- RTTI interval check ----
    {
        volatile int count = 0;
        auto start = now_ns();
        for (int iter = 0; iter < N_ITERS; iter++) {
            for (auto& obj : objects) {
                if (obj->is_instance(*enemy_type)) {
                    count++;
                }
            }
        }
        auto elapsed = now_ns() - start;
        double ns_per = static_cast<double>(elapsed) / (N_OBJECTS * N_ITERS);
        std::cout << "[RTTI interval]  count=" << count
                  << "  avg " << ns_per << " ns/check\n";
    }

    // ---- RTTI interval raw (direct id compare) ----
    {
        volatile int count = 0;
        int low = enemy_type->id;
        int high = enemy_type->high;
        auto start = now_ns();
        for (int iter = 0; iter < N_ITERS; iter++) {
            for (auto& obj : objects) {
                int id = obj->type_id();
                if (id >= low && id <= high) {
                    count++;
                }
            }
        }
        auto elapsed = now_ns() - start;
        double ns_per = static_cast<double>(elapsed) / (N_OBJECTS * N_ITERS);
        std::cout << "[RTTI raw int]   count=" << count
                  << "  avg " << ns_per << " ns/check\n";
    }

    // ---- C++ dynamic_cast ----
    {
        volatile int count = 0;
        auto start = now_ns();
        for (int iter = 0; iter < N_ITERS; iter++) {
            for (auto& obj : objects) {
                if (dynamic_cast<Enemy*>(obj.get()) != nullptr) {
                    count++;
                }
            }
        }
        auto elapsed = now_ns() - start;
        double ns_per = static_cast<double>(elapsed) / (N_OBJECTS * N_ITERS);
        std::cout << "[dynamic_cast]   count=" << count
                  << "  avg " << ns_per << " ns/check\n";
    }

    // ---- C++ virtual call dispatch (indirect benchmark) ----
    {
        volatile int count = 0;
        auto start = now_ns();
        for (int iter = 0; iter < N_ITERS; iter++) {
            for (auto& obj : objects) {
                if (obj->type_id() > 0) {
                    count++;
                }
            }
        }
        auto elapsed = now_ns() - start;
        double ns_per = static_cast<double>(elapsed) / (N_OBJECTS * N_ITERS);
        std::cout << "[noop baseline]  count=" << count
                  << "  avg " << ns_per << " ns/check\n";
    }

    return 0;
}
