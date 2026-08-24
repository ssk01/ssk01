#include "Optional.hpp"
#include <iostream>
#include <string>

struct Logged {
    int id;
    static int alive;

    Logged(int i) : id(i) {
        ++alive;
        std::cout << "    Logged(" << id << ") constructed, alive=" << alive << "\n";
    }
    Logged(const Logged& other) : id(other.id) {
        ++alive;
        std::cout << "    Logged(" << id << ") copy-constructed, alive=" << alive << "\n";
    }
    ~Logged() {
        --alive;
        std::cout << "    Logged(" << id << ") destroyed, alive=" << alive << "\n";
    }
};
int Logged::alive = 0;

int main() {
    std::cout << "== 1. default construct: empty, and T is NOT constructed ==\n";
    std::cout << "    (no Logged ctor log below = T never constructed)\n";
    Optional<Logged> o1;
    std::cout << "    o1.has_value() = " << o1.has_value() << "\n\n";

    std::cout << "== 2. construct with value (placement new) ==\n";
    Optional<Logged> o2(42);
    std::cout << "    o2.has_value() = " << o2.has_value() << "\n";
    std::cout << "    o2.value().id = " << o2.value().id << "\n\n";

    std::cout << "== 3. operator bool ==\n";
    if (o2) std::cout << "    o2 is truthy\n";
    if (!o1) std::cout << "    o1 is falsy\n\n";

    std::cout << "== 4. value() on empty throws ==\n";
    try {
        o1.value();
    } catch (const bad_optional_access& e) {
        std::cout << "    caught: " << e.what() << "\n\n";
    }

    std::cout << "== 5. value_or ==\n";
    Optional<int> a;
    Optional<int> b(7);
    std::cout << "    a.value_or(99) = " << a.value_or(99) << "\n";
    std::cout << "    b.value_or(99) = " << b.value_or(99) << "\n\n";

    std::cout << "== 6. reset: explicit ~T, then flag cleared ==\n";
    std::cout << "    o2 has value, calling reset():\n";
    o2.reset();
    std::cout << "    o2.has_value() = " << o2.has_value() << "\n\n";

    std::cout << "== 7. copy construct ==\n";
    Optional<Logged> o3(100);
    Optional<Logged> o4(o3);
    std::cout << "    o4.value().id = " << o4.value().id << "\n\n";

    std::cout << "== 8. copy assign (reset + rebuild) ==\n";
    Optional<Logged> o5;
    o5 = o4;
    std::cout << "    o5.value().id = " << o5.value().id << "\n\n";

    std::cout << "== 9. sizeof: union storage + 1 bool ==\n";
    std::cout << "    sizeof(Logged)           = " << sizeof(Logged) << "\n";
    std::cout << "    sizeof(Optional<Logged>)  = " << sizeof(Optional<Logged>) << "\n";
    std::cout << "    (extra bytes = the 'has value' bool, same as std::optional)\n\n";

    std::cout << "== 10. use with std::string ==\n";
    Optional<std::string> name;
    std::cout << "    empty name -> " << name.value_or("(none)") << "\n";
    name = std::string("C++17");
    std::cout << "    name -> " << name.value() << "\n";

    std::cout << "\n== main ends: o5/o4/o3 destroyed, alive back to 0 ==\n";
    return 0;
}
