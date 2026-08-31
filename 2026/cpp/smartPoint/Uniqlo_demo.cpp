#include "unique_ptr.h"

#include <cassert>
#include <iostream>
#include <type_traits>

struct Logged {
    int id;
    static int alive;
    static int destroyed;

    explicit Logged(int i) : id(i) {
        ++alive;
        std::cout << "    Logged(" << id << ") constructed, alive=" << alive << "\n";
    }
    ~Logged() {
        ++destroyed;
        --alive;
        std::cout << "    Logged(" << id << ") destroyed, alive=" << alive << "\n";
    }
};
int Logged::alive = 0;
int Logged::destroyed = 0;

int main() {
    std::cout << "== 1. construct from raw pointer, dtor deletes ==\n";
    {
        Uniqlo<Logged> p(new Logged(1));
        assert(Logged::alive == 1);
    }
    assert(Logged::alive == 0);
    assert(Logged::destroyed == 1);
    std::cout << "    ok\n\n";

    std::cout << "== 2. nullptr raw pointer: dtor must NOT delete ==\n";
    {
        Uniqlo<Logged> p(nullptr);
        assert(Logged::alive == 0);
    }
    assert(Logged::destroyed == 1);
    std::cout << "    ok\n\n";

    std::cout << "== 3. two objects, both deleted at scope end ==\n";
    {
        Uniqlo<Logged> a(new Logged(10));
        Uniqlo<Logged> b(new Logged(20));
        assert(Logged::alive == 2);
    }
    assert(Logged::alive == 0);
    assert(Logged::destroyed == 3);
    std::cout << "    ok\n\n";

    std::cout << "== 4. copy ctor / copy assign are deleted (compile-time) ==\n";
    std::cout << "    Uniqlo(Uniqlo&)    = delete\n";
    std::cout << "    operator=(Uniqlo&) = delete\n";
    std::cout << "    uncomment the following lines to see the compile error:\n";
    std::cout << "      Uniqlo<Logged> x(new Logged(1));\n";
    std::cout << "      Uniqlo<Logged> y(x);   // error: use of deleted function\n";
    std::cout << "      Uniqlo<Logged> z = x;  // error\n";
    std::cout << "      y = x;                 // error\n\n";

    std::cout << "== 5. sizeof(Uniqlo<T>) == sizeof(T*) == " << sizeof(void*) << "\n";
    std::cout << "    sizeof(Uniqlo<Logged>) = " << sizeof(Uniqlo<Logged>) << "\n\n";

    std::cout << "== 6. dtor is noexcept ==\n";
    static_assert(std::is_nothrow_destructible<Uniqlo<Logged>>::value,
                  "Uniqlo dtor should be noexcept");
    std::cout << "    ok\n\n";

    std::cout << "all tests passed\n";
    return 0;
}
