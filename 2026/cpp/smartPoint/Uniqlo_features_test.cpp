#include "unique_ptr.h"

#include <cassert>
#include <iostream>

struct Logged {
    int id;
    static int alive;

    explicit Logged(int i) : id(i) { ++alive; }
    ~Logged() { --alive; }
};
int Logged::alive = 0;

int main() {
    std::cout << "== move ctor ==\n";
    {
        Uniqlo<Logged> a(new Logged(1));
        Uniqlo<Logged> b(std::move(a));
        assert(Logged::alive == 1);
    }
    assert(Logged::alive == 0);

    std::cout << "== move assign ==\n";
    {
        Uniqlo<Logged> a(new Logged(1));
        Uniqlo<Logged> b(new Logged(2));
        b = std::move(a);
    }
    assert(Logged::alive == 0);

    std::cout << "== operator* / -> ==\n";
    {
        Uniqlo<Logged> p(new Logged(3));
        Logged& r = *p;
        r.id = 100;
        assert(p->id == 100);
        const Uniqlo<Logged>& cp = p;
        assert(cp->id == 100);
        assert((*cp).id == 100);
    }

    std::cout << "== get / release ==\n";
    {
        Uniqlo<Logged> p(new Logged(4));
        assert(p.get() != nullptr);
        Logged* raw = p.release();
        assert(p.get() == nullptr);
        assert(Logged::alive == 1);
        delete raw;
    }

    std::cout << "== reset ==\n";
    {
        Uniqlo<Logged> p(new Logged(5));
        p.reset(new Logged(6));
        assert(Logged::alive == 1);
        p.reset(nullptr);
        assert(Logged::alive == 0);
    }

    std::cout << "== swap ==\n";
    {
        Uniqlo<Logged> a(new Logged(7));
        Uniqlo<Logged> b(new Logged(8));
        a.swap(b);
        assert(a->id == 8 && b->id == 7);
    }

    std::cout << "== operator bool ==\n";
    {
        Uniqlo<Logged> p(new Logged(9));
        if (p) std::cout << "    p is truthy\n";
        p.reset(nullptr);
        if (!p) std::cout << "    p is falsy after reset\n";
    }

    assert(Logged::alive == 0);
    std::cout << "\nall tests passed\n";
    return 0;
}
