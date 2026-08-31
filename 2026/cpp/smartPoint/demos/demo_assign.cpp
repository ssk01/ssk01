#include "unique_ptr.h"

#include <iostream>
#include <utility>

struct Foo
{
    int id;
    Foo(int id) : id(id) { std::cout << "Foo " << id << '\n'; }
    ~Foo() { std::cout << "~Foo " << id << '\n'; }
};

int main()
{
    Uniqlo<Foo> p1(make_uniqlo<Foo>(1));

    {
        std::cout << "Creating new Foo...\n";
        Uniqlo<Foo> p2(make_uniqlo<Foo>(2));
        // p1 = p2; // Error ! can't copy unique_ptr
        p1 = std::move(p2);
        std::cout << "About to leave inner block...\n";
    }

    std::cout << "About to leave program...\n";
}
