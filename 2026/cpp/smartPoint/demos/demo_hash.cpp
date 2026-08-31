#include "unique_ptr.h"

#include <functional>
#include <iostream>

struct Foo
{
    Foo(int num) : nr(num) { std::cout << "Foo(" << nr << ")\n"; }

    ~Foo() { std::cout << "~Foo()\n"; }

    bool operator==(const Foo &other) const { return nr == other.nr; }

    int nr;
};

int main()
{
    std::cout << std::boolalpha << std::hex;

    Foo* foo = new Foo(5);
    Uniqlo<Foo> up(foo);
    std::cout << "hash(up):    " << std::hash<Uniqlo<Foo>>()(up) << '\n'
              << "hash(foo):   " << std::hash<Foo*>()(foo) << '\n'
              << "*up==*foo:   " << (*up == *foo) << "\n\n";

    Uniqlo<Foo> other = make_uniqlo<Foo>(5);
    std::cout << "hash(up):    " << std::hash<Uniqlo<Foo>>()(up) << '\n'
              << "hash(other): " << std::hash<Uniqlo<Foo>>()(other) << '\n'
              << "*up==*other: " << (*up == *other) << "\n\n";
}
