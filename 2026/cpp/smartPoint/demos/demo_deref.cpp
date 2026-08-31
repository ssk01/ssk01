#include "unique_ptr.h"

#include <iostream>

struct Foo
{
    void bar() { std::cout << "Foo::bar\n"; }
};

void f(const Foo&)
{
    std::cout << "f(const Foo&)\n";
}

int main()
{
    Uniqlo<Foo> ptr(new Foo);

    ptr->bar();
    f(*ptr);
}
