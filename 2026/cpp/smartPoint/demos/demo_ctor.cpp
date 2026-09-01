#include "unique_ptr.h"

#include <iostream>
#include <utility>

struct Foo // object to manage
{
    Foo() { std::cout << "Foo ctor\n"; }
    Foo(const Foo&) { std::cout << "Foo copy ctor\n"; }
    Foo(Foo&&) { std::cout << "Foo move ctor\n"; }
    ~Foo() { std::cout << "~Foo dtor\n"; }
};

struct D // deleter
{
    D() {}
    D(const D&) { std::cout << "D copy ctor\n"; }
    D(D&) { std::cout << "D non-const copy ctor\n"; }
    D(D&&) { std::cout << "D move ctor \n"; }
    void operator()(Foo* p) const
    {
        std::cout << "D is deleting a Foo\n";
        delete p;
    }
};

int main()
{
    std::cout << "Example constructor(1)...\n";
    Uniqlo<Foo> up1; // up1 is empty
    Uniqlo<Foo> up1b(nullptr); // up1b is empty

    std::cout << "Example constructor(2)...\n";
    {
        Uniqlo<Foo> up2(new Foo);
    } // Foo deleted

    std::cout << "Example constructor(3)...\n";
    D d;
    {   // deleter type is not a reference
        Uniqlo<Foo, D> up3(new Foo, d); // deleter copied
    }
    {   // deleter type is a reference
        Uniqlo<Foo, D&> up3b(new Foo, d); // up3b holds a reference to d
    }

    std::cout << "Example constructor(4)...\n";
    {   // deleter is not a reference
        Uniqlo<Foo, D> up4(new Foo, D()); // deleter moved
    }

    std::cout << "Example constructor(5)...\n";
    {
        Uniqlo<Foo> up5a(new Foo);
        Uniqlo<Foo> up5b(std::move(up5a)); // ownership transfer
    }

    std::cout << "Example constructor(6)...\n";
    {
        Uniqlo<Foo, D> up6a(new Foo, d); // D is copied
        Uniqlo<Foo, D> up6b(std::move(up6a)); // D is moved

        Uniqlo<Foo, D&> up6c(new Foo, d); // D is a reference
        Uniqlo<Foo, D> up6d(std::move(up6c)); // D is copied
    }

    std::cout << "Example array constructor...\n";
    {
        Uniqlo<Foo[]> up(new Foo[3]);
    } // three Foo objects deleted
}
