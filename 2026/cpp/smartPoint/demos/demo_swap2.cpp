#include "unique_ptr.h"

#include <iostream>
#include <string>
#include <utility>

struct Foo {
    Foo(int _val) : val(_val) { std::cout << "Foo...\n"; }
    ~Foo() { std::cout << "~Foo...\n"; }
    std::string print() { return std::to_string(val); }
    int val;
};

int main()
{
    Uniqlo<Foo> p1 = make_uniqlo<Foo>(100);
    Uniqlo<Foo> p2 = make_uniqlo<Foo>(200);
    auto print = [&]() {
        std::cout << " p1=" << (p1 ? p1->print() : "nullptr");
        std::cout << " p2=" << (p2 ? p2->print() : "nullptr") << '\n';
    };
    print();

    std::swap(p1, p2);
    print();

    p1.reset();
    print();

    std::swap(p1, p2);
    print();
}
