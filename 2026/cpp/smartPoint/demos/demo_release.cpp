#include "unique_ptr.h"

#include <cassert>
#include <iostream>

struct Foo
{
    Foo() { std::cout << "Foo\n"; }
    ~Foo() { std::cout << "~Foo\n"; }
};

// Ownership of the Foo resource is transferred when calling this function
void legacy_api(Foo* owning_foo)
{
    std::cout << __func__ << '\n';
    delete owning_foo;
}

int main()
{
    Uniqlo<Foo> managed_foo(new Foo);
    legacy_api(managed_foo.release());

    assert(managed_foo == nullptr);
}
