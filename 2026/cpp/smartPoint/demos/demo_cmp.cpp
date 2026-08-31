#include "unique_ptr.h"

#include <iostream>

int main()
{
    Uniqlo<int> p1(new int(42));
    Uniqlo<int> p2(new int(42));

    std::cout << std::boolalpha
        << "(p1 == p1)       : " << (p1 == p1) << '\n'
        << "(p1 == p2)       : " << (p1 == p2) << '\n'
        << "(p1 != p2)       : " << (p1 != p2) << '\n'
        << "(p1 < p2)        : " << (p1 < p2) << '\n'
        << "(p1 == nullptr)  : " << (p1 == nullptr) << '\n';
}
