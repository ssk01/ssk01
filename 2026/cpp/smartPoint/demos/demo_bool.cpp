#include "unique_ptr.h"

#include <iostream>

int main()
{
    Uniqlo<int> ptr(new int(42));

    if (ptr)
        std::cout << "before reset, ptr is: " << *ptr << '\n';
    ptr.reset();
    (ptr ? (std::cout << "after reset, ptr is: " << *ptr)
        : (std::cout << "after reset ptr is empty")) << '\n';
}
