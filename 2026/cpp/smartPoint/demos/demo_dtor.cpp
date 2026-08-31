#include "unique_ptr.h"

#include <iostream>

int main()
{
    auto deleter = [](int* ptr)
    {
        std::cout << "[deleter called]\n";
        delete ptr;
    };

    Uniqlo<int, decltype(deleter)> uniq(new int, deleter);
    std::cout << (uniq ? "not empty\n" : "empty\n");
    uniq.reset();
    std::cout << (uniq ? "not empty\n" : "empty\n");
}
