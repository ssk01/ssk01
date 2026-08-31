#include "unique_ptr.h"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <utility>

struct Vec3
{
    int x, y, z;

    Vec3(int x = 0, int y = 0, int z = 0) noexcept : x(x), y(y), z(z) {}

    friend std::ostream& operator<<(std::ostream& os, const Vec3& v)
    {
        return os << "{ x=" << v.x << ", y=" << v.y << ", z=" << v.z << " }";
    }
};

template<typename OutputIt>
OutputIt fibonacci(OutputIt first, OutputIt last)
{
    for (int a = 0, b = 1; first != last; ++first)
    {
        *first = b;
        b += std::exchange(a, b);
    }
    return first;
}

int main()
{
    Uniqlo<Vec3> v1 = make_uniqlo<Vec3>();
    Uniqlo<Vec3> v2 = make_uniqlo<Vec3>(0, 1, 2);
    Uniqlo<Vec3[]> v3 = make_uniqlo<Vec3[]>(5);

    Uniqlo<int[]> i1 = make_uniqlo<int[]>(10);
    fibonacci(i1.get(), i1.get() + 10);

    std::cout << "make_uniqlo<Vec3>():      " << *v1 << '\n'
              << "make_uniqlo<Vec3>(0,1,2): " << *v2 << '\n'
              << "make_uniqlo<Vec3[]>(5):   ";
    for (std::size_t i = 0; i < 5; ++i)
        std::cout << std::setw(i ? 30 : 0) << v3[i] << '\n';
    std::cout << '\n';

    std::cout << "make_uniqlo<int[]>(10), fibonacci(...): [" << i1[0];
    for (std::size_t i = 1; i < 10; ++i)
        std::cout << ", " << i1[i];
    std::cout << "]\n";
}
