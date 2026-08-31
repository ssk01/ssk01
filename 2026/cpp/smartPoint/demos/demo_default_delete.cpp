#include "unique_ptr.h"

#include <algorithm>
#include <vector>

int main()
{
    {
        Uniqlo<int> ptr(new int(5));
    } // Uniqlo<int> uses DefaultDeleter<int>

    {
        Uniqlo<int[]> ptr(new int[10]);
    } // Uniqlo<int[]> uses DefaultDeleter<int[]>

    // DefaultDeleter can be used anywhere a delete functor is needed
    std::vector<int*> v;
    for (int n = 0; n < 100; ++n)
        v.push_back(new int(n));
    std::for_each(v.begin(), v.end(), DefaultDeleter<int>());
}
