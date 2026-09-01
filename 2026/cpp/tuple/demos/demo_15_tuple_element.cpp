// 来源: cppreference tuple_element<std::tuple> Example
// 适配: std::tuple->mytup::Tuple; boost::typeindex 打印换成 static_assert + typeid（避免外部依赖）
// 预期: 过 —— std::tuple_element 对 mytup::Tuple 的特化，含引用/volatile 元素
#include "tuple.h"

#include <cstddef>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>

using namespace std;
using namespace mytup;

struct MyStruct {};

using MyTuple = mytup::Tuple<int, long&, const char&, bool&&,
                             std::string, volatile MyStruct>;

static_assert(std::is_same_v<std::tuple_element_t<0, MyTuple>, int>);
static_assert(std::is_same_v<std::tuple_element_t<1, MyTuple>, long&>);
static_assert(std::is_same_v<std::tuple_element_t<2, MyTuple>, const char&>);
static_assert(std::is_same_v<std::tuple_element_t<3, MyTuple>, bool&&>);
static_assert(std::is_same_v<std::tuple_element_t<4, MyTuple>, std::string>);
static_assert(std::is_same_v<std::tuple_element_t<5, MyTuple>, volatile MyStruct>);

template <typename TupleLike, std::size_t I = 0>
void printTypes()
{
    if constexpr (I == 0)
        std::cout << typeid(TupleLike).name() << '\n';

    if constexpr (I < std::tuple_size_v<TupleLike>)
    {
        using SelectedType = std::tuple_element_t<I, TupleLike>;
        std::cout << "  The type at index " << I << " is: "
                  << typeid(SelectedType).name() << '\n';
        printTypes<TupleLike, I + 1>();
    }
}

int main()
{
    printTypes<MyTuple>();
}
