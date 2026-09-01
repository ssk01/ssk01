// 来源: cppreference std::tuple 主页 Example（get_student）
// 适配: std::tuple->mytup::Tuple, std::get->mytup::get, std::tie->mytup::tie
// 预期: 挂 —— ① return {..} 拷贝列表初始化需要"条件 explicit"值构造（我们全 explicit）;
//       ② mytup::get<T> 按类型取未实现; ③ mytup::tie 未实现; 结构化绑定部分可过
#include "tuple.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

using namespace std;
using namespace mytup;

mytup::Tuple<double, char, string> get_student(int id)
{
    switch (id)
    {
        case 0: return {3.8, 'A', "Lisa Simpson"};
        case 1: return {2.9, 'C', "Milhouse Van Houten"};
        case 2: return {1.7, 'D', "Ralph Wiggum"};
        case 3: return {0.6, 'F', "Bart Simpson"};
    }
    throw std::invalid_argument("id");
}

int main()
{
    const auto student0 = get_student(0);
    std::cout << "ID: 0, "
              << "GPA: " << mytup::get<0>(student0) << ", "
              << "grade: " << mytup::get<1>(student0) << ", "
              << "name: " << mytup::get<2>(student0) << '\n';

    const auto student1 = get_student(1);
    std::cout << "ID: 1, "
              << "GPA: " << mytup::get<double>(student1) << ", "
              << "grade: " << mytup::get<char>(student1) << ", "
              << "name: " << mytup::get<std::string>(student1) << '\n';

    double gpa2;
    char grade2;
    std::string name2;
    mytup::tie(gpa2, grade2, name2) = get_student(2);
    std::cout << "ID: 2, "
              << "GPA: " << gpa2 << ", "
              << "grade: " << grade2 << ", "
              << "name: " << name2 << '\n';

    const auto [gpa3, grade3, name3] = get_student(3);
    std::cout << "ID: 3, "
              << "GPA: " << gpa3 << ", "
              << "grade: " << grade3 << ", "
              << "name: " << name3 << '\n';
}
