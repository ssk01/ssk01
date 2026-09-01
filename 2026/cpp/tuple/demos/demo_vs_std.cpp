// 对照 std::tuple：同类型下 sizeof 应当一致，结构化绑定同样工作。
// 注意：本文件 include <tuple>，与我们的全局 apply 无冲突（这里不调用 apply）。
#include "tuple.h"

#include <iostream>
#include <string>
#include <tuple>
#include <utility>

using namespace std;
using namespace mytup;

struct Empty1 {};
struct Empty2 {};

int main() {
  cout << "sizeof(std::tuple<Empty1, Empty2, int>) = "
       << sizeof(std::tuple<Empty1, Empty2, int>) << '\n';
  cout << "sizeof(    ::Tuple<Empty1, Empty2, int>) = "
       << sizeof(Tuple<Empty1, Empty2, int>) << '\n';

  Tuple<int, string, double> mine(1, "mine", 2.5);
  std::tuple<int, string, double> stdt(1, "std", 2.5);

  cout << "mine tuple_size = " << tuple_size<decltype(mine)>::value
       << ", std tuple_size = " << tuple_size<decltype(stdt)>::value << '\n';

  auto [mi, ms, md] = mine;
  auto [si, ss, sd] = stdt;
  cout << mi << ' ' << ms << ' ' << md << '\n';
  cout << si << ' ' << ss << ' ' << sd << '\n';
}
