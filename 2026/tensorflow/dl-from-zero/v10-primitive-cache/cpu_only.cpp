#include "../v9-op-registry/core/metal_matmul.h"
#include <stdexcept>
namespace lf::metal {
bool Available() { return false; }
void MatmulAsync(const float *, const float *, float *, int, int,
                 std::function<void()>) {
  throw std::runtime_error("v10 CPU-only demo has no Metal backend");
}
} // namespace lf::metal
