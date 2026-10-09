#pragma once
#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>
#ifdef LF_LIBXSMM
#include <libxsmm.h>
#endif
namespace primitive {
// Only these fixed semantics are supported. Leading dimensions are explicit.
struct MatmulDesc {
  int m, n, k, lda, ldb, ldc;
  enum Layout { ColumnMajor, RowMajor } layout = ColumnMajor;
  int dtype = 32;
  bool transpose_a = false, transpose_b = false;
  float alpha = 1, beta = 0;
  auto key() const {
    return std::make_tuple(m, n, k, lda, ldb, ldc, int(layout), dtype,
                           transpose_a, transpose_b, alpha, beta);
  }
  bool operator<(const MatmulDesc &d) const { return key() < d.key(); }
  bool operator==(const MatmulDesc &d) const { return key() == d.key(); }
  void validate() const {
    if (m <= 0 || n <= 0 || k <= 0 || dtype != 32 || transpose_a ||
        transpose_b || alpha != 1 || beta != 0 ||
        (layout != ColumnMajor && layout != RowMajor) ||
        lda < (layout == ColumnMajor ? m : k) ||
        ldb < (layout == ColumnMajor ? k : n) ||
        ldc < (layout == ColumnMajor ? m : n))
      throw std::invalid_argument("requires positive FP32 GEMM, no transpose, "
                                  "alpha=1 beta=0, valid strides/layout");
  }
};
inline size_t registry_size() {
#ifdef LF_LIBXSMM
  libxsmm_registry_info i{};
  if (libxsmm_get_registry_info(&i))
    throw std::runtime_error("registry query failed");
  return i.size;
#else
  return 0;
#endif
}
struct Backend {
  Backend() = default;
  Backend(const Backend &) = delete;
  Backend &operator=(const Backend &) = delete;
#ifdef LF_LIBXSMM
  libxsmm_gemmfunction fn = nullptr;
#endif
  size_t bytes = 0;
  ~Backend() {
#ifdef LF_LIBXSMM
    if (fn)
      libxsmm_release_kernel(reinterpret_cast<const void *>(fn));
#endif
  }
};
struct CompileResult {
  std::shared_ptr<Backend> owner;
  bool generated;
};
// Weak pool coordinates ownership across caches and callers; it retains no
// code. Entire API is single-request-thread only. Do not concurrently
// dispatch/release.
inline CompileResult compile_backend(const MatmulDesc &d) {
  d.validate();
#ifdef LF_LIBXSMM
  libxsmm_init();
  static std::map<MatmulDesc, std::weak_ptr<Backend>> pool;
  for (auto i = pool.begin(); i != pool.end();)
    if (i->second.expired())
      i = pool.erase(i);
    else
      ++i;
  auto found = pool.find(d);
  if (found != pool.end())
    if (auto p = found->second.lock())
      return {p, false};
  // Normalize row-major C=A*B to column-major C^T=B^T*A^T.
  const bool row = d.layout == MatmulDesc::RowMajor;
  auto shape = libxsmm_create_gemm_shape(
      row ? d.n : d.m, row ? d.m : d.n, d.k, row ? d.ldb : d.lda,
      row ? d.lda : d.ldb, d.ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
      LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
  auto before = registry_size();
  auto fn = libxsmm_dispatch_gemm(shape, LIBXSMM_GEMM_FLAG_BETA_0,
                                  LIBXSMM_GEMM_PREFETCH_NONE);
  if (!fn)
    throw std::runtime_error("LIBXSMM JIT unavailable");
  // A normalized descriptor may coincide with another public description.
  // Share that owner by pointer; never release someone else's code.
  for (const auto &e : pool)
    if (auto p = e.second.lock())
      if (p->fn == fn) {
        pool[d] = p;
        return {p, false};
      }
  if (registry_size() != before + 1)
    throw std::runtime_error(
        "unowned backend registry hit; require isolated dereg-enabled LIBXSMM");
  auto p = std::make_shared<Backend>();
  p->fn = fn;
  libxsmm_kernel_info info{};
  if (libxsmm_get_kernel_info(reinterpret_cast<const void *>(fn), &info) ||
      info.is_reference_kernel || !info.code_size)
    throw std::runtime_error("backend is not generated code");
  p->bytes = info.code_size;
  pool[d] = p;
  return {p, true};
#else
  throw std::runtime_error("LIBXSMM support not built; JIT cannot fallback");
#endif
}
class Primitive {
  friend class PrimitiveCache;
  MatmulDesc desc_;
  std::shared_ptr<Backend> owner_;

  Primitive(MatmulDesc d, std::shared_ptr<Backend> owner)
      : desc_(d), owner_(std::move(owner)) {
    desc_.validate();
    if (!owner_ || !owner_->bytes)
      throw std::invalid_argument("null primitive backend");
  }

public:
  const MatmulDesc &desc() const { return desc_; }
  size_t code_bytes() const { return owner_->bytes; }
  void execute(const float *a, const float *b, float *c) const {
    if (!a || !b || !c)
      throw std::invalid_argument("null GEMM buffer");
#ifdef LF_LIBXSMM
    libxsmm_gemm_param p{};
    bool row = desc_.layout == MatmulDesc::RowMajor;
    p.a.primary = const_cast<float *>(row ? b : a);
    p.b.primary = const_cast<float *>(row ? a : b);
    p.c.primary = c;
    owner_->fn(&p);
#else
    throw std::runtime_error("JIT not built");
#endif
  }
};
struct Stats {
  size_t hits = 0, misses = 0, generated = 0, evictions = 0,
         generated_bytes = 0;
};
class PrimitiveCache {
  size_t capacity_;
  std::list<std::pair<MatmulDesc, std::shared_ptr<Primitive>>> lru_;

public:
  Stats stats;
  explicit PrimitiveCache(size_t capacity) : capacity_(capacity) {}
  void clear() { lru_.clear(); }
  void reset_stats() { stats = {}; }
  size_t size() const { return lru_.size(); }
  std::shared_ptr<Primitive> get(const MatmulDesc &d) {
    d.validate();
    auto i = std::find_if(lru_.begin(), lru_.end(),
                          [&](const auto &e) { return e.first == d; });
    if (i != lru_.end()) {
      ++stats.hits;
      auto p = i->second;
      lru_.splice(lru_.begin(), lru_, i);
      return p;
    }
    ++stats.misses;
    auto result = compile_backend(d);
    if (result.generated) {
      ++stats.generated;
      stats.generated_bytes += result.owner->bytes;
    }
    auto p =
        std::shared_ptr<Primitive>(new Primitive(d, std::move(result.owner)));
    if (capacity_) {
      lru_.emplace_front(d, p);
      if (lru_.size() > capacity_) {
        lru_.pop_back();
        ++stats.evictions;
      }
    }
    return p;
  }
};
} // namespace primitive
