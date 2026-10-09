#include "../v9-op-registry/graph/gradients.h"
#include "tf_matmul.h"
#include <cstdio>
#include <random>
using namespace lf10;
void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
lf::Tensor input(int m, int n, int seed) {
  lf::Tensor t(std::vector<int>{m, n});
  std::mt19937 r(seed);
  std::uniform_real_distribution<float> d(-1, 1);
  for (auto &v : t.data)
    v = d(r);
  return t;
}
double check(const lf::Tensor &a, const lf::Tensor &b) {
  require(a.shape == b.shape, "output shape");
  double error = 0;
  for (size_t i = 0; i < a.data.size(); ++i) {
    double e = std::abs(a.data[i] - b.data[i]);
    require(std::isfinite(a.data[i]) && e <= 2e-4 * (1 + std::abs(b.data[i])),
            "numerical mismatch");
    error = std::max(error, e);
  }
  return error;
}
void scalar_reference(const lf::Tensor &a, const lf::Tensor &b,
                      const lf::Tensor &out) {
  int m = a.shape[0], k = a.shape[1], n = b.shape[1];
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      double sum = 0;
      for (int z = 0; z < k; ++z)
        sum += double(a.data[i * k + z]) * b.data[z * n + j];
      require(std::abs(out.data[i * n + j] - sum) <= 2e-4 * (1 + std::abs(sum)),
              "double reference mismatch");
    }
}
void lifecycle() {
#ifdef LF_LIBXSMM
  require(primitive::registry_size() == 0, "initial registry not empty");
  primitive::MatmulDesc d{
      144, 128, 512, 512, 128, 128, primitive::MatmulDesc::RowMajor};
  auto a = input(144, 512, 1), b = input(512, 128, 2);
  MatmulRuntime ref(Backend::Accelerate, 0);
  lf::Tensor c(std::vector<int>{144, 128});
  {
    primitive::PrimitiveCache cache(1), other(0);
    auto held = cache.get(d);
    held->execute(a.data.data(), b.data.data(), c.data.data());
    check(c, ref.compute(a, b));
    auto changedA = input(144, 512, 3), changedB = input(512, 128, 4);
    held->execute(changedA.data.data(), changedB.data.data(), c.data.data());
    check(c, ref.compute(changedA, changedB));
    scalar_reference(changedA, changedB, c);
    require(cache.get(d) == held && cache.stats.generated == 1,
            "data update recompiled");
    auto d2 = d;
    d2.m = 176;
    cache.get(d2);
    require(cache.stats.evictions == 1, "no eviction");
    held->execute(a.data.data(), b.data.data(), c.data.data());
    check(c, ref.compute(a, b));
    auto equivalent = d;
    equivalent.m = d.n;
    equivalent.n = d.m;
    equivalent.lda = d.ldb;
    equivalent.ldb = d.lda;
    equivalent.ldc = d.ldc;
    equivalent.layout = primitive::MatmulDesc::ColumnMajor;
    auto alias = other.get(equivalent);
    require(other.stats.generated == 0,
            "normalized layout sharing generated twice");
    alias.reset();
    other.reset_stats();
    auto shared = other.get(d);
    require(other.stats.misses == 1 && other.stats.generated == 0,
            "weak owner reuse counted as generation");
    shared.reset();
    held.reset();
    cache.clear();
    require(primitive::registry_size() == 0, "deregistration failed");
    cache.get(d);
    require(cache.stats.generated == 3, "released code not regenerated");
    cache.clear();
  }
  require(primitive::registry_size() == 0, "leaked registry");
#endif
  primitive::PrimitiveCache cache(0);
  auto bad = primitive::MatmulDesc{128, 128, 512, 128, 512, 128};
  bad.beta = 1;
  bool rejected = false;
  try {
    cache.get(bad);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, "unsupported beta accepted");
}
void run(const std::string &mode, Backend backend, int cap, int rounds,
         bool fixed) {
  auto owner = std::make_shared<MatmulRuntime>(backend, cap);
  auto &runtime = *owner;
  install(owner);
  lf::Graph g;
  auto x = g.placeholder("A", {144, 512}), w = g.placeholder("B", {512, 128});
  auto y = matmul2d(g, x, w);
  lf::Session session;
  session.SetWorkers(1);
  std::vector<lf::Tensor> as, bs, refs;
  MatmulRuntime reference(Backend::Accelerate, 0);
  for (int m : fixed
                   ? std::vector<int>{144}
                   : std::vector<int>{128, 144, 160, 176, 192, 208, 224, 240}) {
    as.push_back(input(m, 512, m));
    bs.push_back(input(512, 128, m + 1));
    refs.push_back(reference.compute(as.back(), bs.back()));
  }
  auto request = [&](size_t i) {
    if (mode == "primitive")
      return runtime.compute(as[i], bs[i]);
    return session.run(g, {y}, {{x, as[i]}, {w, bs[i]}})[0];
  };
  double error = 0;
  for (size_t i = 0; i < as.size(); ++i)
    error = std::max(error, check(request(i), refs[i]));
  // Feed different weights at the same shape through the real executor.
  bs[0].data[0] += .125f;
  refs[0] = reference.compute(as[0], bs[0]);
  error = std::max(error, check(request(0), refs[0]));
  // Complete warmup again so capacity experiments have identical starting
  // order.
  for (size_t i = 0; i < as.size(); ++i)
    request(i);
  runtime.reset();
  double total = 0, checksum = 0;
  for (int r = 0; r < rounds; ++r)
    for (size_t i = 0; i < as.size(); ++i) {
      auto start = Clock::now();
      auto out = request(i);
      total += us(start, Clock::now());
      checksum += out.data[0];
    }
  // Full verification outside timing and after statistics snapshot.
  auto stats = runtime.cache.stats;
  if (backend == Backend::Jit) {
    size_t count = rounds * as.size();
    bool fits = cap >= int(as.size());
    require(stats.hits == (fits ? count : 0) &&
                stats.misses == (fits ? 0 : count) &&
                stats.generated == (fits ? 0 : count),
            "cache counts differ from expected true generation");
  }
  double prep = runtime.prepare_us, exec = runtime.execute_us;
  for (size_t i = 0; i < as.size(); ++i)
    error = std::max(error, check(request(i), refs[i]));
  printf(
      "{\"mode\":\"%s\",\"backend\":\"%s\",\"fixed\":%s,\"capacity\":%d,"
      "\"requests\":%zu,\"hits\":%zu,\"misses\":%zu,\"generated\":%zu,"
      "\"evictions\":%zu,\"generated_bytes\":%zu,\"prepare_us\":%.3f,\"execute_"
      "us\":%.3f,\"total_us\":%.3f,\"max_error\":%.9f,\"checksum\":%.9f}\n",
      mode.c_str(),
      backend == Backend::Jit          ? "jit"
      : backend == Backend::Accelerate ? "accelerate"
                                       : "scalar",
      fixed ? "true" : "false", cap, rounds * as.size(), stats.hits,
      stats.misses, stats.generated, stats.evictions, stats.generated_bytes,
      prep, exec, total, error, checksum);
  runtime.cache.clear();
  require(primitive::registry_size() == 0,
          "registry must empty after cache destruction");
}
int main(int argc, char **argv) try {
  if (argc == 2 && std::string(argv[1]) == "test") {
    lifecycle();
    {
      lf::Graph graph;
      auto a = graph.placeholder("a", {128, 512});
      auto b = graph.placeholder("b", {512, 128});
      auto matvec = graph.matmul(a, b);
      auto matrix = graph.matmul2d(a, b);
      lf::cse(graph);
      require(graph.nodes().size() == 4 && matvec->type != matrix->type,
              "CSE confused 2D and matvec semantics");
      bool rejected = false;
      try {
        lf::build_gradients(graph, matrix);
      } catch (const std::runtime_error &) {
        rejected = true;
      }
      require(rejected, "2D gradient was not explicitly rejected");
    }

#ifdef LF_LIBXSMM
    run("tf", Backend::Jit, 8, 2, true);
    for (auto mode : {"primitive", "tf"})
      for (int cap : {0, 2, 8})
        run(mode, Backend::Jit, cap, 2, false);
#endif
    {
      auto owner = std::make_shared<MatmulRuntime>(Backend::Accelerate, 0);
      install(owner);
      lf::matmul_use_sme2 = false;
      lf::Graph g;
      auto a = g.placeholder("a", {2, 3}), b = g.placeholder("b", {3});
      auto y = g.matmul(a, b);
      lf::Session session;
      auto out = session.run(g, {y},
                             {{a, lf::Tensor({1, 2, 3, 4, 5, 6}, {2, 3})},
                              {b, lf::Tensor({1, 2, 3}, {3})}})[0];
      require(out.shape == std::vector<int>{2} &&
                  out.data == std::vector<float>({14, 32}),
              "legacy matvec changed");
    }
    {
      auto owner = std::make_shared<MatmulRuntime>(Backend::Accelerate, 0);
      std::weak_ptr<MatmulRuntime> weak = owner;
      install(owner);
      owner.reset();
      require(!weak.expired(), "registry runtime expired");
      lf::Graph g;
      auto a = g.constant("a", input(128, 512, 5)),
           b = g.constant("b", input(512, 128, 6));
      auto y = g.matmul2d(a, b);
      auto zero = g.placeholder("zero", {128, 128});
      auto fetch = g.add(y, zero);
      lf::Session session;
      auto expected = weak.lock()->compute(a->init, b->init);
      check(session.run(g, {fetch},
                        {{zero, lf::Tensor(std::vector<int>{128, 128})}})[0],
            expected);
      require(y->type == lf::CONST, "constant 2D op not folded");
    }
    run("tf", Backend::Scalar, 0, 1, true);
    run("tf", Backend::Accelerate, 0, 2, true);
    puts("PASS");
  } else if (argc == 6) {
    std::string mode = argv[1], back = argv[2];
    require(mode == "primitive" || mode == "tf",
            "mode must be primitive or tf");
    require(back == "jit" || back == "accelerate" || back == "scalar",
            "unknown backend");
    int cap = std::stoi(argv[3]), rounds = std::stoi(argv[4]);
    require(cap >= 0 && rounds > 0, "invalid capacity/rounds");
    require(std::string(argv[5]) == "fixed" || std::string(argv[5]) == "mixed",
            "invalid workload");
    run(mode,
        back == "jit"          ? Backend::Jit
        : back == "accelerate" ? Backend::Accelerate
                               : Backend::Scalar,
        cap, rounds, std::string(argv[5]) == "fixed");
  } else
    throw std::runtime_error("usage: test | primitive|tf jit|accelerate|scalar "
                             "capacity rounds fixed|mixed");
} catch (const std::exception &e) {
  fprintf(stderr, "ERROR: %s\n", e.what());
  return 1;
}
