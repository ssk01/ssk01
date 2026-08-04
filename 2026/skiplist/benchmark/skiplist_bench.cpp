// skiplist_bench.cpp
//
// Benchmark comparing Redis's zskiplist vs std::map vs std::unordered_map.
//
// Workloads modeled after Redis sorted-set operations:
//   insert  (ZADD-like)          - insert N (ele, score) pairs in random order
//   find    (ZSCORE/ZRANK-like)  - locate all N elements
//   update  (ZINCRBY-like)       - change the score of all N elements
//   range   (ZRANGEBYSCORE)      - count elements inside random score ranges
//   delete  (ZREM-like)          - remove all N elements
//
// Note: skiplist lookups are by (score, ele) pair via zslGetRank; the C++ maps
// look up by ele alone. Range queries are indexed by score only in the
// skiplist; the maps must scan (they have no score index) - this is the
// skiplist's raison d'etre.
//
// Usage: ./bench [num_elements] [num_range_queries]

#include "redis_zskiplist.h"
#include "../skiplist.cpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using std::string;
using std::vector;

struct Data {
    vector<string> ele;     // element names, "element:NNNNN"
    vector<double> score;   // scores, uniform in [0, 2*N)
};

static size_t g_n = 200000;
static size_t g_range_queries = 200;

static double now_sec() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static Data gen_data(size_t n, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> score_dist(0.0, (double)n * 2.0);
    Data d;
    d.ele.reserve(n);
    d.score.reserve(n);
    for (size_t i = 0; i < n; i++) {
        d.ele.push_back("element:" + std::to_string(i));
        d.score.push_back(score_dist(rng));
    }
    return d;
}

/* -------------------- correctness self-check -------------------- */

static void self_check(const Data &d) {
    zskiplist *zsl = zslCreate();
    for (size_t i = 0; i < d.ele.size(); i++)
        zslInsert(zsl, d.score[i], d.ele[i]);

    if (zsl->length != d.ele.size()) {
        printf("SELF-CHECK FAIL: length %lu != %zu\n", zsl->length, d.ele.size());
        exit(1);
    }
    for (size_t i = 0; i < d.ele.size(); i++) {
        if (zslGetRank(zsl, d.score[i], d.ele[i]) == 0) {
            printf("SELF-CHECK FAIL: element %s not found\n", d.ele[i].c_str());
            exit(1);
        }
    }
    /* ranks must be strictly increasing w.r.t. sorted score order */
    double prev = -1;
    for (unsigned long r = 1; r <= zsl->length; r++) {
        zskiplistNode *x = zslGetElementByRank(zsl, r);
        if (x->score < prev) {
            printf("SELF-CHECK FAIL: order broken at rank %lu\n", r);
            exit(1);
        }
        prev = x->score;
    }
    /* delete every other element */
    for (size_t i = 0; i < d.ele.size(); i += 2)
        zslDelete(zsl, d.score[i], d.ele[i], NULL);
    if (zsl->length != (d.ele.size() + 1) / 2) {
        printf("SELF-CHECK FAIL: after delete length %lu\n", zsl->length);
        exit(1);
    }
    zslFree(zsl);
    printf("self-check passed\n");
}

/* -------------------- range helpers -------------------- */

static size_t skiplist_count_in_range(zskiplist *zsl, double mn, double mx) {
    zrangespec spec = {mn, mx, 0, 0};
    zskiplistNode *x = zslFirstInRange(zsl, &spec);
    if (!x) return 0;
    size_t cnt = 0;
    while (x && x->score <= mx) {
        cnt++;
        x = x->level[0].forward;
    }
    return cnt;
}

static size_t map_count_in_range(const std::map<string, double> &m, double mn, double mx) {
    size_t cnt = 0;
    for (auto &kv : m)
        if (kv.second >= mn && kv.second <= mx) cnt++;
    return cnt;
}

static size_t umap_count_in_range(const std::unordered_map<string, double> &m, double mn, double mx) {
    size_t cnt = 0;
    for (auto &kv : m)
        if (kv.second >= mn && kv.second <= mx) cnt++;
    return cnt;
}

/* -------------------- workloads -------------------- */

static void bench_skiplist(const Data &d, const vector<size_t> &order) {
    size_t n = d.ele.size();

    double t = now_sec();
    {
        zskiplist *zsl = zslCreate();
        for (size_t i : order)
            zslInsert(zsl, d.score[i], d.ele[i]);
        printf("  skiplist   insert %5zu : %8.3f ms\n", n, (now_sec() - t) * 1e3);
        zslFree(zsl);
    }

    t = now_sec();
    {
        zskiplist *zsl = zslCreate();
        for (size_t i = 0; i < n; i++)
            zslInsert(zsl, d.score[i], d.ele[i]);
        unsigned long hits = 0;
        for (size_t i : order)
            hits += zslGetRank(zsl, d.score[i], d.ele[i]) != 0;
        printf("  skiplist   find   %5zu : %8.3f ms  (hits=%lu)\n", n, (now_sec() - t) * 1e3, hits);
        zslFree(zsl);
    }

    t = now_sec();
    {
        zskiplist *zsl = zslCreate();
        for (size_t i = 0; i < n; i++)
            zslInsert(zsl, d.score[i], d.ele[i]);
        for (size_t i : order)
            zslUpdateScore(zsl, d.score[i], d.ele[i], d.score[i] + 0.5);
        printf("  skiplist   update %5zu : %8.3f ms\n", n, (now_sec() - t) * 1e3);
        zslFree(zsl);
    }

    t = now_sec();
    {
        zskiplist *zsl = zslCreate();
        for (size_t i = 0; i < n; i++)
            zslInsert(zsl, d.score[i], d.ele[i]);
        size_t total = 0;
        for (size_t q = 0; q < g_range_queries; q++) {
            double mn = (double)q / g_range_queries * n * 2.0;
            double mx = mn + n * 0.1;
            total += skiplist_count_in_range(zsl, mn, mx);
        }
        printf("  skiplist   range  %5zu : %8.3f ms  (%zu queries, %zu elems)\n",
               n, (now_sec() - t) * 1e3, g_range_queries, total);
        zslFree(zsl);
    }

    t = now_sec();
    {
        zskiplist *zsl = zslCreate();
        for (size_t i = 0; i < n; i++)
            zslInsert(zsl, d.score[i], d.ele[i]);
        for (size_t i : order)
            zslDelete(zsl, d.score[i], d.ele[i], NULL);
        printf("  skiplist   delete %5zu : %8.3f ms  (remaining=%lu)\n",
               n, (now_sec() - t) * 1e3, zsl->length);
        zslFree(zsl);
    }
}

static void bench_map(const Data &d, const vector<size_t> &order) {
    size_t n = d.ele.size();

    double t = now_sec();
    {
        std::map<string, double> m;
        for (size_t i : order)
            m.emplace(d.ele[i], d.score[i]);
        printf("  map        insert %5zu : %8.3f ms\n", n, (now_sec() - t) * 1e3);
    }

    t = now_sec();
    {
        std::map<string, double> m;
        for (size_t i = 0; i < n; i++)
            m.emplace(d.ele[i], d.score[i]);
        size_t hits = 0;
        for (size_t i : order)
            hits += m.find(d.ele[i]) != m.end();
        printf("  map        find   %5zu : %8.3f ms  (hits=%zu)\n", n, (now_sec() - t) * 1e3, hits);
    }

    t = now_sec();
    {
        std::map<string, double> m;
        for (size_t i = 0; i < n; i++)
            m.emplace(d.ele[i], d.score[i]);
        for (size_t i : order) {
            m.erase(d.ele[i]);
            m.emplace(d.ele[i], d.score[i] + 0.5);
        }
        printf("  map        update %5zu : %8.3f ms\n", n, (now_sec() - t) * 1e3);
    }

    t = now_sec();
    {
        std::map<string, double> m;
        for (size_t i = 0; i < n; i++)
            m.emplace(d.ele[i], d.score[i]);
        size_t total = 0;
        for (size_t q = 0; q < g_range_queries; q++) {
            double mn = (double)q / g_range_queries * n * 2.0;
            double mx = mn + n * 0.1;
            total += map_count_in_range(m, mn, mx);
        }
        printf("  map        range  %5zu : %8.3f ms  (%zu queries, %zu elems, full scan)\n",
               n, (now_sec() - t) * 1e3, g_range_queries, total);
    }

    t = now_sec();
    {
        std::map<string, double> m;
        for (size_t i = 0; i < n; i++)
            m.emplace(d.ele[i], d.score[i]);
        for (size_t i : order)
            m.erase(d.ele[i]);
        printf("  map        delete %5zu : %8.3f ms  (remaining=%zu)\n",
               n, (now_sec() - t) * 1e3, m.size());
    }
}

static void bench_umap(const Data &d, const vector<size_t> &order) {
    size_t n = d.ele.size();

    double t = now_sec();
    {
        std::unordered_map<string, double> m;
        m.reserve(n * 2);
        for (size_t i : order)
            m.emplace(d.ele[i], d.score[i]);
        printf("  unordered_map insert %5zu : %8.3f ms\n", n, (now_sec() - t) * 1e3);
    }

    t = now_sec();
    {
        std::unordered_map<string, double> m;
        m.reserve(n * 2);
        for (size_t i = 0; i < n; i++)
            m.emplace(d.ele[i], d.score[i]);
        size_t hits = 0;
        for (size_t i : order)
            hits += m.find(d.ele[i]) != m.end();
        printf("  unordered_map find   %5zu : %8.3f ms  (hits=%zu)\n", n, (now_sec() - t) * 1e3, hits);
    }

    t = now_sec();
    {
        std::unordered_map<string, double> m;
        m.reserve(n * 2);
        for (size_t i = 0; i < n; i++)
            m.emplace(d.ele[i], d.score[i]);
        for (size_t i : order)
            m[d.ele[i]] = d.score[i] + 0.5;
        printf("  unordered_map update %5zu : %8.3f ms\n", n, (now_sec() - t) * 1e3);
    }

    t = now_sec();
    {
        std::unordered_map<string, double> m;
        m.reserve(n * 2);
        for (size_t i = 0; i < n; i++)
            m.emplace(d.ele[i], d.score[i]);
        size_t total = 0;
        for (size_t q = 0; q < g_range_queries; q++) {
            double mn = (double)q / g_range_queries * n * 2.0;
            double mx = mn + n * 0.1;
            total += umap_count_in_range(m, mn, mx);
        }
        printf("  unordered_map range  %5zu : %8.3f ms  (%zu queries, %zu elems, full scan)\n",
               n, (now_sec() - t) * 1e3, g_range_queries, total);
    }

    t = now_sec();
    {
        std::unordered_map<string, double> m;
        m.reserve(n * 2);
        for (size_t i = 0; i < n; i++)
            m.emplace(d.ele[i], d.score[i]);
        for (size_t i : order)
            m.erase(d.ele[i]);
        printf("  unordered_map delete %5zu : %8.3f ms  (remaining=%zu)\n",
               n, (now_sec() - t) * 1e3, m.size());
    }
}

static void bench_handwritten(const vector<size_t> &order) {
    size_t n = order.size();
    int max_level = 32;

    double t = now_sec();
    {
        Skiplist sl(max_level);
        for (size_t k : order)
            sl.insert((int)k, (int)k);
        printf("  handwritten insert %5zu : %8.3f ms\n", n, (now_sec() - t) * 1e3);
    }

    t = now_sec();
    {
        Skiplist sl(max_level);
        for (size_t k = 0; k < n; k++)
            sl.insert((int)k, (int)k);
        size_t hits = 0;
        for (size_t k : order)
            hits += sl.get((int)k) != -1;
        printf("  handwritten find   %5zu : %8.3f ms  (hits=%zu)\n", n, (now_sec() - t) * 1e3, hits);
    }

    t = now_sec();
    {
        Skiplist sl(max_level);
        for (size_t k = 0; k < n; k++)
            sl.insert((int)k, (int)k);
        for (size_t k : order)
            sl.insert((int)k, (int)k + 1);
        printf("  handwritten update %5zu : %8.3f ms\n", n, (now_sec() - t) * 1e3);
    }
}

static void bench_map_int(const vector<size_t> &order) {
    size_t n = order.size();

    double t = now_sec();
    {
        std::map<int, int> m;
        for (size_t k : order)
            m.emplace((int)k, (int)k);
        printf("  map<int,int>      insert %5zu : %8.3f ms\n", n, (now_sec() - t) * 1e3);
    }

    t = now_sec();
    {
        std::map<int, int> m;
        for (size_t k = 0; k < n; k++)
            m.emplace((int)k, (int)k);
        size_t hits = 0;
        for (size_t k : order)
            hits += m.find((int)k) != m.end();
        printf("  map<int,int>      find   %5zu : %8.3f ms  (hits=%zu)\n", n, (now_sec() - t) * 1e3, hits);
    }

    t = now_sec();
    {
        std::map<int, int> m;
        for (size_t k = 0; k < n; k++)
            m.emplace((int)k, (int)k);
        for (size_t k : order) {
            m.erase((int)k);
            m.emplace((int)k, (int)k + 1);
        }
        printf("  map<int,int>      update %5zu : %8.3f ms\n", n, (now_sec() - t) * 1e3);
    }
}

static void bench_umap_int(const vector<size_t> &order) {
    size_t n = order.size();

    double t = now_sec();
    {
        std::unordered_map<int, int> m;
        m.reserve(n * 2);
        for (size_t k : order)
            m.emplace((int)k, (int)k);
        printf("  unordered_map<int,int> insert %5zu : %8.3f ms\n", n, (now_sec() - t) * 1e3);
    }

    t = now_sec();
    {
        std::unordered_map<int, int> m;
        m.reserve(n * 2);
        for (size_t k = 0; k < n; k++)
            m.emplace((int)k, (int)k);
        size_t hits = 0;
        for (size_t k : order)
            hits += m.find((int)k) != m.end();
        printf("  unordered_map<int,int> find   %5zu : %8.3f ms  (hits=%zu)\n", n, (now_sec() - t) * 1e3, hits);
    }

    t = now_sec();
    {
        std::unordered_map<int, int> m;
        m.reserve(n * 2);
        for (size_t k = 0; k < n; k++)
            m.emplace((int)k, (int)k);
        for (size_t k : order)
            m[(int)k] = (int)k + 1;
        printf("  unordered_map<int,int> update %5zu : %8.3f ms\n", n, (now_sec() - t) * 1e3);
    }
}

int main(int argc, char **argv) {
    if (argc > 1) g_n = (size_t)strtoull(argv[1], NULL, 10);
    if (argc > 2) g_range_queries = (size_t)strtoull(argv[2], NULL, 10);

    srand(0);
    Data d = gen_data(g_n, 12345);

    vector<size_t> order(g_n);
    for (size_t i = 0; i < g_n; i++) order[i] = i;
    std::shuffle(order.begin(), order.end(), std::mt19937_64(42));

    printf("num_elements=%zu  range_queries=%zu\n", g_n, g_range_queries);
    self_check(d);

    printf("\n--- skiplist (redis port) ---\n");
    bench_skiplist(d, order);
    printf("\n--- std::map ---\n");
    bench_map(d, order);
    printf("\n--- std::unordered_map ---\n");
    bench_umap(d, order);
    printf("\n--- skiplist (handwritten, int keys) ---\n");
    bench_handwritten(order);
    printf("\n--- std::map (int keys) ---\n");
    bench_map_int(order);
    printf("\n--- std::unordered_map (int keys) ---\n");
    bench_umap_int(order);
    return 0;
}
