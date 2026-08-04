// skiplist_test.cpp - simple unit tests for the hand-written Skiplist.
//
// skiplist.cpp is NOT modified here; this TU provides the missing standard
// headers (<vector>/<tuple>/<cstdlib>) before including it.

#include <vector>
#include <tuple>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <map>

#include "skiplist.cpp"

static int failures = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);         \
            failures++;                                                    \
        }                                                                  \
    } while (0)

static void test_single_insert_get() {
    Skiplist sl(10);
    sl.insert(5, 50);
    CHECK(sl.get(5) == 50);
    CHECK(sl.get(6) == -1);
    CHECK(sl.get(0) == -1);
}

static void test_insert_update_existing() {
    Skiplist sl(10);
    sl.insert(7, 100);
    sl.insert(7, 200);
    CHECK(sl.get(7) == 200);
}

static void test_multiple_keys() {
    Skiplist sl(10);
    for (int k = 0; k < 100; k++)
        sl.insert(k, k * 10);
    for (int k = 0; k < 100; k++)
        CHECK(sl.get(k) == k * 10);
    CHECK(sl.get(100) == -1);
    CHECK(sl.get(-1) == -1);
}

static void test_random_keys_against_map() {
    const int N = 5000;
    Skiplist sl(16);
    std::map<int, int> ref;
    for (int i = 0; i < N; i++) {
        int key = rand() % 100000;
        int val = rand();
        sl.insert(key, val);
        ref[key] = val;
    }
    for (int i = 0; i < N; i++) {
        int key = rand() % 100000;
        int got = sl.get(key);
        auto it = ref.find(key);
        CHECK((got == -1) == (it == ref.end()));
        if (it != ref.end()) CHECK(got == it->second);
    }
    for (auto &kv : ref) CHECK(sl.get(kv.first) == kv.second);
}

static void test_boundary_keys() {
    Skiplist sl(10);
    sl.insert(INT_MIN, 1);
    sl.insert(INT_MAX, 2);
    CHECK(sl.get(INT_MIN) == 1);
    CHECK(sl.get(INT_MAX) == 2);
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    srand(1234);
    printf("case 1: single insert/get ...\n");
    test_single_insert_get();
    printf("case 1 passed\n");

    printf("case 2: insert then update existing key ...\n");
    test_insert_update_existing();
    printf("case 2 passed\n");

    printf("case 3: multiple keys 0..99 ...\n");
    test_multiple_keys();
    printf("case 3 passed\n");

    printf("case 4: random 5000 keys vs std::map ...\n");
    test_random_keys_against_map();
    printf("case 4 passed\n");

    printf("case 5: boundary INT_MIN/INT_MAX ...\n");
    test_boundary_keys();
    printf("case 5 passed\n");

    if (failures == 0) {
        printf("ALL TESTS PASSED\n");
        return 0;
    }
    printf("%d FAILURES\n", failures);
    return 1;
}
