cmake_minimum_required(VERSION 3.16)
project(rtti VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_library(rtti STATIC
    src/type_registry.cpp
)

target_include_directories(rtti PUBLIC include)

# ---- Tests ----

add_executable(rtti_test
    test/rtti_test.cpp
)
target_link_libraries(rtti_test PRIVATE rtti)

# ---- Benchmark ----

add_executable(rtti_benchmark
    src/benchmark.cpp
)
target_link_libraries(rtti_benchmark PRIVATE rtti)
target_compile_options(rtti_benchmark PRIVATE -O2)
