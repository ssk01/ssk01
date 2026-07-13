OMP_DIR := third_party/libomp
CXX := clang++
CXXFLAGS := -std=c++17 -O3 -mcpu=apple-m4 -ffast-math -funroll-loops -Wall -DACCELERATE_NEW_LAPACK
OMPFLAGS := -Xpreprocessor -fopenmp -I$(OMP_DIR)/include
LDFLAGS := -L$(OMP_DIR)/lib -lomp -Wl,-rpath,@executable_path/$(OMP_DIR)/lib -framework Accelerate

gemm: gemm.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS)

run: gemm
	./gemm

clean:
	rm -f gemm

.PHONY: run clean
