CXX = g++
# -mcpu=apple-m4: 开启 SME2/AMX (__ARM_FEATURE_SME2 + __ARM_FEATURE_SME),
# kernels/sme2_gemm.h 的 ZA/mopa 路径; 其他 CPU 自动回退 cblas_sgemm
CXXFLAGS = -std=c++17 -O2 -Wall -mcpu=apple-m4
OBJCXX = clang++
OBJCXXFLAGS = -std=c++17 -ObjC++ -fobjc-arc -O2 -Wall -Ibuild
METAL = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib
FRAMEWORKS = -framework Metal -framework Foundation

TARGET = build/train
OBJS = build/main.o build/metal_matmul.o
HEADERS = framework/tensor.h graph/graph.h graph/gradients.h graph/prune.h \
          graph/optimize.h kernels/kernels.h kernels/sme2_gemm.h \
          kernels/metal_matmul.metal \
          session.h runtime.h core/executor.h core/threadpool.h core/place.h \
          core/metal_matmul.h core/metal_matmul.mm

PYBIND_EXT := $(shell python3 -m pybind11 --extension-suffix)
PYBIND_FLAGS := $(shell python3 -m pybind11 --includes)

.PHONY: all clean run pybind

all: $(TARGET)

build:
	@mkdir -p build

# Metal 编译链: shader (.metal) → air → metallib → 嵌入的 C 数组头
build/metal_matmul.air: kernels/metal_matmul.metal | build
	$(METAL) -c $< -o $@

build/metal_matmul.metallib: build/metal_matmul.air
	$(METALLIB) $< -o $@

build/metallib_data.h: build/metal_matmul.metallib
	xxd -i $< > $@

build/metal_matmul.o: core/metal_matmul.mm core/metal_matmul.h build/metallib_data.h | build
	$(OBJCXX) $(OBJCXXFLAGS) -c $< -o $@

build/main.o: main.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) -c main.cpp -o $@

$(TARGET): $(OBJS) build/metal_matmul.metallib
	$(CXX) -o $@ $(OBJS) $(FRAMEWORKS)

pybind: lfpy$(PYBIND_EXT)

lfpy$(PYBIND_EXT): bindings/lfpy.cpp $(HEADERS) build/metal_matmul.o build/metal_matmul.metallib
	$(CXX) $(CXXFLAGS) -shared -fPIC -I. -undefined dynamic_lookup $(PYBIND_FLAGS) \
		-o $@ build/metal_matmul.o bindings/lfpy.cpp $(FRAMEWORKS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf build lfpy$(PYBIND_EXT)
