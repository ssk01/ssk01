CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

TARGET = build/train
SRC = main.cpp
PYBIND_EXT := $(shell python3 -m pybind11 --extension-suffix)
PYBIND_FLAGS := $(shell python3 -m pybind11 --includes)

.PHONY: all clean run pybind

all: $(TARGET)

$(TARGET): $(SRC) framework/tensor.h graph/graph.h graph/gradients.h graph/prune.h graph/optimize.h graph/quantize.h kernels/kernels.h kernels/quantize.h session.h runtime.h
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

pybind: lfpy$(PYBIND_EXT)

lfpy$(PYBIND_EXT): bindings/lfpy.cpp framework/tensor.h graph/graph.h graph/gradients.h graph/prune.h graph/optimize.h graph/quantize.h kernels/kernels.h kernels/quantize.h session.h runtime.h
	$(CXX) $(CXXFLAGS) -shared -fPIC -I. -undefined dynamic_lookup $(PYBIND_FLAGS) -o $@ bindings/lfpy.cpp

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) lfpy$(PYBIND_EXT)
