CXX = clang++
ARCH ?= arm64
ifeq ($(shell uname -s),Darwin)
TARGET = -arch $(ARCH)
endif
BASE = -std=c++17 -Wall -Wextra -Wpedantic -ffile-prefix-map=$(CURDIR)=. -fdebug-prefix-map=$(CURDIR)=.
OPT = -O3
SCALAR = -fno-vectorize -fno-slp-vectorize
.PHONY: all test sanitize ubsan assembly clean
all: build/demo build/demo-auto
build:
	mkdir -p build
build/demo: demo.cpp match.h | build
	$(CXX) $(TARGET) $(BASE) $(OPT) $(SCALAR) demo.cpp -o $@
build/demo-auto: demo.cpp match.h | build
	$(CXX) $(TARGET) $(BASE) $(OPT) demo.cpp -o $@
build/test: test.cpp match.h | build
	$(CXX) $(TARGET) $(BASE) $(OPT) $(SCALAR) test.cpp -o $@
test: build/test
	./build/test
sanitize: | build
	$(CXX) $(TARGET) $(BASE) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer test.cpp -o build/test-sanitize
	./build/test-sanitize
ubsan: | build
	$(CXX) $(TARGET) $(BASE) -O1 -g -fsanitize=undefined test.cpp -o build/test-ubsan
	./build/test-ubsan
assembly: | build
	$(CXX) $(TARGET) $(BASE) $(OPT) $(SCALAR) -S instruction_probe.cpp -o build/instruction_probe.s
clean:
	rm -rf build
