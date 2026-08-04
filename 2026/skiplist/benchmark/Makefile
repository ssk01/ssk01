CXX ?= c++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

BUILD := build
TARGET := $(BUILD)/skiplist_bench

.PHONY: all run clean

all: $(TARGET)

$(TARGET): skiplist_bench.cpp redis_zskiplist.h
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ skiplist_bench.cpp

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD)
