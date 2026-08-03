CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

TARGET = build/train
SRC = main.cpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC) framework/tensor.h graph/graph.h kernels/kernels.h session.h
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
