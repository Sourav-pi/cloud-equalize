NVCC ?= nvcc 
TARGET := sol
SRC := sol.cpp

NVCCFLAGS := -O2 -std=c++11 -x cu

.PHONY: all clean run-knn

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

run-knn: $(TARGET)
	./$(TARGET) input.txt knn

clean:
	rm -f $(TARGET)
