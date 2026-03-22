NVCC ?= nvcc 
TARGET := sol
SRC := sol.cpp knn.cu approx_knn.cu kmeans.cu

NVCCFLAGS := -O2 -std=c++11 -x cu

.PHONY: all clean run-knn

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

run-knn: $(TARGET)
	./$(TARGET) input.txt knn

clean:
	rm -f $(TARGET)
