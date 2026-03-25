NVCC ?= nvcc 
TARGET := sol
SRC := sol.cpp knn.cu approx_knn.cu kmeans.cu
CPU_TARGET := cpu_seq
CPU_SRC := cpu_seq.cpp

NVCCFLAGS= -O3 -std=c++11 -x cu -arch=sm_35 -Xcompiler -fopenmp

.PHONY: all cpu clean run-knn benchmark

all: $(TARGET)

cpu: $(CPU_TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(CPU_TARGET): $(CPU_SRC)
	g++ -O2 -std=c++17 $^ -o $@ -fopenmp

run-knn: $(TARGET)
	./$(TARGET) input.txt knn

benchmark: $(TARGET) $(CPU_TARGET)
	python3 benchmark_speedup.py --input input.txt

clean:
	rm -f $(TARGET) $(CPU_TARGET) knn.txt kmeans.txt approx_knn.txt
