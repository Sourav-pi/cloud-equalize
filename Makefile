NVCC ?= nvcc 
TARGET := sol
SRC := sol.cpp knn.cu approx_knn.cu kmeans.cu
CPU_TARGET := cpu_seq
CPU_SRC := cpu_seq.cpp

NVCCFLAGS= -O3 -std=c++11 -x cu -arch=sm_35 -Xcompiler -fopenmp --use_fast_math

.PHONY: all cpu clean run-knn benchmark

all: $(TARGET)

cpu: $(CPU_TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(CPU_TARGET): $(CPU_SRC)
	g++ -std=c++17 $^ -o $@ -fopenmp

run-knn: $(TARGET)
	./$(TARGET) input.txt knn

benchmark: $(TARGET) $(CPU_TARGET)
	python3 benchmark_speedup.py --input input.txt

clean:
	rm -f $(TARGET) $(CPU_TARGET) knn.txt kmeans.txt approx_knn.txt cpu_seq cpu_*.txt *_ans.txt v

cmp_knn : $(TARGET) $(CPU_TARGET)
	@./$(TARGET) input.txt knn
	@./$(CPU_TARGET) input.txt knn
	@echo "diff : "
	@diff knn.txt cpu_knn.txt

cmp_kmeans : $(TARGET) $(CPU_TARGET)
	@./$(TARGET) input.txt kmeans
	@./$(CPU_TARGET) input.txt kmeans
	@echo "diff : "
	@diff kmeans.txt cpu_kmeans.txt

cmp_aknn : $(TARGET) $(CPU_TARGET)
	@./$(TARGET) input.txt approx_knn
	@./$(CPU_TARGET) input.txt approx_knn
	@echo "diff : "
	@diff approx_knn.txt cpu_approx_knn.txt

v : v.cu
	$(NVCC) $(NVCCFLAGS) v.cu -o v
