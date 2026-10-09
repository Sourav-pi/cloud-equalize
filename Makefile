CXX ?= g++
NVCC ?= nvcc
CUDA_ARCH ?= native
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra
NVCCFLAGS ?= -O3 -std=c++17 -Xcompiler -fopenmp
CPPFLAGS += -Iinclude
GPU := build/pointcloud_eq
CPU := build/cpu_reference
GPU_SRC := src/main.cpp src/knn.cu src/approx_knn.cu src/kmeans.cu
HEADERS := $(wildcard include/*.hpp include/*.cuh)
INPUT ?= examples/sample.txt
RUNS ?= 5
WARMUP ?= 1

.PHONY: all gpu cpu clean validate validate-cpu test benchmark
all: gpu cpu
gpu: $(GPU)
cpu: $(CPU)

build:
	mkdir -p $@

$(GPU): $(GPU_SRC) $(HEADERS) Makefile scripts/record_build.py | build
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -arch=$(CUDA_ARCH) $(GPU_SRC) -o $@
	python3 scripts/record_build.py --binary $@ -- $(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -arch=$(CUDA_ARCH) $(GPU_SRC) -o $@

$(CPU): src/cpu_reference.cpp $(HEADERS) Makefile scripts/record_build.py | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@
	python3 scripts/record_build.py --binary $@ -- $(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@

validate: all
	python3 scripts/validate.py --input "$(INPUT)" --backend both

validate-cpu: cpu
	python3 scripts/validate.py --input "$(INPUT)" --backend cpu

test: cpu
	python3 -m unittest discover -s tests -v

ifeq ($(POINTCLOUD_TEST_GPU),1)
test: gpu
endif

benchmark: all
	python3 scripts/benchmark.py --input "$(INPUT)" --runs $(RUNS) --warmup $(WARMUP)

clean:
	rm -rf build
