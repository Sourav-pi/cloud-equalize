# Benchmark Evidence

## Jetson AGX Orin

[jetson-agx-orin.json](jetson-agx-orin.json) records a CUDA/CPU comparison run on October 10, 2026. The report is preserved unchanged from the Jetson execution.

- Hardware: NVIDIA Jetson AGX Orin GPU and ARM Cortex-A78AE CPU; 12 logical CPU cores were reported, while the CPU baseline executes sequentially.
- Toolchain: CUDA 12.6.68, GCC 11.4.0; GPU target `sm_87`; `-O3` and C++17 for both builds, with OpenMP enabled for CUDA host preprocessing.
- Source revision: `cb4fa646dd18518be29ad7a4c445bd3e5675025b`; the recorded source worktree was clean.
- Dataset: 20,000 unique uniformly sampled integer coordinates in `[-10000,10000]` and random 8-bit intensities; `k = 64`, `T = 20`, seed 42.
- Input SHA-256: `bd954d9e9874cffb67c009e9e8072ff1ca47739351c53fdfd9057597901b0a54`. Regeneration with seed 42 reproduces this hash.
- Protocol: five measured runs and one discarded warmup per mode, each in a fresh process; OpenMP thread count used the runtime default.

| Mode | CPU Median | GPU Median | Time Ratio |
| --- | ---: | ---: | --- |
| Exact KNN | 26.341734 s | 0.168498 s | CPU/GPU: 156.33x |
| Approximate KNN | - | 0.164639 s | Exact GPU/approximate GPU: 1.023x |
| K-means | 0.083752 s | 0.125978 s | GPU/CPU: 1.50x (CPU faster) |

Exact KNN and K-means output intensities matched the sequential CPU baseline for all 20,000 points. Morton approximate KNN differed from exact GPU KNN at 2,319 points, with intensity MAE 0.90905 and maximum absolute error 44. All measured and warmup output hashes were deterministic within each mode. The brought-back visualization outputs match the benchmark GPU output hashes.

These are process wall times, including startup, validation, file I/O, preprocessing, allocations, transfers, computation, and teardown. The exact KNN CPU baseline sorts all candidate distances. K-means uses a different neighborhood and is compared with CPU K-means, not with KNN for correctness.

Reproduce from the repository root on the Jetson:

```sh
export PATH="/usr/local/cuda/bin:$PATH"
make clean
make all CUDA_ARCH=sm_87 CXX=g++ NVCC=nvcc
make validate CUDA_ARCH=sm_87
python3 scripts/generate_dataset.py --n 20000 --k 64 --iterations 20 --seed 42 --output outputs/orin-input.txt
python3 scripts/benchmark.py --input outputs/orin-input.txt --backend both --runs 5 --warmup 1 --timeout 3600 --cxx g++ --output results/jetson-agx-orin.json
```

The raw report retains execution paths from the Jetson, including temporary output paths; they describe the original run. New executions use paths on the current machine. Record the selected power mode and clock settings alongside future reports for comparisons across configurations.

## Historical Runs

`legacy/stdout.log` and `legacy/stderr.log` preserve the original scheduler outputs byte-for-byte. They identify a Tesla K40m, driver 470.82.01, CUDA toolkit 11.0, GCC 9.1.0, and a build targeting `sm_35`. The log contains three GPU wall-time lines: 5.73007, 4.25865, and 2.37861 seconds.

Those times are **unattributed historical observations**, not comparative results. The logs do not identify the model associated with each line, the input hash, a source revision, repeated trials, warmup policy, output accuracy, or a CPU timing. The logged compiler command also differs from the original checked-in Makefile flags. The original scheduler job number is 800980; no speedup can be derived from these files.

The original Linux ELF executable was removed after verifying it was a generated binary. The original input is retained as `examples/legacy-input.txt`; its relationship to the logged executions is unverified. The assignment PDF is retained under `docs/assignment.pdf`.

## Reproduce A Measurement

Run from the repository root on a CUDA host:

```sh
make clean
make all CUDA_ARCH=sm_80
make validate
POINTCLOUD_TEST_GPU=1 make test
python3 scripts/generate_dataset.py --n 1000 --k 10 --iterations 20 --seed 42 --output outputs/benchmark-input.txt
python3 scripts/validate.py --input outputs/benchmark-input.txt --backend both
OMP_NUM_THREADS=8 python3 scripts/benchmark.py --input outputs/benchmark-input.txt --runs 5 --warmup 1 --output results/benchmark.json
```

Select the architecture appropriate to your hardware. For larger datasets, validate smaller fixtures first and increase `--timeout` if needed. The CPU's exhaustive sorting can dominate runtime. A GPU-only report (`--backend gpu`) still measures approximation MAE and its ratio against GPU exact KNN, but does not validate exact GPU output against a CPU oracle or report a CPU speedup.

The JSON report records raw measured and discarded warmup runs, hardware/toolchain probes, exact build receipts, dataset parameters/hashes, source hashes, and binary hashes. The benchmark refuses mismatched build receipts or stale source files. Version probes describe the current environment; the build receipt describes the compiler actually used to build each executable.

Each sample launches a fresh process; warmups may warm filesystem caches but do not preserve GPU allocations or a CUDA context. Process wall times include startup, parsing and uniqueness checks, algorithm execution, synchronous transfers, output serialization, and teardown. Program-printed timers are retained in raw stdout as supporting observations; speedup calculations use the external process timer consistently for both backends. No kernel-only performance claim is supported by this workflow.

Repeated outputs must be deterministic and structurally valid. With both backends, exact KNN and K-means CPU/GPU output equality is required before publishing their time ratios. Morton error is reported as intensity MAE, maximum absolute error, and number of changed points against GPU exact KNN. Approximation speedup and error should always be presented together. K-means is a different neighborhood model; it is not required to reproduce KNN intensities.

Generated reports live in ignored `results/`. Published evidence lives in `docs/benchmarks/`, including the Jetson AGX Orin report above. To publish another reviewed result, place its JSON here with a short note identifying the hardware, dataset distribution/seed, source revision, validation results, timing scope, and whether the machine was otherwise idle. Avoid reporting tiny fixture timings as scalability evidence.

## Cluster Execution

The wrapper uses your current scheduler allocation and environment; it does not submit a job or choose a GPU resource for you. Configure site-specific modules only if needed:

```sh
COMPILER_MODULE=compiler/gcc/9.1.0 \
CUDA_MODULE=suite/nvidia-hpc-sdk/20.7/cuda11.0 \
CUDA_ARCH=sm_35 \
INPUT_FILE=examples/sample.txt RUNS=5 WARMUP=1 \
bash scripts/run_cluster.sh
```

These module names come from the historical environment and are examples, not required defaults. The shell must provide the site's `module` function when module variables are set. The wrapper validates the selected input against Python before benchmarking, so keep its input small enough for reference validation. For large timing inputs, run validation separately and invoke `scripts/benchmark.py` directly inside the allocation. Rebuild after changing modules or compiler flags.
