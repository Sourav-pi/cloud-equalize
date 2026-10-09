# Benchmark Evidence

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

Generated reports live in ignored `results/`. To publish a reviewed result, place its JSON in `docs/benchmarks/` with a short note identifying the hardware, dataset distribution/seed, source revision, validation results, timing scope, and whether the machine was otherwise idle. Avoid reporting tiny fixture timings as scalability evidence. There are currently no reviewed GPU benchmark results for the reorganized project.

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
