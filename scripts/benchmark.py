#!/usr/bin/env python3
"""Record repeatable process wall times, provenance, and output quality as JSON."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile
import time

from mae import compare_rows, read_output
from reference import MODES, read_input

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "unavailable": str(error)}


def benchmark(args):
    if args.runs < 1 or args.warmup < 0 or args.timeout <= 0:
        raise ValueError("Require runs>=1, warmup>=0, timeout>0")
    input_path = args.input.resolve()
    n, k, iterations, points = read_input(input_path)
    if k >= n:
        raise ValueError("Benchmark runs all neighbor modes and requires k<n")
    backends = ("cpu", "gpu") if args.backend == "both" else (args.backend,)
    binaries = {b: ROOT / "build" / ("cpu_reference" if b == "cpu" else "pointcloud_eq") for b in backends}
    for binary in binaries.values():
        if not binary.is_file():
            raise FileNotFoundError(f"Build executable first: {binary}")
    report = {
        "schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"path": str(input_path), "sha256": digest(input_path),
                    "n": n, "k": k, "iterations": iterations},
        "protocol": {"runs": args.runs, "warmup": args.warmup, "timeout_seconds": args.timeout,
                     "timing": "perf_counter around each subprocess: startup, input validation and I/O, "
                               "allocations, preprocessing, computation, transfers, output I/O, teardown",
                     "warmup_scope": "discarded fresh processes; no persistent CUDA context",
                     "cpu_approx_mode": "excluded: exact-KNN fallback, not a Morton baseline"},
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                 "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "runtime default"),
                 "cpu_info": probe(["sysctl", "-n", "machdep.cpu.brand_string"]) if platform.system() == "Darwin"
                             else probe(["lscpu"]),
                 "gpu": probe(["nvidia-smi"]), "nvcc": probe(["nvcc", "--version"]),
                 "cxx": probe([args.cxx, "--version"])},
        "source": {"commit": probe(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
                   "worktree": probe(["git", "-C", str(ROOT), "status", "--short"]),
                   "file_sha256": {str(p.relative_to(ROOT)): digest(p) for directory in
                                   ("src", "include", "scripts") for p in sorted((ROOT / directory).glob("*"))
                                   if p.is_file()}, "makefile_sha256": digest(ROOT / "Makefile")},
        "executables": {b: {"path": str(p), "sha256": digest(p)} for b, p in binaries.items()},
        "measurements": {}, "quality": {},
    }
    for backend, binary in binaries.items():
        receipt_path = binary.with_suffix(".build.json")
        if not receipt_path.is_file():
            raise ValueError(f"Missing build receipt; rebuild through make: {binary}")
        receipt = json.loads(receipt_path.read_text())
        if receipt["binary_sha256"] != digest(binary):
            raise ValueError(f"Executable differs from build receipt: {binary}")
        if any(digest(ROOT / path) != value for path, value in receipt["source_sha256"].items()):
            raise ValueError(f"Sources changed since build; run make: {binary}")
        report["executables"][backend]["build"] = receipt
    final_rows = {}
    with tempfile.TemporaryDirectory(prefix="pointcloud-bench-") as temp:
        for backend in backends:
            for mode in (("knn", "kmeans") if backend == "cpu" else MODES):
                key = f"{backend}/{mode}"
                output = Path(temp) / f"{backend}_{mode}.txt"
                samples, warmups = [], []
                expected_hash = None
                for index in range(args.warmup + args.runs):
                    command = [str(binaries[backend]), str(input_path), mode, str(output)]
                    start = time.perf_counter()
                    process = subprocess.run(command, capture_output=True, text=True,
                                             check=True, timeout=args.timeout)
                    elapsed = time.perf_counter() - start
                    rows = read_output(output)
                    if len(rows) != n or any(row[:3] != (p.x, p.y, p.z) for row, p in zip(rows, points)):
                        raise ValueError(f"Invalid point order or count from {key}")
                    output_hash = digest(output)
                    if expected_hash is not None and output_hash != expected_hash:
                        raise ValueError(f"Nondeterministic output from {key}")
                    expected_hash = output_hash
                    trial = {"seconds": elapsed, "command": command, "stdout": process.stdout,
                             "stderr": process.stderr, "output_sha256": output_hash}
                    (warmups if index < args.warmup else samples).append(trial)
                times = [trial["seconds"] for trial in samples]
                report["measurements"][key] = {"median_seconds": statistics.median(times),
                    "min_seconds": min(times), "max_seconds": max(times),
                    "warmups": warmups, "trials": samples}
                final_rows[key] = rows
                print(f"{key}: median {statistics.median(times):.6f} s ({args.runs} runs)")
    if args.backend == "both":
        for mode in ("knn", "kmeans"):
            quality = compare_rows(final_rows[f"cpu/{mode}"], final_rows[f"gpu/{mode}"])
            if quality["different_points"]:
                raise AssertionError(f"CPU/GPU {mode} mismatch: {quality}")
            report["quality"][f"cpu_gpu_{mode}"] = quality
            report["measurements"][f"gpu/{mode}"]["speedup_vs_sequential_cpu"] = (
                report["measurements"][f"cpu/{mode}"]["median_seconds"] /
                report["measurements"][f"gpu/{mode}"]["median_seconds"])
    if "gpu" in backends:
        report["quality"]["morton_vs_exact_gpu"] = compare_rows(final_rows["gpu/knn"], final_rows["gpu/approx_knn"])
        report["measurements"]["gpu/approx_knn"]["speedup_vs_exact_gpu"] = (
            report["measurements"]["gpu/knn"]["median_seconds"] /
            report["measurements"]["gpu/approx_knn"]["median_seconds"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "examples/sample.txt")
    parser.add_argument("--backend", choices=("cpu", "gpu", "both"), default="both")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--cxx", default="g++", help="Compiler version probe; build provenance remains separate")
    parser.add_argument("--output", type=Path, default=ROOT / "results/benchmark.json")
    benchmark(parser.parse_args())
