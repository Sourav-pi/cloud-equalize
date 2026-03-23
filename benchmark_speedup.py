#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List


def run_cmd(cmd: List[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print("Command failed:", " ".join(cmd), file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError("Command execution failed")


def timed_run(cmd: List[str], cwd: Path) -> float:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    end = time.perf_counter()
    if proc.returncode != 0:
        print("Timed command failed:", " ".join(cmd), file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError("Timed command failed")
    return end - start


def read_intensities(path: Path) -> List[int]:
    vals: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 4:
                continue
            vals.append(int(parts[3]))
    return vals


def mae(a: List[int], b: List[int]) -> float:
    if len(a) != len(b):
        raise ValueError("Length mismatch while computing MAE")
    if not a:
        return 0.0
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def benchmark_model(
    cwd: Path,
    input_file: str,
    model: str,
    runs: int,
    warmup: int,
    verify: bool,
) -> Dict[str, float]:
    cpu_out = f"cpu_{model}.txt"
    gpu_out = f"{model}.txt"

    for _ in range(warmup):
        run_cmd(["./cpu_seq", input_file, model, cpu_out], cwd)
        run_cmd(["./sol", input_file, model], cwd)

    cpu_times = [timed_run(["./cpu_seq", input_file, model, cpu_out], cwd) for _ in range(runs)]
    gpu_times = [timed_run(["./sol", input_file, model], cwd) for _ in range(runs)]

    cpu_avg = mean(cpu_times)
    gpu_avg = mean(gpu_times)
    speedup = cpu_avg / gpu_avg if gpu_avg > 0 else float("inf")

    result: Dict[str, float] = {
        "cpu_avg_s": cpu_avg,
        "gpu_avg_s": gpu_avg,
        "speedup": speedup,
    }

    if verify:
        cpu_i = read_intensities(cwd / cpu_out)
        gpu_i = read_intensities(cwd / gpu_out)
        result["mae"] = mae(cpu_i, gpu_i)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CUDA solution speedup against sequential C++ baseline")
    parser.add_argument("--input", default="input.txt", help="Input file")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["knn", "approx_knn", "kmeans"],
        choices=["knn", "approx_knn", "kmeans"],
        help="Models to benchmark",
    )
    parser.add_argument("--runs", type=int, default=5, help="Measured runs per model")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per model")
    parser.add_argument("--no-build", action="store_true", help="Skip rebuilding binaries")
    parser.add_argument("--no-verify", action="store_true", help="Skip MAE check between CPU and CUDA outputs")
    args = parser.parse_args()

    cwd = Path(__file__).resolve().parent

    if not args.no_build:
        run_cmd(["g++", "-O2", "-std=c++17", "cpu_seq.cpp", "-o", "cpu_seq"], cwd)
        run_cmd(["make", "all"], cwd)

    input_path = cwd / args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("Benchmark configuration")
    print(f"  input   : {args.input}")
    print(f"  models  : {' '.join(args.models)}")
    print(f"  runs    : {args.runs}")
    print(f"  warmup  : {args.warmup}")
    print(f"  verify  : {not args.no_verify}")
    print()

    header = f"{'model':<12} {'cpu_avg(s)':>12} {'gpu_avg(s)':>12} {'speedup':>10} {'MAE':>10}"
    print(header)
    print("-" * len(header))

    for model in args.models:
        stats = benchmark_model(
            cwd=cwd,
            input_file=args.input,
            model=model,
            runs=args.runs,
            warmup=args.warmup,
            verify=not args.no_verify,
        )

        mae_val = stats.get("mae", float("nan"))
        mae_text = f"{mae_val:10.4f}" if not args.no_verify else f"{'-':>10}"
        print(
            f"{model:<12} {stats['cpu_avg_s']:12.6f} {stats['gpu_avg_s']:12.6f} "
            f"{stats['speedup']:10.2f} {mae_text}"
        )


if __name__ == "__main__":
    main()
