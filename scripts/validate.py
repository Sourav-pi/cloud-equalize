#!/usr/bin/env python3
"""Validate small clouds against the Python reference; report approximation quality separately."""
import argparse
from pathlib import Path
import subprocess
import tempfile

from mae import compare_rows, read_output
from reference import MODES, equalize, read_input

ROOT = Path(__file__).resolve().parents[1]


def validate(input_path, backend="both", max_points=2000, mode="all"):
    n, k, iterations, points = read_input(input_path)
    if n > max_points:
        raise ValueError(f"Reference validation is expensive: n={n} exceeds --max-points={max_points}")
    modes = MODES if mode == "all" else (mode,)
    expected = {m: [(p.x, p.y, p.z, value) for p, value in
                    zip(points, equalize(points, k, iterations, m))] for m in modes}
    with tempfile.TemporaryDirectory(prefix="pointcloud-validate-") as temp:
        for target in (("cpu", "gpu") if backend == "both" else (backend,)):
            binary = ROOT / "build" / ("cpu_reference" if target == "cpu" else "pointcloud_eq")
            if not binary.is_file():
                raise FileNotFoundError(f"Build {binary} first with make {target}")
            for m in modes:
                output = Path(temp) / f"{target}_{m}.txt"
                subprocess.run([str(binary), str(Path(input_path).resolve()), m, str(output)],
                               check=True, capture_output=True, text=True, timeout=600)
                # The CPU approximate mode is an exact-KNN compatibility fallback.
                oracle = expected["knn"] if target == "cpu" and m == "approx_knn" and "knn" in expected else expected[m]
                if target == "cpu" and m == "approx_knn" and "knn" not in expected:
                    oracle = [(p.x, p.y, p.z, value) for p, value in
                              zip(points, equalize(points, k, iterations, "knn"))]
                result = compare_rows(oracle, read_output(output))
                if result["different_points"]:
                    raise AssertionError(f"{target}/{m} differs from its reference: {result}")
                label = " (exact fallback)" if target == "cpu" and m == "approx_knn" else ""
                print(f"PASS {target}/{m}{label}: {n} records")
    if "approx_knn" in expected and "knn" in expected:
        print("Morton approximation versus exact KNN:", compare_rows(expected["knn"], expected["approx_knn"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "examples/sample.txt")
    parser.add_argument("--backend", choices=("cpu", "gpu", "both"), default="both")
    parser.add_argument("--mode", choices=("all",) + MODES, default="all")
    parser.add_argument("--max-points", type=int, default=2000)
    args = parser.parse_args()
    validate(args.input, args.backend, args.max_points, args.mode)
