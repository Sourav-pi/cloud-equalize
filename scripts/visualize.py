#!/usr/bin/env python3
"""Run all CUDA equalization modes and plot their outputs with matching display scales."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from mae import compare_rows, read_output
from reference import MODES, Point, read_input

ROOT = Path(__file__).resolve().parents[1]


def terrain_cloud(grid_size=32, k=64, iterations=20):
    points = []
    for row in range(grid_size):
        for column in range(grid_size):
            x, y = (column - grid_size // 2) * 20, (row - grid_size // 2) * 20
            wave = math.sin(x / 90) * math.cos(y / 110)
            ridge = math.cos((x + y) / 65)
            z = round(100 * wave + 35 * ridge)
            intensity = round(125 + 12 * wave + 5 * ridge)
            points.append(Point(x, y, z, intensity))
    return points, k, iterations


def save_input(path, points, k, iterations):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{len(points)}\n{k}\n{iterations}\n" + "".join(
        f"{p.x} {p.y} {p.z} {p.intensity}\n" for p in points), encoding="utf-8")


def run_cuda(binary, input_path, points, data_dir, timeout):
    original = [(p.x, p.y, p.z, p.intensity) for p in points]
    outputs, runs = {}, {}
    data_dir.mkdir(parents=True, exist_ok=True)
    # Fresh paths prevent old outputs from being mistaken for successful executions.
    with tempfile.TemporaryDirectory(prefix="cuda-", dir=data_dir) as temporary:
        for mode in MODES:
            output = Path(temporary) / f"{mode}.txt"
            command = [str(binary.resolve()), str(input_path.resolve()), mode, str(output.resolve())]
            start = time.perf_counter()
            try:
                process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f"CUDA {mode} failed (exit {error.returncode}): "
                                   f"{error.stderr.strip() or error.stdout.strip()}") from error
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"CUDA {mode} exceeded {timeout} seconds") from error
            elapsed = time.perf_counter() - start
            rows = read_output(output)
            compare_rows(original, rows)
            outputs[mode] = rows
            runs[mode] = {"command": command, "process_seconds": elapsed,
                          "stdout": process.stdout, "stderr": process.stderr,
                          "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                          "output_path": str(data_dir / f"{mode}.txt")}
        for mode in MODES:
            shutil.copyfile(Path(temporary) / f"{mode}.txt", data_dir / f"{mode}.txt")
    return outputs, runs


def plot_cloud(points, outputs, k, iterations, destination, cache_dir, plot_points):
    # Keep Matplotlib's cache writable and render without a display server.
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 13, "axes.labelsize": 10})
    figure = plt.figure(figsize=(18, 8), facecolor="white", layout="constrained")
    grid = figure.add_gridspec(2, 4, height_ratios=(3, 1.2))
    original = [p.intensity for p in points]
    all_coords = [[getattr(p, axis) for p in points] for axis in ("x", "y", "z")]
    sample_size = min(len(points), plot_points)
    indices = [i * len(points) // sample_size for i in range(sample_size)]
    coords = [[values[i] for i in indices] for values in all_coords]
    limits = [(min(values), max(values)) for values in all_coords]
    spans = [max(1, upper - lower) for lower, upper in limits]
    norm = Normalize(vmin=0, vmax=255)
    figure.suptitle(f"Point-cloud intensity equalization\n"
                   f"{len(points):,} points | k = {k} | T = {iterations} | CUDA outputs | "
                   f"{sample_size:,} points shown per view",
                   fontsize=16, fontweight="bold")
    views = [(original, "Original", "#158579")]
    labels = {"knn": "Exact KNN", "approx_knn": "Approximate KNN", "kmeans": "K-means"}
    colors = {"knn": "#db5a68", "approx_knn": "#4278b5", "kmeans": "#785496"}
    views.extend(([row[3] for row in outputs[mode]], labels[mode], colors[mode]) for mode in MODES)
    axes = []
    for column, (values, label, _) in enumerate(views):
        axis = figure.add_subplot(grid[0, column], projection="3d")
        axes.append(axis)
        scatter = axis.scatter(*coords, c=[values[i] for i in indices], cmap="gray", norm=norm,
                               s=14, alpha=1, depthshade=False, edgecolors="none")
        axis.view_init(elev=38, azim=-55)
        axis.set_box_aspect(spans)
        for dimension, bounds in zip(("x", "y", "z"), limits):
            lower, upper = bounds
            padding = max(1, (upper - lower) * 0.03)
            getattr(axis, f"set_{dimension}lim")(lower - padding, upper + padding)
            getattr(axis, f"set_{dimension}label")(dimension.upper(), labelpad=5)
        axis.tick_params(labelsize=8)
        axis.set_title(f"{label}\nIntensity range: {min(values)}-{max(values)}", pad=10, fontsize=12)

    colorbar = figure.colorbar(scatter, ax=axes, orientation="horizontal", shrink=0.6, pad=0.02)
    colorbar.set_label("Intensity (shared grayscale; no automatic rescaling)")
    colorbar.set_ticks((0, 64, 128, 192, 255))
    histogram = None
    bins = list(range(0, 257, 16))
    for column, (values, _, color) in enumerate(views):
        axis = figure.add_subplot(grid[1, column], sharey=histogram)
        if histogram is None:
            histogram = axis
        axis.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.7)
        axis.set_xlim(0, 255)
        axis.set_xticks((0, 64, 128, 192, 255))
        axis.set_xlabel("Intensity")
        axis.set_ylabel("Point count")
        axis.set_title("Intensity distribution (all points)", fontsize=10)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Omit to generate a deterministic terrain cloud")
    parser.add_argument("--binary", type=Path, default=ROOT / "build/pointcloud_eq", help="CUDA executable")
    parser.add_argument("--grid-size", type=int, default=32, help="Generated terrain width/height (2-316)")
    parser.add_argument("--k", type=int, default=64, help="Generated terrain neighbor/cluster count")
    parser.add_argument("--iterations", type=int, default=20, help="Generated terrain K-means iterations")
    parser.add_argument("--timeout", type=float, default=600, help="Timeout in seconds per CUDA process")
    parser.add_argument("--plot-points", type=int, default=5000, help="Maximum points shown per 3D view")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cuda-comparison/comparison.png")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "outputs/cuda-comparison")
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error(f"CUDA executable not found: {args.binary}. Build it with make gpu on a CUDA host.")
    if args.timeout <= 0 or args.plot_points < 1 or args.output.suffix.lower() != ".png":
        parser.error("Require --timeout>0, --plot-points>=1, and a .png output path")
    destinations = [args.data_dir / f"{mode}.txt" for mode in MODES]
    destinations.append(args.data_dir / "visualization.json")
    if args.output.resolve() in [p.resolve() for p in destinations]:
        parser.error("Figure path must differ from data and metadata paths")
    if args.input:
        _, k, iterations, points = read_input(args.input)
        input_path = args.input
    else:
        if not (2 <= args.grid_size <= 316 and 1 <= args.k <= min(128, args.grid_size ** 2 - 1)
                and 0 <= args.iterations <= 50):
            parser.error("Invalid generated cloud parameters: grid-size=2-316, k<=min(128,n-1), T=0-50")
        points, k, iterations = terrain_cloud(args.grid_size, args.k, args.iterations)
        input_path = args.data_dir / "input.txt"
        save_input(input_path, points, k, iterations)
    if k >= len(points):
        parser.error("All-mode CUDA comparison requires k<n")
    if input_path.resolve() in [p.resolve() for p in destinations + [args.output]]:
        parser.error("Input must differ from every output path")
    outputs, runs = run_cuda(args.binary, input_path, points, args.data_dir, args.timeout)
    comparison = {"approx_knn_vs_exact_knn": compare_rows(outputs["knn"], outputs["approx_knn"]),
                  "kmeans_vs_exact_knn": compare_rows(outputs["knn"], outputs["kmeans"]),
                  "kmeans_vs_approx_knn": compare_rows(outputs["approx_knn"], outputs["kmeans"])}
    plot_cloud(points, outputs, k, iterations, args.output, args.data_dir / ".matplotlib", args.plot_points)
    metadata = {"points": len(points), "k": k, "iterations": iterations,
                "source": "CUDA executable", "binary": str(args.binary.resolve()),
                "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                "input": str(input_path),
                "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "original_intensity_range": [min(p.intensity for p in points), max(p.intensity for p in points)],
                "runs": runs, "comparisons": comparison, "figure": str(args.output),
                "notes": ["K-means defines a different neighborhood; its differences are not correctness failures.",
                          "Each timing is a single process wall time, not a repeated benchmark.",
                          "Histograms and comparison metrics use all points; 3D views may be subsampled."]}
    args.data_dir.mkdir(parents=True, exist_ok=True)
    (args.data_dir / "visualization.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Figure: {args.output}\nInput: {input_path}\nCUDA outputs and report: {args.data_dir}")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
