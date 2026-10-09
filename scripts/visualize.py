#!/usr/bin/env python3
"""Plot before/after point-cloud intensities and histograms with matching display scales."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from mae import compare_rows, read_output
from reference import MODES, Point, equalize, read_input, write_output

ROOT = Path(__file__).resolve().parents[1]


def terrain_cloud():
    points = []
    for row in range(32):
        for column in range(32):
            x, y = (column - 16) * 20, (row - 16) * 20
            wave = math.sin(x / 90) * math.cos(y / 110)
            ridge = math.cos((x + y) / 65)
            z = round(100 * wave + 35 * ridge)
            intensity = round(125 + 12 * wave + 5 * ridge)
            points.append(Point(x, y, z, intensity))
    return points, 64, 20


def save_input(path, points, k, iterations):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{len(points)}\n{k}\n{iterations}\n" + "".join(
        f"{p.x} {p.y} {p.z} {p.intensity}\n" for p in points), encoding="utf-8")


def plot_cloud(points, intensities, mode, backend, k, destination, cache_dir):
    # Keep Matplotlib's cache writable and render without a display server.
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 13, "axes.labelsize": 10})
    figure = plt.figure(figsize=(12, 8), facecolor="white", layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(3, 1.2))
    original = [p.intensity for p in points]
    coords = [[getattr(p, axis) for p in points] for axis in ("x", "y", "z")]
    limits = [(min(values), max(values)) for values in coords]
    spans = [max(1, upper - lower) for lower, upper in limits]
    norm = Normalize(vmin=0, vmax=255)
    title = {"knn": "Exact KNN", "approx_knn": "Morton Approximate KNN", "kmeans": "K-means"}[mode]
    figure.suptitle(f"Point-cloud intensity equalization\n"
                   f"{len(points):,} points | k = {k} | {title} | {backend}",
                   fontsize=16, fontweight="bold")
    axes = []
    for column, (values, label) in enumerate(((original, "Original"), (intensities, "Equalized"))):
        axis = figure.add_subplot(grid[0, column], projection="3d")
        axes.append(axis)
        scatter = axis.scatter(*coords, c=values, cmap="gray", norm=norm,
                               s=14, alpha=1, depthshade=False, edgecolors="none")
        axis.view_init(elev=38, azim=-55)
        axis.set_box_aspect(spans)
        for dimension, bounds in zip(("x", "y", "z"), limits):
            lower, upper = bounds
            padding = max(1, (upper - lower) * 0.03)
            getattr(axis, f"set_{dimension}lim")(lower - padding, upper + padding)
            getattr(axis, f"set_{dimension}label")(dimension.upper(), labelpad=5)
        axis.tick_params(labelsize=8)
        axis.set_title(f"{label}\nIntensity range: {min(values)}-{max(values)}", pad=10)

    colorbar = figure.colorbar(scatter, ax=axes, orientation="horizontal", shrink=0.6, pad=0.02)
    colorbar.set_label("Intensity (shared grayscale; no automatic rescaling)")
    colorbar.set_ticks((0, 64, 128, 192, 255))
    histogram = None
    bins = list(range(0, 257, 16))
    for column, (values, color) in enumerate(((original, "#158579"), (intensities, "#db5a68"))):
        axis = figure.add_subplot(grid[1, column], sharey=histogram)
        if histogram is None:
            histogram = axis
        axis.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.7)
        axis.set_xlim(0, 255)
        axis.set_xticks((0, 64, 128, 192, 255))
        axis.set_xlabel("Intensity")
        axis.set_ylabel("Point count")
        axis.set_title("Intensity distribution", fontsize=11)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Omit to generate the deterministic 1,024-point terrain demo")
    parser.add_argument("--equalized", type=Path, help="Existing CPU/CUDA output; omit to compute with Python")
    parser.add_argument("--mode", choices=MODES, default="knn")
    parser.add_argument("--label", help="Source label for a supplied output, e.g. 'CUDA output'")
    parser.add_argument("--max-points", type=int, default=2000, help="Python computation size guard")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/demo/equalization.png")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "outputs/demo")
    args = parser.parse_args()
    if args.equalized and not args.input:
        parser.error("--equalized requires --input so point order can be verified")
    if args.label and not args.equalized:
        parser.error("--label requires an existing --equalized file")
    if args.input:
        _, k, iterations, points = read_input(args.input)
        input_path = args.input
    else:
        points, k, iterations = terrain_cloud()
        input_path = args.data_dir / "input.txt"
        save_input(input_path, points, k, iterations)
    original_rows = [(p.x, p.y, p.z, p.intensity) for p in points]
    if args.equalized:
        rows = read_output(args.equalized)
        compare_rows(original_rows, rows)
        intensities = [row[3] for row in rows]
        output_path = args.equalized
        backend = args.label or "Supplied output"
    else:
        if len(points) > args.max_points:
            parser.error("Cloud exceeds --max-points; supply --equalized output or explicitly increase the guard")
        intensities = equalize(points, k, iterations, args.mode)
        output_path = args.data_dir / f"{args.mode}.txt"
        write_output(output_path, points, intensities)
        backend = "Python reference"
    plot_cloud(points, intensities, args.mode, backend, k, args.output, args.data_dir / ".matplotlib")
    metadata = {"points": len(points), "k": k, "iterations": iterations, "mode": args.mode,
                "source": backend, "input": str(input_path), "equalized": str(output_path),
                "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                "original_intensity_range": [min(p.intensity for p in points), max(p.intensity for p in points)],
                "equalized_intensity_range": [min(intensities), max(intensities)],
                "figure": str(args.output)}
    args.data_dir.mkdir(parents=True, exist_ok=True)
    (args.data_dir / "visualization.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Figure: {args.output}\nInput: {input_path}\nEqualized: {output_path}")


if __name__ == "__main__":
    main()
