#!/usr/bin/env python3
"""Readable reference for the CUDA implementation's integer arithmetic and neighborhoods."""
import argparse
from dataclasses import dataclass
from pathlib import Path

MODES = ("knn", "approx_knn", "kmeans")


@dataclass(frozen=True)
class Point:
    x: int
    y: int
    z: int
    intensity: int


def read_input(path):
    tokens = Path(path).read_text(encoding="utf-8").split()
    if len(tokens) < 3:
        raise ValueError("Missing n, k, T header")
    n, k, iterations = map(int, tokens[:3])
    if not (1 <= n <= 100000 and 1 <= k <= min(128, n) and 0 <= iterations <= 50):
        raise ValueError("Invalid n, k, T bounds")
    if len(tokens) != 3 + 4 * n:
        raise ValueError("Expected exactly n records of x y z intensity")
    points = [Point(*map(int, tokens[i:i + 4])) for i in range(3, len(tokens), 4)]
    if any(not all(-10000 <= c <= 10000 for c in (p.x, p.y, p.z)) or
           not 0 <= p.intensity <= 255 for p in points):
        raise ValueError("Coordinates must be in [-10000,10000], intensity in [0,255]")
    if len({(p.x, p.y, p.z) for p in points}) != n:
        raise ValueError("Duplicate coordinates are unsupported")
    return n, k, iterations, points


def trunc_div(num, den):
    # C++ truncates negative centroid averages toward zero.
    return (1 if num >= 0 else -1) * (abs(num) // den)


def remap(old, values, size):
    minimum = min(values)
    cmin = values.count(minimum)
    if size == cmin:
        return old
    cdf = sum(value <= old for value in values)
    return max(0, cdf - cmin) * 255 // (size - cmin)


def neighbor_key(points, i, j):
    a, b = points[i], points[j]
    distance = (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2
    return distance, b.x, b.y, b.z


def morton(point):
    coords = (point.x + 10000, point.y + 10000, point.z + 10000)
    return sum(((coordinate >> bit) & 1) << (3 * bit + axis)
               for bit in range(21) for axis, coordinate in enumerate(coords))


def build_neighbors(points, k, approximate=False):
    n = len(points)
    if not 1 <= k <= min(128, n - 1):
        raise ValueError("Neighbors require 1<=k<=min(128,n-1)")
    order = sorted(range(n), key=lambda i: (morton(points[i]), i)) if approximate else list(range(n))
    neighbors = [[] for _ in points]
    for position, i in enumerate(order):
        if approximate:
            candidates = morton_candidates(order, position)
        else:
            candidates = order
        neighbors[i] = sorted((j for j in candidates if j != i),
                              key=lambda j: neighbor_key(points, i, j))[:k]
    return neighbors


def morton_candidates(order, position):
    center = (position // 128) * 128 + 64
    start = max(0, center - 7500)
    end = min(len(order) - 1, start + 15000)
    return order[start:end + 1]


def neighbor_equalized(points, k, approximate=False):
    neighbors = build_neighbors(points, k, approximate)
    return [remap(p.intensity, [p.intensity] + [points[j].intensity for j in neighbors[i]], k + 1)
            for i, p in enumerate(points)]


def kmeans_equalized(points, k, iterations):
    if not 1 <= k <= min(128, len(points)):
        raise ValueError("Invalid cluster count")
    centroids = [(p.x, p.y, p.z) for p in points[:k]]
    assignments = []
    for _ in range(max(1, iterations)):
        assignments = [min(range(k), key=lambda c: (
            sum((a - b) ** 2 for a, b in zip((p.x, p.y, p.z), centroids[c])),
            *centroids[c], c)) for p in points]
        if iterations == 0:
            break
        updated = []
        for c in range(k):
            members = [p for p, assignment in zip(points, assignments) if assignment == c]
            updated.append(tuple(trunc_div(sum(getattr(p, axis) for p in members), len(members))
                                 for axis in ("x", "y", "z")) if members else centroids[c])
        if updated == centroids:
            break
        centroids = updated
    values = [[] for _ in range(k)]
    for point, cluster in zip(points, assignments):
        values[cluster].append(point.intensity)
    return [remap(p.intensity, values[c], len(values[c])) for p, c in zip(points, assignments)]


def equalize(points, k, iterations, mode):
    if mode == "kmeans":
        return kmeans_equalized(points, k, iterations)
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    return neighbor_equalized(points, k, mode == "approx_knn")


def write_output(path, points, intensities):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{p.x} {p.y} {p.z} {value}\n"
                            for p, value in zip(points, intensities)), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--mode", choices=MODES, default="knn")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/reference"))
    args = parser.parse_args()
    _, k, iterations, points = read_input(args.input)
    write_output(args.out_dir / f"{args.mode}.txt", points,
                 equalize(points, k, iterations, args.mode))
