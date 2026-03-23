#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

L = 256


@dataclass
class Point:
    x: int
    y: int
    z: int
    intensity: int


def read_input(path: str) -> Tuple[int, int, int, List[Point]]:
    with open(path, "r", encoding="utf-8") as f:
        tokens = f.read().strip().split()

    if len(tokens) < 3:
        raise ValueError("Input must contain at least n, k, T")

    n = int(tokens[0])
    k = int(tokens[1])
    t = int(tokens[2])

    expected = 3 + 4 * n
    if len(tokens) != expected:
        raise ValueError(f"Expected {expected} tokens, got {len(tokens)}")

    points: List[Point] = []
    idx = 3
    for _ in range(n):
        x = int(tokens[idx])
        y = int(tokens[idx + 1])
        z = int(tokens[idx + 2])
        intensity = int(tokens[idx + 3])
        if not (0 <= intensity <= 255):
            raise ValueError("Intensity must be in [0, 255]")
        points.append(Point(x, y, z, intensity))
        idx += 4

    return n, k, t, points


def sq_dist(a: Point, b: Point) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return dx * dx + dy * dy + dz * dz


def cdf_and_cmin(hist: Sequence[int]) -> Tuple[List[int], int]:
    cdf = [0] * L
    running = 0
    cmin = None
    for v in range(L):
        running += hist[v]
        cdf[v] = running
        if cdf[v] > 0 and cmin is None:
            cmin = cdf[v]
    if cmin is None:
        cmin = 0
    return cdf, cmin


def cxx_trunc_div(num: int, den: int) -> int:
    if den == 0:
        raise ZeroDivisionError("division by zero")
    # Match C/C++ integer division semantics (truncate toward zero).
    return int(num / den)


def remap_intensity(old_i: int, cdf: Sequence[int], cmin: int, m: int) -> int:
    if m == cmin:
        return old_i
    num = cdf[old_i] - cmin
    if num < 0:
        num = 0
    return cxx_trunc_div(num * (L - 1), (m - cmin))


def build_knn_neighbors(points: Sequence[Point], k: int) -> List[List[int]]:
    n = len(points)
    if k > n - 1:
        raise ValueError("For KNN, k must be <= n - 1")

    neighbors: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        d: List[Tuple[float, int]] = []
        pi = points[i]
        for j in range(n):
            if i == j:
                continue
            d.append((sq_dist(pi, points[j]), j))
        d.sort(key=lambda x: (x[0], x[1]))
        neighbors[i] = [idx for _, idx in d[:k]]
        print(neighbors[i])
    return neighbors


def knn_equalized(points: Sequence[Point], k: int) -> List[int]:
    neighbors = build_knn_neighbors(points, k)
    out = [0] * len(points)

    for i in range(len(points)):
        hist = [0] * L
        for j in neighbors[i]:
            hist[points[j].intensity] += 1
        cdf, cmin = cdf_and_cmin(hist)
        out[i] = remap_intensity(points[i].intensity, cdf, cmin, k+1)
    return out


def cell_of(p: Point, cell_size: float) -> Tuple[int, int, int]:
    return (
        int(math.floor(p.x / cell_size)),
        int(math.floor(p.y / cell_size)),
        int(math.floor(p.z / cell_size)),
    )


def approx_knn_neighbors(points: Sequence[Point], k: int) -> List[List[int]]:
    n = len(points)
    if k > n - 1:
        raise ValueError("For Approx-KNN, k must be <= n - 1")

    # Simple deterministic approximation:
    # 1) bucket points in a uniform 3D grid,
    # 2) collect candidates from nearby cells,
    # 3) if candidates are insufficient, fall back to global scan.
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]

    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    span_z = max(zs) - min(zs)
    span = max(span_x, span_y, span_z)

    target_per_cell = max(8, int(math.sqrt(max(1, n))))
    if span == 0:
        cell_size = 1.0
    else:
        # Heuristic cube size so cells are not too sparse/dense.
        cell_size = max(1e-6, span / max(1.0, (n / target_per_cell) ** (1.0 / 3.0)))

    grid: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    point_cells: List[Tuple[int, int, int]] = []
    for i, p in enumerate(points):
        c = cell_of(p, cell_size)
        point_cells.append(c)
        grid[c].append(i)

    neighbors: List[List[int]] = [[] for _ in range(n)]

    for i, p in enumerate(points):
        cx, cy, cz = point_cells[i]
        candidates: List[int] = []
        seen = set()

        # Expand the cube of neighboring cells until enough candidates are found.
        radius = 0
        max_radius = 6
        while len(candidates) < max(4 * k, k + 1) and radius <= max_radius:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        cell = (cx + dx, cy + dy, cz + dz)
                        if cell not in grid:
                            continue
                        for j in grid[cell]:
                            if j == i or j in seen:
                                continue
                            seen.add(j)
                            candidates.append(j)
            radius += 1

        # Fallback keeps correctness close to exact KNN when local density is low.
        if len(candidates) < k:
            candidates = [j for j in range(n) if j != i]

        d = [(sq_dist(p, points[j]), j) for j in candidates]
        d.sort(key=lambda x: (x[0], x[1]))
        neighbors[i] = [idx for _, idx in d[:k]]

    return neighbors


def approx_knn_equalized(points: Sequence[Point], k: int) -> List[int]:
    neighbors = approx_knn_neighbors(points, k)
    out = [0] * len(points)

    for i in range(len(points)):
        hist = [0] * L
        for j in neighbors[i]:
            hist[points[j].intensity] += 1
        cdf, cmin = cdf_and_cmin(hist)
        out[i] = remap_intensity(points[i].intensity, cdf, cmin, k+1)
    return out


def assign_points(points: Sequence[Point], centroids: Sequence[Tuple[float, float, float]]) -> List[int]:
    assignments = [0] * len(points)
    for i, p in enumerate(points):
        best_c = 0
        best_d = float("inf")
        for c_idx, (cx, cy, cz) in enumerate(centroids):
            dx = p.x - cx
            dy = p.y - cy
            dz = p.z - cz
            d = dx * dx + dy * dy + dz * dz
            if d < best_d or (d == best_d and c_idx < best_c):
                best_d = d
                best_c = c_idx
        assignments[i] = best_c
    return assignments


def recompute_centroids(
    points: Sequence[Point], assignments: Sequence[int], k: int, old_centroids: Sequence[Tuple[int, int, int]]
) -> List[Tuple[int, int, int]]:
    sum_x = [0] * k
    sum_y = [0] * k
    sum_z = [0] * k
    cnt = [0] * k

    for i, p in enumerate(points):
        c = assignments[i]
        sum_x[c] += p.x
        sum_y[c] += p.y
        sum_z[c] += p.z
        cnt[c] += 1

    new_centroids: List[Tuple[int, int, int]] = []
    for c in range(k):
        if cnt[c] == 0:
            new_centroids.append(old_centroids[c])
        else:
            new_centroids.append(
                (
                    cxx_trunc_div(sum_x[c], cnt[c]),
                    cxx_trunc_div(sum_y[c], cnt[c]),
                    cxx_trunc_div(sum_z[c], cnt[c]),
                )
            )
    return new_centroids


def kmeans_assignments(points: Sequence[Point], k: int, t: int) -> List[int]:
    n = len(points)
    if k < 1 or k > n:
        raise ValueError("For K-means, k must satisfy 1 <= k <= n")

    centroids = [(points[i].x, points[i].y, points[i].z) for i in range(k)]
    assignments = [-1] * n

    for _ in range(t):
        new_assignments = assign_points(points, centroids)
        if new_assignments == assignments:
            assignments = new_assignments
            break
        assignments = new_assignments
        centroids = recompute_centroids(points, assignments, k, centroids)

    if assignments[0] == -1:
        assignments = assign_points(points, centroids)

    return assignments


def kmeans_equalized(points: Sequence[Point], k: int, t: int) -> List[int]:
    assignments = kmeans_assignments(points, k, t)
    n = len(points)

    cluster_hist = [[0] * L for _ in range(k)]
    cluster_size = [0] * k

    for i in range(n):
        c = assignments[i]
        cluster_hist[c][points[i].intensity] += 1
        cluster_size[c] += 1

    out = [0] * n
    for i in range(n):
        c = assignments[i]
        hist = cluster_hist[c]
        m = cluster_size[c]
        cdf, cmin = cdf_and_cmin(hist)
        out[i] = remap_intensity(points[i].intensity, cdf, cmin, m)
    return out


def write_output(path: str, points: Sequence[Point], new_intensities: Sequence[int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p, ni in zip(points, new_intensities):
            # Keep integer formatting when coordinates are integral in input.
            def fmt(v: float) -> str:
                if float(v).is_integer():
                    return str(int(v))
                return f"{v:.6f}".rstrip("0").rstrip(".")

            f.write(f"{fmt(p.x)} {fmt(p.y)} {fmt(p.z)} {int(ni)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Brute-force reference for local histogram equalization on 3D point clouds")
    parser.add_argument("input", help="Input file path")
    parser.add_argument("--out-dir", default=".", help="Directory for knn.txt, approx_knn.txt, kmeans.txt")
    args = parser.parse_args()

    n, k, t, points = read_input(args.input)
    if n != len(points):
        raise ValueError("Point count mismatch")

    knn_i = knn_equalized(points, k)
    approx_i = approx_knn_equalized(points, k)
    kmeans_i = kmeans_equalized(points, k, t)

    write_output(f"{args.out_dir}/correct_knn.txt", points, knn_i)
    write_output(f"{args.out_dir}/correct_approx_knn.txt", points, approx_i)
    write_output(f"{args.out_dir}/correct_kmeans.txt", points, kmeans_i)


if __name__ == "__main__":
    main()
