#!/usr/bin/env python3
"""Generate reproducible unique integer point clouds within CUDA distance bounds."""
import argparse
from pathlib import Path
import random


def generate_dataset(n, k, iterations, output, seed=42):
    if not (2 <= n <= 100000 and 1 <= k <= min(128, n - 1) and 0 <= iterations <= 50):
        raise ValueError("Require 2<=n<=100000, 1<=k<=min(128,n-1), 0<=iterations<=50")
    rng = random.Random(seed)
    used = set()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        stream.write(f"{n}\n{k}\n{iterations}\n")
        while len(used) < n:
            coords = tuple(rng.randint(-10000, 10000) for _ in range(3))
            if coords in used:
                continue
            used.add(coords)
            stream.write("{} {} {} {}\n".format(*coords, rng.randint(0, 255)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/input.txt"))
    args = parser.parse_args()
    generate_dataset(args.n, args.k, args.iterations, args.output, args.seed)
    print(args.output)
