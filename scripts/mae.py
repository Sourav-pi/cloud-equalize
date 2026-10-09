#!/usr/bin/env python3
"""Compare point-cloud outputs without accepting missing, reordered, or invalid records."""
import argparse
from pathlib import Path


def read_output(path):
    rows = [tuple(map(int, line.split())) for line in Path(path).read_text().splitlines()]
    if not rows or any(len(row) != 4 or not 0 <= row[3] <= 255 for row in rows):
        raise ValueError(f"Invalid output: {path}")
    return rows


def compare_rows(left, right):
    if len(left) != len(right) or any(a[:3] != b[:3] for a, b in zip(left, right)):
        raise ValueError("Point counts or coordinate order differ")
    errors = [abs(a[3] - b[3]) for a, b in zip(left, right)]
    return {"mae": sum(errors) / len(errors), "max_absolute_error": max(errors),
            "different_points": sum(error != 0 for error in errors), "points": len(errors)}


def mae_loss(left, right):
    return compare_rows(read_output(left), read_output(right))["mae"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exact")
    parser.add_argument("candidate")
    args = parser.parse_args()
    print(compare_rows(read_output(args.exact), read_output(args.candidate)))
