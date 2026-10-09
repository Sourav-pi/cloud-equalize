import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_dataset import generate_dataset
from mae import compare_rows, read_output
from reference import (Point, build_neighbors, equalize, morton, morton_candidates,
                       neighbor_equalized, read_input, trunc_div)


class ReferenceTests(unittest.TestCase):
    def test_self_inclusion_and_constant_histogram(self):
        points = [Point(0, 0, 0, 100), Point(1, 0, 0, 200)]
        self.assertEqual(neighbor_equalized(points, 1), [0, 255])
        self.assertEqual(neighbor_equalized([Point(0, 0, 0, 70), Point(1, 0, 0, 70)], 1), [70, 70])

    def test_coordinate_tie_break(self):
        points = [Point(0, 0, 0, 100), Point(1, 0, 0, 200), Point(-1, 0, 0, 50)]
        self.assertEqual(build_neighbors(points, 1)[0], [2])
        self.assertEqual(neighbor_equalized(points, 1)[0], 255)

    def test_morton_bits_and_window_boundaries(self):
        self.assertEqual(morton(Point(-10000, -10000, -10000, 0)), 0)
        self.assertEqual(morton(Point(-9999, -10000, -10000, 0)), 1)
        self.assertEqual(morton(Point(-10000, -9999, -10000, 0)), 2)
        order = list(range(20000))
        self.assertEqual(morton_candidates(order, 0), order[:15001])
        self.assertEqual(morton_candidates(order, 127), morton_candidates(order, 0))
        self.assertEqual(morton_candidates(order, 8192), order[756:15757])
        self.assertEqual(morton_candidates(order, 19999), order[12532:])

    def test_kmeans_zero_iterations_tie_and_negative_mean(self):
        points = [Point(2, 0, 0, 200), Point(-2, 0, 0, 10), Point(0, 0, 0, 100)]
        self.assertEqual(equalize(points, 2, 0, "kmeans"), [200, 0, 255])
        self.assertEqual(trunc_div(-5, 2), -2)
        self.assertEqual(equalize([Point(-3, 0, 0, 10), Point(-2, 0, 0, 200)], 1, 5, "kmeans"), [0, 255])

    def test_mae_rejects_reordered_and_truncated_data(self):
        rows = [(0, 0, 0, 10), (1, 0, 0, 20)]
        with self.assertRaises(ValueError):
            compare_rows(rows, rows[:1])
        with self.assertRaises(ValueError):
            compare_rows(rows, rows[::-1])
        self.assertEqual(compare_rows(rows, [(0, 0, 0, 20), (1, 0, 0, 20)])["mae"], 5)

    def test_generator_reproducible_and_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            left, right = Path(temp) / "left.txt", Path(temp) / "right.txt"
            generate_dataset(40, 4, 0, left, seed=7)
            generate_dataset(40, 4, 0, right, seed=7)
            self.assertEqual(left.read_bytes(), right.read_bytes())
            self.assertEqual(read_input(left)[:3], (40, 4, 0))


class ExecutableTests(unittest.TestCase):
    def run_and_compare(self, binary, points, k, iterations, modes):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "input.txt", Path(temp) / "output.txt"
            source.write_text(f"{len(points)}\n{k}\n{iterations}\n" + "".join(
                f"{p.x} {p.y} {p.z} {p.intensity}\n" for p in points))
            for mode in modes:
                subprocess.run([str(binary), str(source), mode, str(output)],
                               capture_output=True, text=True, check=True, timeout=60)
                oracle_mode = "knn" if binary.name == "cpu_reference" and mode == "approx_knn" else mode
                expected = [(p.x, p.y, p.z, value) for p, value in
                            zip(points, equalize(points, k, iterations, oracle_mode))]
                self.assertEqual(compare_rows(expected, read_output(output))["different_points"], 0)

    def test_cpu_matches_reference_on_sample_and_seeded_clouds(self):
        binary = ROOT / "build/cpu_reference"
        _, k, iterations, points = read_input(ROOT / "examples/sample.txt")
        self.run_and_compare(binary, points, k, iterations, ("knn", "approx_knn", "kmeans"))
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "input.txt"
            for seed in range(3):
                for iterations in (0, 1, 10):
                    generate_dataset(35, 4, iterations, source, seed)
                    _, k, _, points = read_input(source)
                    self.run_and_compare(binary, points, k, iterations, ("knn", "kmeans"))

    def test_cpu_coordinate_ties_and_parameter_edges(self):
        points = [Point(0, 0, 0, 100), Point(1, 0, 0, 200), Point(-1, 0, 0, 50)]
        for iterations in (0, 1, 50):
            self.run_and_compare(ROOT / "build/cpu_reference", points, 1, iterations, ("knn", "kmeans"))
        self.run_and_compare(ROOT / "build/cpu_reference", [Point(0, 0, 0, 70)], 1, 0, ("kmeans",))
        large = [Point(i, 0, 0, i % 256) for i in range(129)]
        self.run_and_compare(ROOT / "build/cpu_reference", large, 128, 1, ("knn", "kmeans"))

    def test_cpu_rejects_invalid_input_and_unwritable_output(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "input.txt"
            for content in ("2 129 1\n", "2 1 1\n0 0 0 256\n1 0 0 20\n",
                            "2 1 1\n0 0 0 10\n0 0 0 20\n",
                            "2 1 1\n-10001 0 0 10\n1 0 0 20\n",
                            "2 1 1\n0 0 0 10\n1 0 0 20\nextra\n"):
                source.write_text(content)
                result = subprocess.run([str(ROOT / "build/cpu_reference"), str(source), "knn"], capture_output=True)
                self.assertNotEqual(result.returncode, 0)
            source.write_text("2 1 1\n0 0 0 10\n1 0 0 20\n")
            result = subprocess.run([str(ROOT / "build/cpu_reference"), str(source), "knn",
                                     str(Path(temp) / "missing/output.txt")], capture_output=True)
            self.assertNotEqual(result.returncode, 0)

    def test_cpu_benchmark_report(self):
        with tempfile.TemporaryDirectory() as temp:
            report_path = Path(temp) / "report.json"
            subprocess.run([sys.executable, str(ROOT / "scripts/benchmark.py"), "--backend", "cpu",
                            "--runs", "2", "--warmup", "1", "--output", str(report_path)],
                           check=True, capture_output=True, text=True, timeout=60)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["dataset"]["n"], 8)
            self.assertEqual(set(report["measurements"]), {"cpu/knn", "cpu/kmeans"})
            for measurement in report["measurements"].values():
                self.assertEqual(len(measurement["trials"]), 2)
                self.assertEqual(len(measurement["warmups"]), 1)
            self.assertIn("build", report["executables"]["cpu"])

    @unittest.skipUnless(os.environ.get("POINTCLOUD_TEST_GPU") == "1", "Set POINTCLOUD_TEST_GPU=1 on a CUDA host")
    def test_gpu_matches_reference(self):
        _, k, iterations, points = read_input(ROOT / "examples/sample.txt")
        self.run_and_compare(ROOT / "build/pointcloud_eq", points, k, iterations, ("knn", "approx_knn", "kmeans"))
        points = [Point(i - 130, i % 17, -(i % 11), (i * 7) % 256) for i in range(257)]
        for iterations in (0, 1, 10):
            self.run_and_compare(ROOT / "build/pointcloud_eq", points, 128, iterations, ("knn", "approx_knn", "kmeans"))


if __name__ == "__main__":
    unittest.main()
