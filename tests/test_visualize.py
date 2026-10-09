"""Test CUDA process orchestration without requiring or impersonating a GPU run."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reference import MODES, Point
from visualize import run_cuda


class CudaVisualizationTests(unittest.TestCase):
    points = [Point(0, 0, 0, 100), Point(1, 0, 0, 120)]

    def test_runs_all_modes_and_keeps_separate_outputs(self):
        def execute(command, **kwargs):
            self.assertTrue(kwargs["check"])
            value = {"knn": 10, "approx_knn": 20, "kmeans": 30}[command[2]]
            Path(command[3]).write_text(f"0 0 0 {value}\n1 0 0 {value}\n")
            return subprocess.CompletedProcess(command, 0, stdout="test log", stderr="")

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch("visualize.subprocess.run", side_effect=execute) as process:
                outputs, runs = run_cuda(directory / "executable", directory / "input.txt",
                                         self.points, directory / "outputs", 30)
            self.assertEqual([call.args[0][2] for call in process.call_args_list], list(MODES))
            for mode in MODES:
                self.assertEqual(runs[mode]["stdout"], "test log")
                self.assertTrue((directory / "outputs" / f"{mode}.txt").is_file())
            self.assertEqual(outputs["knn"][0][3], 10)
            self.assertEqual(outputs["approx_knn"][0][3], 20)

    def test_failed_cuda_run_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            stale = directory / "knn.txt"
            stale.write_text("old output")
            error = subprocess.CalledProcessError(1, ["executable"], stderr="no CUDA device")
            with patch("visualize.subprocess.run", side_effect=error):
                with self.assertRaisesRegex(RuntimeError, "no CUDA device"):
                    run_cuda(directory / "executable", directory / "input.txt", self.points, directory, 30)
            self.assertEqual(stale.read_text(), "old output")

    def test_reordered_cuda_output_is_rejected(self):
        def execute(command, **kwargs):
            Path(command[3]).write_text("1 0 0 10\n0 0 0 20\n")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch("visualize.subprocess.run", side_effect=execute):
                with self.assertRaisesRegex(ValueError, "coordinate order"):
                    run_cuda(directory / "executable", directory / "input.txt", self.points, directory, 30)
            self.assertFalse((directory / "knn.txt").exists())

    def test_missing_executable_has_no_reference_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            result = subprocess.run([sys.executable, str(ROOT / "scripts/visualize.py"),
                                     "--binary", str(directory / "missing"),
                                     "--data-dir", str(directory / "data")],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Build it with make gpu", result.stderr)
            self.assertFalse((directory / "data").exists())


if __name__ == "__main__":
    unittest.main()
