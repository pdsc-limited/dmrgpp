import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment_io", HERE / "experiment_io.py")
io = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = io
spec.loader.exec_module(io)


class ParseTests(unittest.TestCase):
    def test_targeting_and_insitu_output(self):
        text = """gsWeight=0.75 weights= 0.25
 gsNorms=0.999 e0;0.8  norms= 0.5 0.5
CurrentTime=0.125 CurrentTimeStep=1
-------------&*&*&* In-situ measurements start
2 (0.125,-0.25) 0.125 <gs|n|gs> 1
-------------&*&*&* In-situ measurements end
"""
        got = io.parse_output(text)
        self.assertEqual(got["gs_weight"], .75)
        self.assertEqual(got["weights"], [.25])
        self.assertEqual(got["norms"], [.5, .5])
        self.assertEqual(got["current_times"][0]["step"], 1)
        self.assertEqual(got["measurements"][0]["site"], 2)
        self.assertEqual(got["measurements"][0]["bra_ket"], "<gs|n|gs>")
        self.assertEqual(got["missing_metrics"], [])

    def test_pilot_discarded_weight_and_wavefunction_warning(self):
        text = """Discarded weight (Truncation error): 0
WaveFunctionTransformation completed WARNING: orig. norm= 1 resulting norm= 0.999
Transformation completed WARNING: orig. norm= 1 resulting norm= 0.999
"""
        got = io.parse_output(text)
        self.assertEqual(got["discarded_weights"], [0.0])
        self.assertEqual(len(got["wft_warnings"]), 2)

    def test_measurement_without_insitu_markers(self):
        got = io.parse_output("0 (0.125,-0.25) 0.2 <P0.last|c|P1.last> 1\n")
        self.assertEqual(got["measurements"][0]["bra_ket"], "<P0.last|c|P1.last>")
        self.assertEqual(got["parse_errors"], [])

    def test_malformed_expected_insitu_line_is_reported(self):
        got = io.parse_output("-------------&*&*&* In-situ measurements start\nnot a record\n")
        self.assertEqual(got["parse_errors"][0]["field"], "measurement")
        self.assertIn("gs_weight", got["missing_metrics"])


class RunTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith("linux"), "ru_maxrss units and fork inheritance checked on Linux")
    def test_child_rss_isolated_from_large_python_parent(self):
        base = HERE.parents[2] / "build" / "tmp"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as td:
            root = Path(td); inp = root / "input.in"; inp.write_text("input\n")
            # Touch pages so the scientific-style parent has a genuinely high RSS.
            allocation = bytearray(160 * 1024 * 1024)
            for offset in range(0, len(allocation), 4096): allocation[offset] = 1
            got = io.run_job([sys.executable, "-c", "pass"], inp, root)
            self.assertIsNotNone(got["peak_child_rss_bytes"])
            self.assertLess(got["peak_child_rss_bytes"], 100 * 1024 * 1024)
            del allocation

    def test_missing_executable_reports_helper_failure_status(self):
        base = HERE.parents[2] / "build" / "tmp"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as td:
            root = Path(td); inp = root / "input.in"; inp.write_text("input\\n")
            got = io.run_job([str(root / "does-not-exist")], inp, root)
            self.assertFalse(got["timed_out"])
            self.assertEqual(got["returncode"], 1)
            self.assertIsNone(got["peak_child_rss_bytes"])
            self.assertIn("FileNotFoundError", got["stderr"])

    def test_run_capture_provenance_checkpoint_and_timeout(self):
        base = HERE.parents[2] / "build" / "tmp"
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as td:
            root = Path(td); inp = root / "input.in"; inp.write_text("input\n")
            (root / "runForseed.cout").write_text("gsWeight=1 weights=\n")
            (root / "state.hd5").write_bytes(b"12345")
            script = "import os,sys; print(os.environ['OMP_NUM_THREADS']); sys.exit(7)"
            got = io.run_job([sys.executable, "-c", script], inp, os.path.relpath(root))
            self.assertEqual(got["returncode"], 7)
            self.assertEqual(got["parsed"]["gs_weight"], 1)
            self.assertIn("runForseed.cout", got["cout_logs"])
            self.assertIsNotNone(got["peak_child_rss_bytes"])
            self.assertEqual(got["checkpoints"], {"state.hd5": 5})
            self.assertEqual(len(got["input_sha256"]), 64)
            self.assertGreaterEqual(got["wall_seconds"], 0)
            slow = io.run_job([sys.executable, "-c", "import time; time.sleep(2)"], inp, root, timeout=.05)
            self.assertTrue(slow["timed_out"])
            self.assertIsNone(slow["returncode"])


if __name__ == "__main__":
    unittest.main()
