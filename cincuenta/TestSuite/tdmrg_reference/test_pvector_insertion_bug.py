"""Opt-in engine regression for TargetingExpression P-vector insertion."""
import os
from pathlib import Path
import tempfile
import unittest

from run_experiments import make_input, measured, run_stage


class PvectorInsertionBugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        executable = os.environ.get("DMRG_EXECUTABLE")
        if not executable:
            raise unittest.SkipTest("set DMRG_EXECUTABLE to run engine regression")
        path = Path(executable)
        if not path.is_absolute() or not path.is_file():
            raise RuntimeError("DMRG_EXECUTABLE must name an existing absolute executable")
        temporary_base = Path(__file__).resolve().parents[3] / "build" / "tmp"
        temporary_base.mkdir(parents=True, exist_ok=True)
        cls._temporary = tempfile.TemporaryDirectory(
            prefix="pvector-insertion-", dir=temporary_base)
        cls.addClassCleanup(cls._temporary.cleanup)
        root = Path(cls._temporary.name)
        common = dict(nsites=5, nup=3, ndown=2, m=64, couplings=[0.] * 4,
                      onsite_u=0., potential=0.,
                      finite_loops="[[@auto,64,2],[@auto,64,2]]")
        measurements = ["<P1|c|P0>", "<P1|P0>", "<P1|P1>"]
        cls.results = {}
        cls.amplitudes = {}
        for name, p1 in (("control", 'c[0]*|gs>'),
                         ("regression", 'c[0]*|P0>')):
            # A bare P0=|gs> is an alias, not a materialized saved target.
            # H=0 makes TimeEvolve produce an unchanged, nonempty P0 vector.
            text = make_input(name, expressions=[
                'string P0="TimeEvolve{tau=.1,steps=3,advanceEach=3,EnergyForExp=0}*|gs>";',
                f'string P1="{p1}";'], **common)
            cls.results[name] = run_stage(path, root, name, text,
                                          measurements=measurements)
            # Setup/measurement failures must not masquerade as the known bug.
            cls.amplitudes[name] = [measured(cls.results[name], label) for label in measurements]

    def assert_amplitudes(self, values):
        for actual, expected in zip(values, (1., 0., 1.)):
            self.assertAlmostEqual(actual, expected, places=8)

    def test_direct_ground_state_control(self):
        self.assert_amplitudes(self.amplitudes["control"])

    def test_regression_has_known_bug_or_correct_signature(self):
        actual = self.amplitudes["regression"]
        signatures = ((0., 1., 1.), (1., 0., 1.))
        self.assertTrue(any(all(abs(a-b) < 1e-8 for a, b in zip(actual, signature))
                            for signature in signatures),
                        f"unexpected insertion failure signature: {actual}")

    @unittest.expectedFailure
    def test_pvector_source_insertion_regression(self):
        self.assert_amplitudes(self.amplitudes["regression"])


if __name__ == "__main__":
    unittest.main()
