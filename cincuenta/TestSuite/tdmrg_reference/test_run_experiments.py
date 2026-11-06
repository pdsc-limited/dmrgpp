import importlib.util
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run_experiments", HERE / "run_experiments.py")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class GeneratorTests(unittest.TestCase):
    def test_seed_star_defaults_and_population(self):
        text = mod.make_input("seed")
        self.assertIn("GeometryKind=star;", text)
        self.assertIn("SolverOptions=twositedmrg,geometryallinsystem,TargetingExpression,usecomplex;", text)
        self.assertIn("gs=PureState[2,3,3,0,0];", text)
        self.assertIn("dir0:Connectors=[0,0,0,0];", text)

    def test_number_and_spin_populations(self):
        text = mod.make_input("seed", nsites=5, nup=2, ndown=3)
        self.assertIn("gs=PureState[1,3,3,0,0];", text)

    def test_reject_bad_coupling_or_particle_count(self):
        with self.assertRaises(ValueError): mod.make_input("x", couplings=[1])
        with self.assertRaises(ValueError): mod.make_input("x", nup=6)

    def test_insertions_and_propagation_mapping(self):
        self.assertIn('c[0]*|gs>', mod.insertion_expression("alpha", "hole"))
        self.assertIn("c'[0]*|gs>", mod.insertion_expression("beta", "particle"))
        first = mod.paired_expressions(.05)
        self.assertIn('*|P0>";', first[0])
        self.assertIn('*|gs>";', first[1])
        continuation = mod.paired_expressions(.05, continuation=True)
        self.assertIn('*|P0>";', continuation[0])
        self.assertIn('*|P1>";', continuation[1])
        self.assertIn("nup*ndown", mod.paired_measurements()[4])

    def test_restart_uses_checkpoint_basename_mapping(self):
        text = mod.make_input("next", restart="/some/path/state.hd5", restart_mapping=[2, 3])
        self.assertIn("RestartFilename=/some/path/state;", text)
        self.assertIn("RestartMappingTvs=[2,3];", text)
        self.assertIn("usecomplex,restart;", text)
        self.assertNotIn("gs=PureState", text)

    def test_complex_connectors_and_endpoint_time_check(self):
        text = mod.make_input("complex", couplings=[.3 + .1j, -.2j, .2, .1])
        self.assertIn("0.29999999999999999i0.10000000000000001", text)
        case = mod.Case("unused", "unused", 5, 64, .2, 3, 2., 60.)
        case.check_times({"name": "test", "pvector_times": [0., 0., .2, .2]}, [2, 3], [.2, .2])
        with self.assertRaises(RuntimeError):
            case.check_times({"name": "test", "pvector_times": [0., 0.]}, [0, 1], [.2, .2])


if __name__ == "__main__": unittest.main()
