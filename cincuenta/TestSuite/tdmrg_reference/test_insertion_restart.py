from dataclasses import replace
import unittest
from unittest.mock import patch

from insertion_restart import plan_insertion
from run_experiments import Case


class InsertionAdapterTests(unittest.TestCase):
    def test_initial_gs_is_not_a_workaround(self):
        plan = plan_insertion('hole', 0.)
        self.assertEqual(plan.backend, 'initial-gs')
        self.assertIsNone(plan.source_tv_for_psi)
        self.assertEqual(plan.expressions, ('string P0="c[0]*|gs>";',))
        self.assertEqual(plan.physical_clock_origin, 0.)
        self.assertEqual(plan.first_pair_mapping, (0, -1))

    def test_saved_reference_promotion_owns_clock_reset(self):
        plan = plan_insertion('particle', .4, source_endpoint=2)
        self.assertEqual(plan.backend, 'gs-promotion-workaround')
        self.assertEqual(plan.source_tv_for_psi, 2)
        self.assertEqual(plan.restart_mapping, (-1,))
        self.assertEqual(plan.expressions, ('string P0="c\'[0]*|gs>";',))
        self.assertEqual(plan.measurement_label, '<gs|c|P0>')
        self.assertEqual(plan.local_insertion_time, 0.)
        self.assertAlmostEqual(plan.evolution_start(.6), .2)
        self.assertTrue(plan.first_pair_from_gs)

    def test_execution_passes_adapter_fields_without_reinterpreting_them(self):
        case = Case('unused', 'unused', 5, 64, .2, 3, 2., 120.)
        with patch.object(case, 'stage', return_value={'pvector_times': [0.]}) as stage:
            job, plan = case.insert_branch('insert', 'alpha', 'hole', 'checkpoint', .4, 1, [0]*4)
        self.assertEqual(stage.call_args.kwargs, {'source_tv_for_psi': 1})
        self.assertEqual(stage.call_args.args[3:5], ('checkpoint', [-1]))
        self.assertEqual(job['physical_clock_origin'], .4)
        self.assertEqual(job['insertion_backend'], plan.backend)

    def test_execution_contract_also_accepts_physical_clock_backend(self):
        # A future direct backend must satisfy this contract. This is a mocked
        # adapter test, NOT a claim that operator-on-P works in the current engine.
        direct = replace(plan_insertion('hole', .4, 2),
            expressions=('string P0="c[0]*|P1>";', 'string P1=|P1>;'),
            restart_mapping=(-1, 2), source_tv_for_psi=None,
            reference_ket='P1', first_pair_mapping=(0, 1),
            first_pair_from_gs=False, local_insertion_time=.4,
            physical_clock_origin=0., backend='direct-pvector')
        case = Case('unused', 'unused', 5, 64, .2, 3, 2., 120.)
        with patch('run_experiments.insertion_restart.plan_insertion', return_value=direct), \
             patch.object(case, 'stage', return_value={'pvector_times': [.4, .4]}) as stage:
            job, plan = case.insert_branch('insert', 'alpha', 'hole', 'checkpoint', .4, 2, [0]*4)
        self.assertEqual(stage.call_args.args[4], [-1, 2])
        self.assertIsNone(stage.call_args.kwargs['source_tv_for_psi'])
        self.assertEqual(plan.measurement_label, '<P0|c|P1>')
        self.assertFalse(plan.first_pair_from_gs)
        self.assertAlmostEqual(plan.evolution_start(.6), .6)
        self.assertEqual(job['physical_clock_origin'], 0.)

    def test_invalid_requests(self):
        for args in [('bad', 0., None), ('hole', .2, None), ('hole', -.1, 1),
                     ('hole', .2, -1), ('hole', float('nan'), 1)]:
            with self.assertRaises(ValueError):
                plan_insertion(*args)


if __name__ == '__main__':
    unittest.main()
