import unittest

import numpy as np

import ed_reference as ed
from run_experiments import make_input
from run_normal_schedule import (advancing_expression, expected_endpoint_times,
                                 impulse_reference, leg_count, sample_count, select_samples)


class NormalScheduleTests(unittest.TestCase):
    def test_leg_and_endpoint_counts(self):
        self.assertEqual([leg_count(t, .6) for t in (.2, .1, .05)], [4, 6, 12])
        np.testing.assert_allclose(expected_endpoint_times(4, .2), [.8, .8, 1., 1.])
        np.testing.assert_allclose(expected_endpoint_times(4, .2, 3), [.8, .8, .9, 1., .9, 1.])
        self.assertEqual(sample_count(.2, .6), 3)
        for tau, end in ((0, .6), (.2, .65), (float('nan'), .6)):
            with self.assertRaises(ValueError):
                sample_count(tau, end)

    def test_numeric_advancement_expression(self):
        expression = advancing_expression(.1, 7, 'P0')
        self.assertIn('advanceEach=5', expression)
        self.assertIn('steps=2', expression)
        self.assertIn('EnergyForExp=0', expression)
        self.assertTrue(expression.endswith('*|P0>'))
        self.assertIn('steps=3', advancing_expression(.1, 7, 'P0', 3))

    def test_bath_energies_in_both_spin_blocks(self):
        text = make_input('test', nsites=3, nup=2, ndown=1, bath_energies=[-.4, .6])
        self.assertIn('potentialV=[-1.0,-0.4,0.6,-1.0,-0.4,0.6];', text)
        with self.assertRaises(ValueError):
            make_input('test', bath_energies=[1])

    def test_physical_clock_and_latest_duplicate(self):
        def row(base, value, site=0):
            return {'site': site, 'bra_ket': 'label', 'time': base, 'value_raw': value}
        job = {'parsed': {'measurements': [row(0, '1'), row(0, '.9'),
                row(.2, '.8'), row(.4, '.7'), row(.6, '.6'), row(.2, '100', 1)]}}
        samples = select_samples(job, 'label', .2, 3)
        np.testing.assert_allclose([s['time'] for s in samples], [.2, .4, .6])
        self.assertEqual(samples[0]['value'], [.9, 0.])
        self.assertAlmostEqual(samples[0]['duplicate_spread'], .1)
        self.assertEqual(len(samples[0]['records']), 2)
        with self.assertRaises(RuntimeError):
            select_samples(job, 'missing', .2, 3)
        job['parsed']['measurements'].append(row(.03, '1'))
        with self.assertRaises(RuntimeError):
            select_samples(job, 'label', .2, 3)

    def test_impulse_matches_full_grid_oracle(self):
        v = [.3 + .2j, -.2 + .1j]
        energies = [-.4, .7]
        for seed, up, down, channel in [('alpha', [0, 1], [1], 'lesser'),
                                         ('beta', [1], [0, 1], 'greater')]:
            with self.subTest(seed=seed):
                result = impulse_reference(3, 2., seed, v, energies, .1, 3)
                sector, state = ed.product_state(3, up, down)
                exact = ed.green_functions(3, 2., sector, [state], [(.1, v)] * 3,
                                           potential=[-1.] + energies)
                np.testing.assert_allclose(result['greens'], exact[channel][:, 0], atol=1e-12)
                np.testing.assert_allclose(result['observables'][:, 0],
                                           np.diag(exact['lesser']) / 1j, atol=1e-12)
                self.assertGreater(result['scales']['initial_energy_stddev'], 0)


if __name__ == '__main__':
    unittest.main()
