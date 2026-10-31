"""Regression for the per-pose tensor cache leak, without Isaac or CUDA."""
import json
import unittest
from pathlib import Path

import torch

from cad_differential import CadDifferential

HERE = Path(__file__).resolve().parent
SPECS = json.loads((HERE / 'assets/differential_geometry.json').read_text())


class DifferentialCacheTests(unittest.TestCase):
    def test_pose_tensors_never_become_cache_keys(self):
        for name, spec in SPECS.items():
            with self.subTest(pair=name):
                model = CadDifferential(spec)
                for tick in range(128):
                    q = torch.full((32, 2), tick / 10000., dtype=torch.float64)
                    model.state(q)
                    if tick == 0:
                        initial = len(model._tensors)
                    self.assertEqual(initial, len(model._tensors))
                self.assertLessEqual(initial, 16)
                for key in model._tensors:
                    self.assertTrue(all(isinstance(v, (int, float)) for v in key[2]))
                # A second dtype gets a second set of immutable constants only.
                model.state(q.float())
                self.assertEqual(2 * initial, len(model._tensors))

    def test_jacobian_still_matches_physical_closure_derivative(self):
        for name, spec in SPECS.items():
            with self.subTest(pair=name):
                model = CadDifferential(spec)
                q = torch.tensor([[0., 0.], [.02, -.015], [-.015, .02]], dtype=torch.float64)
                theta, jacobian, valid, error = model.state(q)
                self.assertTrue(valid.all())
                self.assertLess(float(error.max()), 1e-8)
                numerical = []
                for axis in range(2):
                    delta = torch.zeros_like(q)
                    delta[:, axis] = 1e-6
                    plus = model.state(q + delta)[0]
                    minus = model.state(q - delta)[0]
                    difference = torch.atan2(torch.sin(plus-minus), torch.cos(plus-minus))
                    numerical.append(difference / 2e-6)
                torch.testing.assert_close(jacobian, torch.stack(numerical, -1), atol=2e-7, rtol=2e-6)

    def test_shared_motor_torque_and_speed_limits_remain_enforced(self):
        for name, spec in SPECS.items():
            with self.subTest(pair=name):
                model = CadDifferential(spec)
                generator = torch.Generator().manual_seed(73)
                q = (torch.rand((256, 2), generator=generator, dtype=torch.float64)-.5)*2.
                velocity = (torch.rand((256, 2), generator=generator, dtype=torch.float64)-.5)*120.
                desired = (torch.rand((256, 2), generator=generator, dtype=torch.float64)-.5)*400.
                output, state = model.limit(q, velocity, desired, 45., 42., .95)
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue((state['motor_applied'].abs() <= 45.).all())
                accelerating_at_cap = (state['motor_velocity'].abs() >= 42.) & (state['motor_applied']*state['motor_velocity'] > 0.)
                self.assertFalse(accelerating_at_cap.any())
                self.assertFalse(output[~state['valid']].any())
                self.assertTrue(state['valid'].any())
                self.assertTrue((~state['valid']).any())
                reconstructed = .95 * (state['J'].transpose(1, 2) @ state['motor_applied'][..., None]).squeeze(-1)
                torch.testing.assert_close(output[state['valid']], reconstructed[state['valid']], atol=1e-10, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
