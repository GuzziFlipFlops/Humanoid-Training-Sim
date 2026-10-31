import unittest
from types import SimpleNamespace
import torch
from inference_helpers import push_laterally
from reward_core import RecoveryReward, Limits
from test_reward_core import laying


class RevisionTests(unittest.TestCase):
    def test_fast_progress_is_discounted_and_not_cashable_later(self):
        f = laying(); f.height[:] = .08
        core = RecoveryReward(2, limits=Limits(progress_budget=4., failure_penalty=8.))
        core.reset(slice(None), f, torch.tensor([0, 1]), torch.ones(2, dtype=torch.bool))
        f.height[:] = .55
        f.vertical_speed[1] = 2.5
        out = core.update(f, 0)
        self.assertLess(float(out.progress[1]), float(out.progress[0]) * .1)
        f.vertical_speed[:] = 0.
        self.assertFalse(core.update(f, 1).progress.any())
        ended = core.update(f, 2, torch.ones(2, dtype=torch.bool))
        self.assertTrue((ended.failure == 8.).all())
        self.assertTrue(((out.progress - ended.failure) < 0.).all())

    def test_inference_buffer_push(self):
        with torch.inference_mode():
            buffer = torch.zeros(2, 6)
        robot = SimpleNamespace(data=SimpleNamespace(root_vel_w=buffer),
                                write_root_velocity_to_sim=lambda v: buffer.copy_(v))
        push_laterally(robot, .1)
        self.assertTrue(torch.allclose(buffer[:, 1], torch.full((2,), .1)))
        self.assertEqual(float(buffer[:, [0, 2, 3, 4, 5]].abs().sum()), 0.)

    def test_small_rise_below_old_dead_zone_has_signal(self):
        f = laying(); f.height[:] = .08
        core = RecoveryReward(2, limits=Limits(progress_budget=4.))
        core.reset(slice(None), f, torch.tensor([0, 1]), torch.ones(2, dtype=torch.bool))
        f.height[:] = .10
        out = core.update(f, 0)
        self.assertTrue((out.progress > .05).all())
        self.assertFalse(out.success.any())
        self.assertEqual(float(core.update(f, 1).progress.sum()), 0.)
        total = out.progress.clone()
        for tick in range(2, 102):
            f.height[:] = .08 if tick % 2 else .10
            total += core.update(f, tick).progress
        self.assertTrue(torch.allclose(total, out.progress))

    def test_rolling_progress_beyond_ninety_degrees(self):
        core = RecoveryReward(2)
        f = laying(); f.torso_tilt[:] = torch.deg2rad(torch.tensor(120.))
        a = core.quality(f).clone()
        f.torso_tilt[:] = torch.deg2rad(torch.tensor(110.))
        self.assertTrue((core.quality(f) > a).all())


if __name__ == '__main__': unittest.main()
