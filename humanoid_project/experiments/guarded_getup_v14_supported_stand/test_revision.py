import unittest
from types import SimpleNamespace
import torch
from inference_helpers import push_laterally
from reward_core import RecoveryReward, Limits, terminal_rehearsal_score, terminal_fall_mask
from test_reward_core import laying
from assistance import FULL_FORCE_N, vertical_assist_force


class RevisionTests(unittest.TestCase):
    def test_assistance_is_bounded_monotonic_and_zero_before_success_height(self):
        height = torch.tensor([.08, .20, .45, .70, .78, 1.0])
        force = vertical_assist_force(height, .20)
        self.assertAlmostEqual(float(force[0]), .20 * FULL_FORCE_N, places=4)
        self.assertAlmostEqual(float(force[1]), .20 * FULL_FORCE_N, places=4)
        self.assertTrue((force[:-1] >= force[1:]).all())
        self.assertTrue((force[3:] == 0.).all())
        self.assertTrue((vertical_assist_force(height, 0.) == 0.).all())
        stronger = vertical_assist_force(height, .35)
        self.assertAlmostEqual(float(stronger[0]), .35 * FULL_FORCE_N, places=4)
        self.assertTrue((stronger[3:] == 0.).all())
        with self.assertRaises(ValueError):
            vertical_assist_force(height, .41)

    def test_terminal_rehearsal_rejects_lying_unsupported_and_thrashing(self):
        f = laying()
        f.height[:] = .86
        f.pelvis_tilt.zero_(); f.torso_tilt.zero_()
        f.foot_load[:] = .5; f.nonfoot_force.zero_()
        nominal = terminal_rehearsal_score(f)
        self.assertTrue((nominal > .99).all())
        f.height[:] = .10
        self.assertTrue((terminal_rehearsal_score(f) < 1e-6).all())
        f.height[:] = .86; f.foot_load.zero_()
        self.assertTrue((terminal_rehearsal_score(f) == 0.).all())
        f.foot_load[:] = .5; f.joint_speed_rms[:] = 8.
        # Motion degrades a valid stand but cannot erase all learning signal.
        self.assertTrue((terminal_rehearsal_score(f) < nominal).all())

    def test_terminal_fall_only_ends_failed_upright_rehearsal(self):
        f = laying()
        groups = torch.tensor([4, 0])
        elapsed = torch.tensor([20, 20])
        self.assertEqual(terminal_fall_mask(f, groups, elapsed).tolist(), [True, False])
        f.height[:] = .86; f.pelvis_tilt.zero_(); f.torso_tilt.zero_()
        self.assertFalse(terminal_fall_mask(f, groups, elapsed).any())
        f.height[:] = .10
        self.assertFalse(terminal_fall_mask(f, groups, torch.tensor([4, 4])).any())

    def test_progress_telescopes_and_unsafe_launch_cannot_profit(self):
        f = laying(); f.height[:] = .08
        core = RecoveryReward(2, limits=Limits(progress_budget=4., failure_penalty=8.))
        core.reset(slice(None), f, torch.tensor([0, 1]), torch.ones(2, dtype=torch.bool))
        f.height[:] = .55
        f.vertical_speed[1] = 2.5
        out = core.update(f, 0)
        # Potential progress is deliberately speed-independent; safe dynamic
        # recovery is permitted and excessive speed is handled by the
        # separate rise-speed/unsafe terms.
        self.assertTrue(torch.allclose(out.progress[0], out.progress[1]))
        f.vertical_speed[:] = 0.
        self.assertFalse(core.update(f, 1).progress.any())
        ended = core.update(f, 2, torch.ones(2, dtype=torch.bool))
        self.assertTrue((ended.failure == 8.).all())
        # Runtime unsafe cost is 40, larger than the full 24 progress budget
        # plus avoidable floor cost, so even a maximally high launch loses.
        self.assertLess(24. - 40., -8.)

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
        # The sequence ends back at its initial height, so every positive
        # increment has been paid back exactly.
        self.assertTrue(torch.allclose(total, torch.zeros_like(total), atol=1e-6))

    def test_rolling_progress_beyond_ninety_degrees(self):
        core = RecoveryReward(2)
        f = laying(); f.torso_tilt[:] = torch.deg2rad(torch.tensor(120.))
        a = core.quality(f).clone()
        f.torso_tilt[:] = torch.deg2rad(torch.tensor(110.))
        self.assertTrue((core.quality(f) > a).all())


if __name__ == '__main__': unittest.main()
