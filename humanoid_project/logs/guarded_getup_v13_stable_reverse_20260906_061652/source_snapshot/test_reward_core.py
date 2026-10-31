"""Synthetic adversarial contract tests, not evidence of a trained get-up policy."""
import unittest
import torch
from reward_core import Frame, RecoveryReward


def standing(n=2):
    def full(value): return torch.full((n,), float(value))
    return Frame(full(.84), full(0), full(0), torch.zeros(n, 2), full(0),
                 full(0), full(0), full(0), full(0), torch.full((n, 2), .5),
                 torch.zeros(n, 2), torch.zeros(n, 2), torch.zeros(n, 2, 2),
                 full(0), full(.03), torch.ones(n, dtype=torch.bool), torch.ones(n, dtype=torch.bool))


def laying(n=2):
    f = standing(n)
    f.height[:] = .25
    f.torso_tilt[:] = torch.pi / 2
    f.pelvis_tilt[:] = torch.pi / 2
    f.foot_load[:] = 0
    f.nonfoot_force[:] = 150
    return f


class ContractTests(unittest.TestCase):
    def init(self, dt=.02, initial=None):
        r = RecoveryReward(2, dt=dt)
        r.reset(slice(None), initial or laying(), torch.tensor([0, 1]), torch.ones(2, dtype=torch.bool))
        return r

    def run_frames(self, r, f, steps):
        total = torch.zeros(2)
        successes = torch.zeros(2)
        for _ in range(steps):
            out = r.update(f, r.last_tick + 1)
            total += out.step_reward
            successes += out.success
        return total, successes, out

    def test_laying_has_no_height_farming(self):
        total, events, _ = self.run_frames(self.init(), laying(), 1000)
        self.assertEqual(float(total.sum()), 0.)
        self.assertFalse(events.any())

    def test_full_seven_seconds_required(self):
        r = self.init()
        _, events, _ = self.run_frames(r, standing(), 349)
        self.assertFalse(events.any())
        out = r.update(standing(), 349)
        self.assertTrue(out.success.all())
        self.assertTrue(out.currently_valid.all())
        _, events, _ = self.run_frames(r, standing(), 100)
        self.assertFalse(events.any())

    def test_standing_initial_state_never_counts(self):
        _, events, out = self.run_frames(self.init(initial=standing()), standing(), 400)
        self.assertFalse(events.any())
        self.assertFalse(out.currently_valid.any())

    def test_bouncing_cannot_repeat_progress(self):
        r = self.init()
        total = torch.zeros(2)
        for i in range(1000):
            out = r.update(standing() if i % 2 else laying(), i)
            total += out.progress
        self.assertTrue((total <= 1.000001).all())
        self.assertFalse(r.ever_success.any())

    def test_torso_foot_support_contact_and_motion_fail_individually(self):
        cases = [('torso_tilt', .6), ('pelvis_tilt', .6), ('foot_load', 0.),
                 ('nonfoot_force', 10.), ('support_margin', -.01), ('foot_tilt', .5),
                 ('foot_slip', .1), ('planar_speed', .2), ('angular_speed', .5),
                 ('joint_speed_rms', 1.), ('height', 1.1)]
        for field, value in cases:
            with self.subTest(field=field):
                r, f = self.init(), standing()
                getattr(f, field)[:] = value
                _, events, _ = self.run_frames(r, f, 400)
                self.assertFalse(events.any())

    def test_one_foot_cannot_stand_in_for_two(self):
        r, f = self.init(), standing()
        f.foot_load[:, 0], f.foot_load[:, 1] = 0., 1.
        _, events, _ = self.run_frames(r, f, 400)
        self.assertFalse(events.any())

    def test_slow_skating_cannot_walk_the_hold_anchor(self):
        r, f = self.init(), standing()
        for i in range(500):
            f.xy[:, 0] += .001  # 0.05 m/s: below instantaneous speed threshold
            f.planar_speed[:] = .05
            out = r.update(f, i)
            self.assertFalse(out.success.any())

    def test_feet_cannot_shuffle_under_stationary_torso(self):
        r, f = self.init(), standing()
        for i in range(500):
            f.foot_xy[:, :, 0] += .0005
            out = r.update(f, i)
            self.assertFalse(out.success.any())

    def test_yaw_wrap_and_yaw_drift(self):
        r, f = self.init(), standing()
        f.yaw[:] = torch.pi - .001
        r.update(f, 0)
        f.yaw[:] = -torch.pi + .001
        self.assertTrue(r.update(f, 1).stable.all())
        self.assertTrue((r.hold_steps == 2).all())
        f.yaw[:] += .3
        r.update(f, 2)
        self.assertTrue((r.hold_steps == 0).all())

    def test_success_then_walkoff_invalidates_entire_trial(self):
        r = self.init()
        self.run_frames(r, standing(), 350)
        f = standing()
        f.xy[:, 0] = .2
        out = r.update(f, 350)
        self.assertTrue(out.invalidated_after_success.all())
        torch.testing.assert_close(out.failure, torch.full((2,), 20.))
        _, events, out = self.run_frames(r, standing(), 400)
        self.assertFalse(events.any())
        self.assertFalse(out.currently_valid.any())
        self.assertFalse(out.failure.any())

    def test_partial_reset_no_cross_episode_success(self):
        r = self.init()
        self.run_frames(r, standing(), 349)
        r.reset(torch.tensor([0]), laying(), torch.tensor([2, 3]), torch.ones(2, dtype=torch.bool))
        out = r.update(standing(), 349)
        self.assertEqual(out.success.tolist(), [False, True])

    def test_stale_contacts_and_unsafe_physics_rejected(self):
        for field in ('contacts_fresh', 'safe'):
            r, f = self.init(), standing()
            getattr(f, field)[:] = False
            _, events, _ = self.run_frames(r, f, 400)
            self.assertFalse(events.any())

    def test_invalid_pose_not_promoted(self):
        r = RecoveryReward(2)
        r.reset(slice(None), laying(), torch.tensor([0, 1]), torch.zeros(2, dtype=torch.bool))
        _, events, _ = self.run_frames(r, standing(), 400)
        self.assertFalse(events.any())

    def test_timeout_failure_once_and_dt_scaling(self):
        for dt in (.01, .02, .04):
            r = self.init(dt)
            f = standing()
            f.height[:] = .6
            out = r.update(f, 0, torch.ones(2, dtype=torch.bool))
            torch.testing.assert_close(out.rate * dt, out.step_reward)
            self.assertTrue((out.failure == 3.).all())
            self.assertTrue((out.step_reward < 0.).all())
            self.assertFalse(r.update(f, 1, torch.ones(2, dtype=torch.bool)).failure.any())

    def test_bonus_independent_of_control_dt(self):
        for dt in (.01, .02, .04):
            r = self.init(dt)
            total, events, out = self.run_frames(r, standing(), r.total_steps + 5)
            expected = 20. + r.quality(standing()) - r.quality(laying())
            torch.testing.assert_close(total, expected)
            torch.testing.assert_close(events, torch.ones(2))
            torch.testing.assert_close(out.rate * dt, out.step_reward)

    def test_nonfinite_raises_instead_of_appearing_safe(self):
        r, f = self.init(), standing()
        f.torso_tilt[0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            r.update(f, 0)

    def test_duplicate_tick_rejected(self):
        r = self.init()
        r.update(standing(), 0)
        with self.assertRaises(ValueError):
            r.update(standing(), 0)

    def test_critic_state_contains_hold_anchor_displacement(self):
        r, f = self.init(), standing()
        r.update(f, 0)
        before = r.state_observation(f)
        f.xy[:, 0] = .02
        after = r.state_observation(f)
        self.assertEqual(after.shape, (2, 13))
        self.assertFalse(torch.equal(before, after))


if __name__ == '__main__':
    unittest.main(verbosity=2)
