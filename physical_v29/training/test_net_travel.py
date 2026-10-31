"""Trajectory counterexamples for repeated rocking and stalled swing shaping."""
import math, unittest
import torch
from walking_rewards import ForwardProgress, walking_terms


class NetTravel(unittest.TestCase):
    def setUp(self):
        self.p = ForwardProgress(1, 'cpu')
        self.command = torch.tensor([.15])
        self.heading = torch.tensor([[1., 0.]])
        self.feet = torch.tensor([[[0., -.15, -.8], [0., .15, -.8]]])

    def tick(self, x, vx, contact=(True, True), event=False):
        events = torch.tensor([[float(event), 0.]])
        fresh, budget, recent = self.p.update(torch.tensor([[x, 0.]]), self.heading, self.command, events, .02)
        terms = walking_terms(self.command, torch.tensor([[vx, 0., 0.]]), torch.ones(1),
            torch.zeros(1), torch.tensor([contact]), torch.tensor([[True, False]]),
            torch.tensor([[0., .035]]), self.feet, self.feet, torch.zeros(1), events, .02,
            advance_fraction=fresh, swing_budget=budget, recent_step=recent)
        return terms, float(fresh), float(budget)

    def test_planted_zero_net_rocking_has_no_positive_locomotion(self):
        total = 0.
        for i in range(1000):
            t = i*.02
            terms, _, _ = self.tick(.15/(2*math.pi)*math.sin(2*math.pi*t), .15*math.cos(2*math.pi*t))
            for key in ('track', 'reference', 'gait', 'completed_step'):
                self.assertEqual(float(terms[key]), 0., key)
            total += float(terms['progress'])*.02
            self.assertAlmostEqual(float(terms['posture_scale']), 1.)
        self.assertLess(total, 0.)

    def test_repeat_excursion_cannot_refresh_forward_credit(self):
        credits = []
        for i in range(1000):
            t = i*.02
            _, fresh, _ = self.tick(.02*math.sin(2*math.pi*t), .126*math.cos(2*math.pi*t), (True, False))
            credits.append(fresh*.15*.02)
        self.assertLessEqual(sum(credits), .020001)
        self.assertLess(sum(credits[100:]), 1e-6)

    def test_stationary_leg_lifting_bonus_expires(self):
        terms = None
        for i in range(300): terms, _, budget = self.tick(0., 0., (True, False))
        self.assertEqual(budget, 0.)
        for key in ('gait', 'reference', 'track', 'progress'): self.assertEqual(float(terms[key]), 0.)

    def test_positive_command_change_does_not_renew_credit(self):
        for _ in range(200): self.tick(0., 0., (True, False))
        self.command[:] = .35
        terms, _, budget = self.tick(0., 0., (True, False))
        self.assertEqual(budget, 0.)
        self.assertEqual(float(terms['gait']), 0.)

    def test_genuine_advance_and_landings_keep_credit(self):
        score = 0.
        for i in range(400):
            terms, fresh, budget = self.tick(.15*i*.02, .15, (True, False), i%25==24)
            score += float(terms['track'])
        self.assertGreater(score/400, 2.45)
        self.assertEqual(budget, 1.)

    def test_episode_reset_does_not_pay_world_origin_jump(self):
        self.tick(0., .15)
        self.p.reset(torch.tensor([0]))
        _, fresh, _ = self.tick(1000., .15, (True, False))
        self.assertEqual(fresh, 0.)

    def test_heading_is_fixed_for_the_moving_bout(self):
        self.tick(0., .15)
        self.heading[:] = torch.tensor([[-1., 0.]])
        _, fresh, _ = self.tick(-.10, .15, (True, False))
        self.assertEqual(fresh, 0.)


if __name__ == '__main__': unittest.main(verbosity=2)
