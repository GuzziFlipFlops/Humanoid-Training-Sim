"""Counterfactual incentive tests; not evidence of a trained gait."""
import math
import unittest

import torch

from stage_rewards import CurriculumRewards
from heading_control import yaw_from_quat, wrap_angle
from test_unitree_reward_port import fixture, NAMES, BODIES, ROOT, FOOT
from test_stage_rewards import telemetry


class FixedHeadingReward(unittest.TestCase):
    def setUp(self):
        self.reward = CurriculumRewards(NAMES, BODIES, BODIES, ROOT, FOOT)

    def score(self, speed, heading, target=0., terminal=False):
        env = fixture(speed, 4)
        d = env.scene['robot'].data
        d.root_quat_w[:] = torch.tensor([[math.cos(heading/2), 0., 0., math.sin(heading/2)]])
        d.root_lin_vel_w[:] = torch.tensor([[speed*math.cos(heading), speed*math.sin(heading), 0.]])
        metrics = telemetry(env)
        metrics['heading_error_rad'] = wrap_angle(torch.tensor([target])-yaw_from_quat(d.root_quat_w))
        metrics['phase'] = torch.tensor([.1])
        env.cad_metrics = metrics
        env.termination_manager.terminated[:] = terminal
        total, _, terms = self.reward.evaluate(env)
        return float(total/env.step_dt), terms

    def test_straight_beats_circling_with_equal_body_speed_and_gait(self):
        for speed in (.15, .35, 1., 1.5, 2.5):
            straight, base = self.score(speed, 0.)
            for angle in (math.radians(20), math.pi/2, math.pi):
                curved, terms = self.score(speed, angle)
                self.assertGreater(straight, curved)
                for key in base:
                    if key != 'fixed_heading':
                        torch.testing.assert_close(base[key], terms[key], atol=1e-6, rtol=1e-6)
                self.assertGreater(float(base['fixed_heading']-terms['fixed_heading']), .6)

    def test_heading_error_is_symmetric_and_wraps_at_pi(self):
        for angle in (.1, .35, 1., 2.9):
            _, left = self.score(.35, angle)
            _, right = self.score(.35, -angle)
            torch.testing.assert_close(left['fixed_heading'], right['fixed_heading'])
        _, a = self.score(.35, -math.pi+.01, target=math.pi-.01)
        _, b = self.score(.35, .02)
        torch.testing.assert_close(a['fixed_heading'], b['fixed_heading'])

    def test_alignment_has_no_global_yaw_preference(self):
        for reset in (0., 1.3, -2.7):
            _, terms = self.score(.35, reset+.1, target=reset)
            _, reference = self.score(.35, .1)
            torch.testing.assert_close(terms['fixed_heading'], reference['fixed_heading'])

    def test_stopped_commands_cannot_earn_heading_bonus(self):
        for heading in (0., .5, math.pi):
            _, terms = self.score(0., heading)
            self.assertEqual(float(terms['fixed_heading']), 0.)

    def test_physical_fault_removes_all_regular_credit(self):
        rate, terms = self.score(.35, 0., terminal=True)
        self.assertAlmostEqual(rate, -200., places=4)
        self.assertTrue(all(float(value)==0. for key, value in terms.items() if key!='physical_termination'))

    def test_heading_credit_cannot_pay_for_in_place_steps_or_slide(self):
        from test_unitree_reward_port import contacts, FEET
        def cycle(mode):
            scores=[]
            for tick in range(80):
                env=fixture(.35,tick);d=env.scene['robot'].data
                phase=tick*.02/.8%1.
                flags=(phase<.55,(phase+.5)%1.<.55)
                if mode=='slide':flags=(True,True);d.body_lin_vel_w[:,FEET,0]=.35
                if mode=='in_place':d.root_lin_vel_w.zero_();d.root_lin_vel_b.zero_()
                contacts(env,flags)
                metrics=telemetry(env,flags);metrics['phase']=torch.tensor([phase])
                metrics['clearance'][0]=torch.tensor([0. if flag else .04 for flag in flags])
                env.cad_metrics=metrics;scores.append(float(self.reward.evaluate(env)[0]/env.step_dt))
            return sum(scores)/len(scores)
        walking=cycle('walk')
        self.assertGreater(walking,cycle('slide'))
        self.assertGreater(walking,cycle('in_place'))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
