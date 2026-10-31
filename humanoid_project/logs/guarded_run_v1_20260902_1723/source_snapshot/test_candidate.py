"""Fast CPU tests: contact state, joint order math, and rejection gates."""
import unittest
import copy
import torch
from contact_state import CompletedStances, differential_cost
from gates import judge


class Tests(unittest.TestCase):
    def test_no_partial_initial_stance(self):
        t=CompletedStances(1,'cpu')
        for _ in range(10): t.update(torch.tensor([[True,True]]))
        for _ in range(4): t.update(torch.tensor([[False,False]]))
        self.assertEqual(int(t.count.sum()),0)

    def test_flicker_and_full_steps(self):
        t=CompletedStances(1,'cpu')
        off=torch.tensor([[False,False]]); on=~off
        t.update(off)
        for _ in range(5): t.update(on)
        t.update(off) # 20ms gap must not split stance
        for _ in range(5): t.update(on)
        for _ in range(2): t.update(off)
        self.assertEqual(t.count.tolist(),[[1,1]])
        self.assertAlmostEqual(float(t.last[0,0]),.22,places=5)
        self.assertEqual(float(t.penalty()[0]),0.)

    def test_uneven_completed_stances_and_staleness(self):
        t=CompletedStances(1,'cpu'); t.update(torch.tensor([[False,False]]))
        for i in range(25): t.update(torch.tensor([[i<10,i<20]]))
        self.assertGreater(float(t.penalty()[0]),0.)
        for _ in range(100): t.update(torch.tensor([[False,False]]))
        self.assertEqual(float(t.penalty()[0]),0.)
        t.reset(); self.assertEqual(int(t.count.sum()),0)

    def test_reset_subset(self):
        t=CompletedStances(2,'cpu'); t.count[:]=4
        t.reset(torch.tensor([True,False])); self.assertEqual(t.count.tolist(),[[0,0],[4,4]])

    def test_ankle_same_leg_and_mirror(self):
        torque=torch.zeros(1,2,2)
        # One pitch on each leg: no combining into a fictitious 12rad/s motor.
        speed=torch.tensor([[[6.,0.],[6.,0.]]])
        self.assertEqual(float(differential_cost(torque,speed)),0.)
        speed=torch.tensor([[[6.,6.],[0.,0.]]])
        self.assertGreater(float(differential_cost(torque,speed)),0.)
        self.assertAlmostEqual(float(differential_cost(torque,speed)),float(differential_cost(torque,speed.flip(1))))

    def test_missing_steps_fail_closed(self):
        values=dict(fall_fraction=0., evaluated_seconds=17., valid_step_replicas=24,
                    speed_mae=.02,lateral_speed=.01,slip=.03,torso_tilt_deg=5.,stance_asymmetry=.2,
                    foot_crossing=0.,knee_crossing=0.,nonfoot=0.)
        report=dict(seed=47,duration=20.,replicas=24,rows={})
        for speed in ('0.5','1.0'):
            for p in ('reference_','candidate_'): report['rows'][p+speed]=copy.deepcopy(values)
        report['rows']['candidate_target']=dict(values,target=1.8)
        self.assertTrue(judge([report],'index')['walking_pass'])
        report['rows']['candidate_1.0']['stance_asymmetry']=None
        self.assertFalse(judge([report],'index')['walking_pass'])


if __name__=='__main__': unittest.main(verbosity=2)
