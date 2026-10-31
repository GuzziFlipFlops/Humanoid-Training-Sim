"""Fast CPU tests: contact state, joint order math, and rejection gates."""
import unittest
import copy
import torch
from contact_state import CompletedStances, SmoothedStances, differential_cost
from gates import judge, stage_checks, proven_velocity_limited


class Tests(unittest.TestCase):
    def test_smoothed_cost_bounded_symmetric_and_stronger(self):
        old=CompletedStances(3,'cpu'); new=SmoothedStances(3,'cpu')
        for t in (old,new):
            t.last[:]=torch.tensor([[.15,.30],[.30,.15],[.2,.2]])
            t.age[:]=0.; t.count[:]=1
        costs=new.penalty()
        self.assertTrue(bool(((costs >= 0) & (costs <= 1)).all()))
        self.assertEqual(float(costs[0]),float(costs[1]))
        self.assertEqual(float(costs[2]),0.)
        self.assertGreater(float(.6*costs[0]),float(.2*old.penalty()[0])*5)

    def test_two_step_average_not_latest_pair(self):
        t=SmoothedStances(1,'cpu')
        t.count[:]=2; t.age[:]=0.
        t.last[:]=torch.tensor([[.1,.3]]); t.previous[:]=torch.tensor([[.3,.1]])
        self.assertEqual(float(t.penalty()[0]),0.)
        t.reset()
        self.assertEqual(float(t.previous.sum()),0.)
        self.assertEqual(float(t.penalty()[0]),0.)

    def test_smoothed_real_contact_history_and_reset(self):
        t=SmoothedStances(2,'cpu')
        for i in range(60):
            c=torch.tensor([[i%20<8,i%20<14]]).expand(2,-1)
            t.update(c)
        self.assertTrue(bool((t.previous>0).all()))
        cost=t.penalty().clone()
        t.reset(torch.tensor([True,False]))
        self.assertEqual(float(t.penalty()[0]),0.)
        self.assertEqual(float(t.penalty()[1]),float(cost[1]))
        for _ in range(100): t.update(torch.zeros(2,2,dtype=torch.bool))
        self.assertEqual(float(t.penalty().sum()),0.)

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

    def test_false_four_mps_success_rejected(self):
        row=dict(target=4.,speed_mae=1.,fall_fraction=0.,evaluated_seconds=17.,valid_step_replicas=24,
                 slip=.1,nonfoot=0.,foot_crossing=0.,knee_crossing=0.,stance_asymmetry=.2,
                 target_oob=0.,waist_oob=0.,torso_tilt_deg=10.,torso_excess_fraction=0.,
                 filtered_flight_fraction=.1)
        report=dict(rows={'candidate_target':row},replicas=24,duration=20.)
        self.assertFalse(stage_checks(report,'speed')['target_speed'])
        row['speed_mae']=.1
        self.assertTrue(all(stage_checks(report,'speed').values()))
        row['fall_fraction']=.25
        self.assertFalse(stage_checks(report,'speed')['falls'])

    def test_velocity_boost_requires_stable_repeated_hard_evidence(self):
        row=dict(target=6.,mean_speed=5.0,fall_fraction=0.,
                 joint_velocity_saturation={'left_knee_pitch':.12,'right_knee_pitch':.03})
        reports=[{'rows':{'candidate_target':copy.deepcopy(row)}} for _ in range(2)]
        self.assertEqual(proven_velocity_limited(reports),['knee'])
        reports[1]['rows']['candidate_target']['fall_fraction']=.10
        self.assertEqual(proven_velocity_limited(reports),[])
        reports[1]['rows']['candidate_target']['fall_fraction']=0.
        reports[1]['rows']['candidate_target']['mean_speed']=5.8
        self.assertEqual(proven_velocity_limited(reports),[])


if __name__=='__main__': unittest.main(verbosity=2)
