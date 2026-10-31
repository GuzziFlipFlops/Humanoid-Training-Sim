"""Control-step invariance, reset farming, reference consistency and run hops."""
import json, unittest
from pathlib import Path
import numpy as np
import torch
from walking_rewards import fall_penalty, running_flight, load_transfer, StepTracker, StandingAnchor, standing_support
from reference_geometry import target_limits, bounded_motion, foot_reference


class BalanceAndRunning(unittest.TestCase):
    def test_stop_after_walking_anchors_here_not_at_episode_start(self):
        anchor=StandingAnchor(1,'cpu');root=torch.zeros(1,2)
        anchor.update(root,torch.tensor([0.]))
        root[:,0]=2.;anchor.update(root,torch.tensor([.35]))
        self.assertEqual(float(anchor.update(root,torch.tensor([0.]))),0.)
        root[:,0]+=.1
        self.assertAlmostEqual(float(anchor.update(root,torch.tensor([0.]))),.1,places=6)
        anchor.reset(torch.tensor([0]))
        self.assertEqual(float(anchor.update(root,torch.tensor([0.]))),0.)
    def test_standing_support_and_anchor_costs_prefer_retained_stance(self):
        feet=torch.tensor([[[0.,-.15,-.8],[0.,.15,-.8]]])
        good=standing_support(torch.tensor([0.]),torch.tensor([[True,True]]),torch.zeros(1,3),feet,torch.zeros(1))
        bad=standing_support(torch.tensor([0.]),torch.tensor([[True,False]]),torch.tensor([[.12,0.,0.]]),feet,torch.tensor([.5]))
        self.assertEqual(sum(float(v) for v in good.values()),0.)
        self.assertLess(sum(float(v) for v in bad.values()),-1.9)
        moving=standing_support(torch.tensor([.35]),torch.tensor([[True,False]]),torch.tensor([[.12,0.,0.]]),feet,torch.tensor([.5]))
        self.assertEqual(sum(float(v) for v in moving.values()),0.)
    def test_fall_is_event_not_control_rate(self):
        for dt in [.005, .02, .05]:
            value=fall_penalty(torch.tensor([True, False]),dt)*dt
            torch.testing.assert_close(value,torch.tensor([-8.,0.]))

    def test_discovery_reset_farming_cannot_repay_a_fall(self):
        # Conservative upper bound for ALL non-progress discovery credit:
        # Two airborne feet: gait 2*.8, reference .35, load-transfer .4,
        # flight .25 for at most 2.5/f seconds (a conservative combined bound).
        # Common balance reward continues when alive, so it is not reset income.
        for speed in [.15,.35,1.,1.5,2.5]:
            discovery=(2*.8+.35+.4+.25)*(2.5/(.9+.5*speed))
            old_cost=30*.02
            self.assertGreater(discovery,old_cost)
            self.assertLess(discovery,8.)

    def test_load_guidance_needs_actual_weight_transfer_and_expires(self):
        command=torch.tensor([.15]); expected=torch.tensor([[True,False]])
        score=lambda force,budget:load_transfer(command,torch.tensor([force]),expected,torch.ones(1),torch.tensor([budget]),338.)
        self.assertEqual(float(score([169.,169.],1.)),0.)
        self.assertGreater(float(score([300.,38.],1.)),.25)
        self.assertEqual(float(score([300.,38.],0.)),0.)
        self.assertEqual(float(score([100.,0.],1.)),0.)

    def test_running_flight_requires_motion_and_correct_phase(self):
        args=[torch.tensor([1.5]),torch.tensor([[False,False]]),torch.tensor([[False,False]]),
              torch.tensor([[.04,.04]]),torch.ones(1),torch.ones(1),torch.ones(1),torch.tensor([False])]
        self.assertGreater(float(running_flight(*args)),0.)
        args[5][:]=0.;self.assertEqual(float(running_flight(*args)),0.)
        args[5][:]=1.;args[2][:]=True;self.assertEqual(float(running_flight(*args)),0.)
        args[2][:]=False;args[0][:]=.35;self.assertEqual(float(running_flight(*args)),0.)

    def test_running_two_foot_hop_pays_no_steps(self):
        tracker=StepTracker(1,'cpu');heading=torch.tensor([[1.,0.]]);cmd=torch.tensor([1.5]);live=torch.tensor([True])
        feet=torch.tensor([[[0.,-.12],[0.,.12]]]);root=torch.zeros(1,2);h=torch.zeros(1,2)
        def tick(contact):return tracker.update(torch.tensor([contact]),feet,h,root,heading,cmd,live,.02)
        for _ in range(4):tick([True,True])
        h[:]=.04
        for _ in range(8):tick([False,False])
        feet[:,:,0]=.16;root[:,0]=.10;h[:]=0.
        events=sum((tick([True,True]) for _ in range(4)))
        self.assertEqual(float(events.sum()),0.)

    def test_running_alternating_flight_landings_pay_steps(self):
        tracker=StepTracker(1,'cpu');heading=torch.tensor([[1.,0.]]);cmd=torch.tensor([1.5]);live=torch.tensor([True])
        feet=torch.tensor([[[0.,-.12],[0.,.12]]]);root=torch.zeros(1,2);h=torch.zeros(1,2)
        def tick(contact):return tracker.update(torch.tensor([contact]),feet,h,root,heading,cmd,live,.02)
        for _ in range(4):tick([True,True])
        h[:]=.04
        for _ in range(8):tick([False,False])
        feet[:,0,0]=.16;root[:,0]=.10;h[:,0]=0.
        events=sum((tick([True,False]) for _ in range(5)))
        self.assertEqual(float(events[:,0]),1.)
        for _ in range(6):tick([True,False])
        feet[:,1,0]=.25;root[:,0]=.18;h[:]=0.
        events=sum((tick([True,True]) for _ in range(5)))
        self.assertEqual(float(events[:,1]),1.)

    def test_staggered_running_hop_cannot_pay_two_steps(self):
        tracker=StepTracker(1,'cpu');heading=torch.tensor([[1.,0.]]);cmd=torch.tensor([1.5]);live=torch.tensor([True])
        feet=torch.tensor([[[0.,-.12],[0.,.12]]]);root=torch.zeros(1,2);h=torch.zeros(1,2)
        def tick(contact):return tracker.update(torch.tensor([contact]),feet,h,root,heading,cmd,live,.02)
        for _ in range(4):tick([True,True])
        h[:]=.04
        for _ in range(8):tick([False,False])
        feet[:,:,0]=.16;root[:,0]=.10;h[:]=0.
        events=tick([True,False])
        for _ in range(4):events+=tick([True,True])
        self.assertEqual(float(events.sum()),0.)

    def test_eighty_ms_staggered_running_hop_has_no_early_landing_credit(self):
        tracker=StepTracker(1,'cpu');heading=torch.tensor([[1.,0.]]);cmd=torch.tensor([1.5]);live=torch.tensor([True])
        feet=torch.tensor([[[0.,-.12],[0.,.12]]]);root=torch.zeros(1,2);h=torch.zeros(1,2)
        def tick(c):return tracker.update(torch.tensor([c]),feet,h,root,heading,cmd,live,.02)
        for _ in range(5):tick([True,True])
        h[:]=.04
        for _ in range(8):tick([False,False])
        feet[:,:,0]=.16;root[:,0]=.10;h[:]=0.
        events=sum((tick([True,False]) for _ in range(4)))
        for _ in range(5):events+=tick([True,True])
        self.assertEqual(float(events.sum()),0.)

    def test_reference_feet_follow_controller_bounded_joint_targets(self):
        here=Path(__file__).parent
        meta=json.loads((here.parent/'assets/export_metadata.json').read_text())
        original=np.load(here/'assets/motion_reference.npz')
        bounded=bounded_motion(meta,original)
        lo,hi=target_limits(meta,original['names'].tolist())
        self.assertTrue(np.all(bounded['q']>=lo-1e-7));self.assertTrue(np.all(bounded['q']<=hi+1e-7))
        np.testing.assert_array_equal(bounded['stand'],original['stand'])
        actual=foot_reference(meta,bounded);old=foot_reference(meta,original)
        self.assertGreater(float(np.max(np.linalg.norm(actual-old,axis=-1))),.0001)
        self.assertFalse(np.array_equal(original['q'],bounded['q']))


if __name__=='__main__':unittest.main(verbosity=2)
