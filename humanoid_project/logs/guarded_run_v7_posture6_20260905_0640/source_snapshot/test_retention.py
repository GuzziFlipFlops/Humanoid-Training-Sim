import math
from pathlib import Path
import json
import unittest
import torch
from retention import high_speed_torso_cost, slow_torso_cost
from gates import judge

PRIOR=Path(__file__).resolve().parents[2]/'logs/guarded_run_v3_20260902_2244'

class Tests(unittest.TestCase):
    def test_high_speed_torso_term_is_exactly_zero_through_walking(self):
        tilt=torch.tensor([0.,.2,1.])
        speed=torch.tensor([0.,.5,1.])
        torch.testing.assert_close(high_speed_torso_cost(tilt,speed),torch.zeros(3))

    def test_high_speed_torso_cone_is_finite_and_direction_independent(self):
        tilt=torch.tensor([12.,12.5,13.5,180.])*math.pi/180.
        positive=high_speed_torso_cost(tilt,torch.full((4,),4.5))
        negative=high_speed_torso_cost(tilt,torch.full((4,),-4.5))
        self.assertEqual(positive[:2].tolist(),[0.,0.])
        self.assertTrue(bool((positive[2:]>0).all()))
        self.assertTrue(bool(torch.isfinite(positive).all()))
        torch.testing.assert_close(positive,negative)

    def test_approved_walk_inside_cone(self):
        tilt=torch.tensor([6.63,4.53])*math.pi/180.
        self.assertEqual(slow_torso_cost(tilt,torch.tensor([.5,1.])).tolist(),[0.,0.])
    def test_regression_penalized(self):
        t=torch.tensor([8.50,9.66,10.63])*math.pi/180.
        c=slow_torso_cost(t,torch.full((3,),.5))
        self.assertTrue(bool((c[1:]>c[:-1]).all()))
        self.assertTrue(bool((c>0).all()))
    def test_not_fixed_vertical_or_running_constraint(self):
        c=slow_torso_cost(torch.tensor([.1,.6,.6]),torch.tensor([1.,1.8,4.]))
        self.assertEqual(c.tolist(),[0.,0.,0.])
    def test_bounded_and_direction_independent(self):
        tilt=torch.linspace(0,math.pi,100)
        c=slow_torso_cost(tilt,torch.full((100,),.5))
        self.assertTrue(bool(torch.isfinite(c).all()))
        self.assertLessEqual(float(c.max()),1.9)
        torch.testing.assert_close(c,slow_torso_cost(tilt,torch.full((100,),-.5)))
    def test_old_transition_is_not_mistaken_for_upright_pass(self):
        reports=[json.loads((PRIOR/f'01_transition_1.8_0200_eval_{s}'/'evaluation.json').read_text()) for s in (47,83)]
        result=judge(reports,'torso')
        self.assertTrue(result['walking_pass'])
        self.assertFalse(result['stage_pass'])
        self.assertTrue(all(not batch['stage']['torso_mean_tilt'] for batch in result['checks']))
    def test_rejected_checkpoint_not_promoted(self):
        reports=[json.loads((PRIOR/f'02_targets_1.8_0400_eval_{s}'/'evaluation.json').read_text()) for s in (47,83)]
        self.assertFalse(judge(reports,'targets')['walking_pass'])

if __name__=='__main__': unittest.main(verbosity=2)
