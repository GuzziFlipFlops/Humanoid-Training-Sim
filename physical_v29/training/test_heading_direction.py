"""Adversarial direction, command timing and compatible actor-transfer checks."""
import ast
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np
import torch
from heading_control import (yaw_from_quat,heading_yaw_command,direction_statistics,
                             direction_checks,COMMAND_VERSION)
from reward_warm_start import validate_actor_transfer
from unitree_reward_port import gait_parameters
from test_gates import GATE,good_stats

HERE=Path(__file__).parent


def quaternion(yaw):
    yaw=np.asarray(yaw)
    return np.stack([np.cos(yaw/2),np.zeros_like(yaw),np.zeros_like(yaw),np.sin(yaw/2)],-1)


def trajectory(speed=1.,yaw=None,lateral=0.,reference=0.):
    t=np.arange(181)*.1
    yaw=np.zeros_like(t) if yaw is None else np.asarray(yaw)
    local=np.stack([speed*t,lateral*t],-1)
    r=np.array([[np.cos(reference),-np.sin(reference)],[np.sin(reference),np.cos(reference)]])
    pos=(local@r.T)[:,None]
    q=quaternion(yaw+reference)[:,None]
    return direction_statistics(pos,q,np.ones(pos.shape[:2],dtype=bool),reference,.1,speed)


class Direction(unittest.TestCase):
    def test_straight_gait_passes_in_any_reset_frame(self):
        for speed in (.15,.35,.65,1.,1.5,2.,2.5):
            for reset in (0.,1.3,-2.7):
                stats=trajectory(speed,reference=reset)
                self.assertTrue(all(direction_checks(speed,stats).values()))
                self.assertAlmostEqual(stats['reset_frame_forward_speed_m_s'],speed)

    def test_moderate_yaw_oscillation_and_stepping_sway_pass(self):
        t=np.arange(181)*.1
        pos=np.stack([t,.005*np.sin(2*math.pi*1.25*t)],-1)[:,None]
        q=quaternion(.05*np.sin(2*math.pi*1.25*t))[:,None]
        stats=direction_statistics(pos,q,np.ones((181,1),bool),0.,.1,1.)
        self.assertTrue(all(direction_checks(1.,stats).values()))

    def test_body_forward_circle_cannot_pass(self):
        t=np.arange(181)*.1;yaw=t*2*math.pi/18
        v=np.stack([np.cos(yaw),np.sin(yaw)],-1)
        p=np.cumsum(v*.1,0)[:,None]
        stats=direction_statistics(p,quaternion(yaw)[:,None],np.ones((181,1),bool),0.,.1,1.)
        old=good_stats(1.);self.assertTrue(GATE(1.,old)['passed'])
        old.update(stats);result=GATE(1.,old)
        self.assertFalse(result['passed'])
        self.assertIn('forward_net_travel',result['failures'])
        self.assertIn('fixed_heading',result['failures'])

    def test_drift_backward_or_stationary_steps_rejected(self):
        for stat,failure in [(trajectory(1.,lateral=.3),'world_lateral_speed'),
                             (trajectory(-1.),'world_forward_speed'),
                             (trajectory(0.),'forward_net_travel')]:
            good=good_stats(1.);good.update(stat)
            self.assertIn(failure,GATE(1.,good)['failures'])

    def test_turning_on_a_straight_path_still_rejected(self):
        stats=trajectory(1.,yaw=np.linspace(0,math.pi/2,181))
        self.assertFalse(direction_checks(1.,stats)['fixed_heading'])

    def test_reset_teleport_does_not_count_as_forward_travel(self):
        p=np.zeros((181,1,2));p[50:,:,0]=1000.
        alive=np.ones((181,1),bool);alive[50:]=False
        stats=direction_statistics(p,quaternion(np.zeros(181))[:,None],alive,0.,.1,1.)
        self.assertEqual(stats['reset_frame_forward_speed_m_s'],0.)
        self.assertEqual(stats['direction_surviving_trials'],0)
        self.assertFalse(direction_checks(1.,stats)['world_forward_speed'])

    def test_empty_live_measurements_fail(self):
        stats=direction_statistics(np.zeros((2,1,2)),quaternion(np.zeros(2))[:,None],
                                   np.zeros((2,1),bool),0.,.1,1.)
        self.assertFalse(all(direction_checks(1.,stats).values()))

    def test_heading_sign_wrap_rate_bound_and_stopped_zero(self):
        angles=torch.tensor([.3,-.3,math.pi-.01,.8],dtype=torch.float64)
        q=torch.tensor(quaternion(angles.numpy()))
        target=torch.tensor([0.,0.,-math.pi+.01,0.],dtype=torch.float64)
        cmd=heading_yaw_command(q,target,torch.tensor([True,True,True,False]))
        torch.testing.assert_close(cmd,torch.tensor([-.15,.15,.01,0.],dtype=torch.float64))
        c=heading_yaw_command(q,torch.zeros(4),torch.ones(4,dtype=torch.bool))
        self.assertLessEqual(float(c.abs().max()),.2)

    def test_observed_command_held_until_action_finishes(self):
        tree=ast.parse((HERE/'environment_unitree.py').read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='LocomotionEnv')
        names=['_update_observation_command','_pre_physics_step']
        selected=copy.deepcopy(cls);selected.bases=[ast.Name(id='PhysicalEnv',ctx=ast.Load())]
        selected.body=[n for n in selected.body if isinstance(n,ast.FunctionDef) and n.name in names]
        class Physical:
            def _pre_physics_step(self,actions):self.received_command=self.commands.clone()
        scope=dict(PhysicalEnv=Physical,torch=torch,gait_parameters=gait_parameters,
                   yaw_from_quat=yaw_from_quat,heading_yaw_command=heading_yaw_command)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[selected],type_ignores=[])),str(HERE/'environment_unitree.py'),'exec'),scope)
        env=scope['LocomotionEnv'].__new__(scope['LocomotionEnv'])
        env.commands=torch.tensor([[1.,0.,0.],[0.,0.,0.]])
        env.robot=NS(data=NS(root_quat_w=torch.tensor(quaternion([.1,-.2]),dtype=torch.float32)))
        env.heading_target=torch.zeros(2);env.cfg=NS(evaluation=False)
        env._command_tick=-1;env.common_step_counter=4;env.command_ticks=torch.tensor([20,20])
        env.episode_length_buf=torch.zeros(2);env.step_dt=.02
        env._sample_commands=lambda ids:None
        env._update_observation_command();seen=env.commands.clone()
        env.robot.data.root_quat_w=torch.tensor(quaternion([.3,-.4]),dtype=torch.float32)
        env._pre_physics_step(torch.zeros(2,23))
        torch.testing.assert_close(env.action_commands,seen)
        torch.testing.assert_close(env.received_command,seen)
        env._update_observation_command();torch.testing.assert_close(env.commands,seen)
        self.assertEqual(env.command_ticks.tolist(),[19,19])
        env.common_step_counter+=1;env._update_observation_command()
        self.assertAlmostEqual(float(env.commands[0,2]),-.15,places=6)
        self.assertEqual(float(env.commands[1,2]),0.)
        # Exercise the actual parent's resampling conditional as well.
        parent=ast.parse((HERE/'environment.py').read_text())
        branch=next(n for n in ast.walk(parent) if isinstance(n,ast.If)
                    and 'defer_command_resampling' in ast.unparse(n.test))
        env.defer_command_resampling=True;before=env.command_ticks.clone()
        exec(compile(ast.fix_missing_locations(ast.Module(body=[branch],type_ignores=[])),
                     str(HERE/'environment.py'),'exec'),dict(self=env,torch=torch))
        torch.testing.assert_close(env.command_ticks,before)

    def test_actual_v4_actor_transfer_preserves_exploration(self):
        path=HERE/'runs/locomotion_20261003_r25_owned_observations/155_run_150_train_003_part_06/candidate.pt'
        saved=torch.load(path,map_location='cpu',weights_only=False)
        protocol=validate_actor_transfer(saved,HERE)
        self.assertTrue(protocol['preserve_trained_exploration'])
        self.assertFalse(protocol['qualification_reused'])
        bad=copy.deepcopy(saved);bad['infos']['actuator_sha256']='different'
        with self.assertRaises(AssertionError):validate_actor_transfer(bad,HERE)
        bad=copy.deepcopy(saved);bad['infos']['contract']='old_22_axis'
        with self.assertRaises(AssertionError):validate_actor_transfer(bad,HERE)
        bad=copy.deepcopy(saved);bad['infos']['reward_signature']='unaudited'
        with self.assertRaises(AssertionError):validate_actor_transfer(bad,HERE)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
