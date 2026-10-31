"""Behavioral regression tests for the observed stationary walking loophole."""
import unittest
import numpy as np
import torch
from pathlib import Path
import json
import ast
from walking_rewards import StepTracker, walking_terms
from reference_geometry import foot_reference


class Steps(unittest.TestCase):
    def setUp(self):
        self.tracker = StepTracker(1, 'cpu')
        self.feet = torch.tensor([[[0., -.15], [0., .15]]])
        self.root = torch.zeros((1, 2))
        self.heading = torch.tensor([[1., 0.]])
        self.command = torch.tensor([.35])
        self.total = 0.

    def tick(self, contacts=(True, True), clearance=(0., 0.), eligible=True):
        events = self.tracker.update(torch.tensor([contacts]), self.feet,
            torch.tensor([clearance]), self.root, self.heading, self.command,
            torch.tensor([eligible]), .02)
        self.total += float(events.sum())
        return events

    def settle(self):
        for _ in range(5):
            self.tick()

    def swing(self, side=0, distance=.06, root_distance=.01, height=.035, air_ticks=6):
        contacts = [True, True]
        contacts[side] = False
        clear = [0., 0.]
        clear[side] = height
        self.tick(contacts, clear)
        self.feet[:, side] += self.heading * distance
        self.root += self.heading * root_distance
        for _ in range(air_ticks - 1):
            self.tick(contacts, clear)
        for _ in range(5):
            self.tick()

    def test_real_alternating_steps(self):
        self.settle()
        self.swing(0)
        for _ in range(10):self.tick()
        self.swing(1)
        for _ in range(10):self.tick()
        self.swing(0)
        self.assertEqual(self.total, 3.)

    def test_excessively_fast_alternating_contacts_do_not_farm_events(self):
        self.settle()
        for side in [0,1]*10:self.swing(side)
        # Physical contacts are faster than the permitted credit cadence.
        self.assertLess(self.total, 20.)

    def test_sliding_during_landing_dwell_cannot_create_a_step(self):
        self.settle()
        self.tick((False,True),(.04,0.))
        for _ in range(5):self.tick((False,True),(.04,0.))
        self.tick()
        self.feet[:,0]+=self.heading*.08;self.root+=self.heading*.02
        self.settle()
        self.assertEqual(self.total,0.)

    def test_stationary_knee_bobbing_has_no_step(self):
        for _ in range(150):
            self.tick()
        self.assertEqual(self.total, 0.)

    def test_planted_sliding_has_no_step(self):
        for _ in range(150):
            self.root += .01
            self.feet += .01
            self.tick()
        self.assertEqual(self.total, 0.)

    def test_in_place_kicking_has_no_step(self):
        self.settle()
        self.swing(distance=0.)
        self.assertEqual(self.total, 0.)

    def test_backward_foot_placement_has_no_step(self):
        self.settle()
        self.swing(distance=-.06)
        self.assertEqual(self.total, 0.)

    def test_foot_step_without_body_progress_rejected(self):
        self.settle()
        self.swing(root_distance=0.)
        self.assertEqual(self.total, 0.)

    def test_uncleared_sole_rejected(self):
        self.settle()
        self.swing(height=.003)
        self.assertEqual(self.total, 0.)

    def test_contact_chatter_rejected(self):
        self.settle()
        for _ in range(25):
            self.swing(air_ticks=1)
        self.assertEqual(self.total, 0.)

    def test_repeated_same_foot_does_not_farm_credit(self):
        self.settle()
        self.swing(0)
        self.swing(0)
        self.swing(0)
        self.assertEqual(self.total, 1.)

    def test_landing_cannot_be_credited_later_by_sliding(self):
        self.settle()
        self.swing(distance=0., root_distance=0.)
        self.feet += self.heading[:, None] * .10
        self.root += self.heading * .05
        self.settle()
        self.assertEqual(self.total, 0.)

    def test_yaw_rotated_step_uses_world_heading(self):
        self.heading = torch.tensor([[-1., 0.]])
        self.settle()
        self.swing(0)
        self.assertEqual(self.total, 1.)

    def test_simultaneous_walking_hop_not_two_steps(self):
        self.settle()
        self.tick((False, False), (.04, .04))
        self.feet += self.heading[:, None] * .08
        self.root += self.heading * .02
        for _ in range(5):
            self.tick((False, False), (.04, .04))
        self.settle()
        self.assertEqual(self.total, 0.)

    def test_reset_during_swing_erases_event(self):
        self.settle()
        for _ in range(6):
            self.tick((False, True), (.04, 0.))
        self.tracker.reset(torch.tensor([0]))
        self.feet += .08
        self.root += .04
        self.settle()
        self.assertEqual(self.total, 0.)

    def test_stopped_and_ineligible_trials_get_no_event(self):
        self.command[:] = 0.
        self.settle()
        self.swing()
        self.assertEqual(self.total, 0.)
        self.command[:] = .35
        for _ in range(20):
            self.tick((False, True), (.04, 0.), eligible=False)
        self.settle()
        self.assertEqual(self.total, 0.)


class Rewards(unittest.TestCase):
    def terms(self, vx=0., slip=0., healthy=1., contact=(True, True), clearance=(0., 0.), events=(0., 0.), command=.35, dt=.02):
        feet = torch.tensor([[[0., -.15, -.8], [0., .15, -.8]]])
        return walking_terms(torch.tensor([command]), torch.tensor([[vx, 0., 0.]]),
            torch.tensor([healthy]), torch.tensor([slip]), torch.tensor([contact]),
            torch.tensor([[True, False]]), torch.tensor([clearance]), feet, feet,
            torch.tensor([0.]), torch.tensor([events]), dt)

    def test_perfect_joint_reference_but_planted_feet_has_no_locomotion_bonus(self):
        result = self.terms()
        for key in ['track', 'progress', 'reference', 'gait', 'completed_step']:
            self.assertEqual(float(result[key]), 0., key)
        self.assertLess(float(result['missing_swing']), 0.)

    def test_true_walking_outranks_stationary(self):
        good = self.terms(vx=.35, contact=(True, False), clearance=(0., .035), events=(1., 0.))
        rate = sum(float(good[k]) for k in ['track', 'progress', 'reference', 'gait'])
        self.assertGreater(rate, 4.)

    def test_loaded_foot_slide_cannot_collect_velocity_reward(self):
        result = self.terms(vx=.35, slip=.5)
        self.assertLess(float(result['track']), 1e-12)
        self.assertLess(float(result['progress']), 1e-12)

    def test_backwards_motion_has_negative_progress(self):
        result = self.terms(vx=-.35)
        self.assertEqual(float(result['track']), 0.)
        self.assertLess(float(result['progress']), 0.)

    def test_fall_has_no_positive_motion_credit(self):
        result = self.terms(vx=.35, healthy=0., contact=(False, False), clearance=(.2, .2), events=(1., 1.))
        for key in ['track', 'progress', 'reference', 'gait', 'completed_step']:
            self.assertEqual(float(result[key]), 0.)

    def test_walking_flight_cannot_earn_positive_progress_or_swing(self):
        result=self.terms(vx=.35,contact=(False,False),clearance=(.05,.05))
        for key in ['track','progress','reference','gait','flight']:
            self.assertEqual(float(result[key]),0.,key)
        self.assertEqual(float(result['walking_flight']),-1.)

    def test_running_flight_is_not_penalized_as_walking(self):
        self.assertEqual(float(self.terms(vx=1.5,command=1.5,contact=(False,False))['walking_flight']),0.)

    def test_step_bonus_integrated_exactly_once_independent_of_dt(self):
        for dt in [.01, .02, .04]:
            self.assertAlmostEqual(float(self.terms(events=(1., 0.), dt=dt)['completed_step']) * dt, .30, places=6)

    def test_stop_command_has_no_walking_terms(self):
        result = self.terms(command=0.)
        for key in ['track', 'progress', 'reference', 'gait', 'missing_swing', 'completed_step']:
            self.assertEqual(float(result[key]), 0.)
        self.assertEqual(float(result['posture_scale']), 1.)

    def test_in_place_swing_is_guidance_not_completed_walking(self):
        result = self.terms(contact=(True, False), clearance=(0., .035))
        self.assertGreater(float(result['gait']), 0.)
        self.assertEqual(float(result['completed_step']), 0.)
        self.assertEqual(float(result['track']), 0.)


class ReferenceGeometry(unittest.TestCase):
    def test_vectorized_foot_table_matches_original_cad_fk(self):
        from build_motion import fk
        here = Path(__file__).parent
        meta = json.loads((here.parent / 'assets/export_metadata.json').read_text())
        motion = np.load(here / 'assets/motion_reference.npz')
        table = foot_reference(meta, motion)
        for clip, phase in [(0, 0), (1, 32), (3, 117), (6, 65)]:
            for side, name in enumerate(['left_foot', 'right_foot']):
                expected, _ = fk(motion['q'][clip, phase], name)
                np.testing.assert_allclose(table[clip, phase, side], expected, atol=1e-6)


class InputMigration(unittest.TestCase):
    def setUp(self):
        self.actor = {'obs_normalizer._mean': torch.zeros((1, 320)),
                      'obs_normalizer._var': torch.zeros((1, 320)),
                      'obs_normalizer._std': torch.zeros((1, 320)),
                      'obs_normalizer.count': torch.tensor(112000000),
                      'mlp.0.weight': torch.randn(10, 320)}

    def test_command_and_phase_scales_are_bounded(self):
        from repair_actor_normalization import repair_state, known_input_ids
        fixed = repair_state(self.actor)
        ids = known_input_ids()
        self.assertTrue(torch.equal(fixed['obs_normalizer._std'][:, ids], torch.ones((1, 20))))
        self.assertTrue(torch.equal(self.actor['obs_normalizer._std'], torch.zeros((1, 320))))

    def test_physical_inputs_count_and_weights_preserved(self):
        from repair_actor_normalization import repair_state, known_input_ids
        fixed = repair_state(self.actor)
        keep = torch.ones(320, dtype=torch.bool)
        keep[known_input_ids()] = False
        self.assertTrue(torch.equal(fixed['obs_normalizer._std'][:, keep], self.actor['obs_normalizer._std'][:, keep]))
        for key in ['obs_normalizer.count', 'mlp.0.weight']:
            self.assertTrue(torch.equal(fixed[key], self.actor[key]))


class CapacityFallback(unittest.TestCase):
    def select(self,counts,requested):
        source = Path(__file__).with_name('supervisor.py')
        tree = ast.parse(source.read_text())
        function = next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name == 'select_environment_count')
        assignment = next(x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ENV_FALLBACK' for t in x.targets))
        scope = {}
        exec(compile(ast.Module(body=[assignment,function],type_ignores=[]),str(source),'exec'),scope)
        eligible = [dict(environments=n,environment_steps_per_second=n) for n in counts]
        return scope['select_environment_count'](eligible,requested)

    def test_failed_requested_benchmark_falls_back(self):
        self.assertEqual(self.select([4096,6144],8192),6144)

    def test_eligible_requested_count_is_preferred(self):
        self.assertEqual(self.select([4096,6144,8192],8192),8192)

    def test_no_safe_count_is_not_ignored(self):
        with self.assertRaises(AssertionError):
            self.select([],8192)

    def test_fusion_memory_can_select_a_smaller_guarded_count(self):
        self.assertEqual(self.select([4096,6144],3072),3072)
        self.assertEqual(self.select([4096],2048),2048)

    def test_arbitrary_unbenchmarked_count_is_rejected(self):
        with self.assertRaises(AssertionError):
            self.select([4096],3000)


if __name__ == '__main__':
    unittest.main(verbosity=2)
