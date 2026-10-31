"""CPU-only behavioral gate checks; safe while Isaac owns the GPU.

Extract the pure gate function without importing Isaac or starting a second
simulator. These cases guard against observed old reward/evaluation exploits.
"""
import ast
from pathlib import Path
import unittest
from heading_control import direction_checks


def load_gate(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'gate_case')
    scope = {'direction_checks':direction_checks}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['gate_case']


GATE = load_gate(Path(__file__).with_name('evaluation.py'))


def good_stats(speed=0.):
    return dict(survival=1., height_error_m=.012, tilt_deg=4., torso_tilt_deg=5.,
                nonfoot_fraction=0., invalid_samples=0, crossing_fraction=0.,
                slip_m_s=.025, mean_speed_m_s=speed, mean_final_drift_m=.03,
                two_feet_fraction=1. if speed == 0 else .25,
                support_outside_fraction=0., forward_speed_m_s=speed,
                lateral_speed_m_s=.015, steps_left=24., steps_right=24.,
                step_asymmetry=0., stance_asymmetry=.03, cycle_hz=.9+.5*speed,
                swing_clearance_m=.045, flight_fraction=.10 if speed>1. else .005,
                hop_landing_fraction=0.,same_foot_fraction=0.,heading_outside_fraction=0.,
                reset_frame_forward_speed_m_s=speed,reset_frame_lateral_speed_m_s=.015,
                forward_path_efficiency=.99)


class Gates(unittest.TestCase):
    def test_nominal_positive_cases(self):
        for speed in [0., .35, .65, 1., 1.5, 2., 2.5]:
            self.assertTrue(GATE(speed, good_stats(speed))['passed'])

    def rejects(self, speed, key, value, failure, **more):
        stats = good_stats(speed)
        stats.update({key: value}, **more)
        result = GATE(speed, stats)
        self.assertFalse(result['passed'])
        self.assertIn(failure, result['failures'])

    def test_rise_then_fall_is_not_standing(self):
        self.rejects(0., 'survival', .0, 'survival', height_error_m=.0)

    def test_good_pelvis_does_not_hide_bad_torso(self):
        self.rejects(1., 'torso_tilt_deg', 32., 'torso_tilt', tilt_deg=3.)

    def test_moving_policy_cannot_pass_stand(self):
        self.rejects(0., 'mean_speed_m_s', .5, 'stationary')

    def test_foot_force_required_for_stand(self):
        self.rejects(0., 'two_feet_fraction', .3, 'both_loaded_feet')

    def test_com_outside_support_rejected(self):
        self.rejects(0., 'support_outside_fraction', .4, 'com_over_support')

    def test_sliding_cannot_pass_walking(self):
        self.rejects(1., 'slip_m_s', .4, 'foot_slip')

    def test_contact_chatter_without_steps_cannot_pass(self):
        self.rejects(1., 'steps_left', 1., 'completed_left_steps', steps_right=1.)

    def test_one_leg_hopping_not_bilateral_gait(self):
        self.rejects(1., 'step_asymmetry', 1.8, 'bilateral_steps', steps_right=0.)

    def test_running_requires_flight(self):
        self.rejects(2., 'flight_fraction', 0., 'flight')

    def test_walking_cannot_pass_with_frequent_flight(self):
        self.rejects(.35,'flight_fraction',.10,'walking_support')

    def test_balanced_counts_do_not_hide_same_foot_repetition(self):
        self.rejects(1.5,'same_foot_fraction',.4,'alternating_landings')

    def test_invalid_combined_transmission_rejected(self):
        self.rejects(1., 'invalid_samples', 1, 'no_invalid_differential')

    def test_knee_hand_or_torso_support_rejected(self):
        self.rejects(0., 'nonfoot_fraction', .2, 'nonfoot_contact')

    def test_nan_metrics_do_not_pass(self):
        self.rejects(1., 'forward_speed_m_s', float('nan'), 'speed')

    def test_push_test_still_rejects_frequent_falls(self):
        stats = good_stats(1.)
        stats['survival'] = .5
        self.assertFalse(GATE(1., stats, perturb=True)['passed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
