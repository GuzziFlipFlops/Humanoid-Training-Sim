"""Independent rollout evaluator regression tests with synthetic pre-reset frames."""
from dataclasses import fields
import unittest
import numpy as np
from evaluation import audit
from test_reward_core import standing


def trace():
    f = standing()
    result = {k.name: np.repeat(getattr(f, k.name).numpy()[None], 1250, axis=0) for k in fields(f)}
    result.update(terminated=np.zeros((1250, 2), dtype=bool), penetration=np.zeros((1250, 2)),
                  joint_pos=np.zeros((1250, 2, 22)), joint_vel=np.zeros((1250, 2, 22)),
                  torque_estimate=np.zeros((1250, 2, 22)))
    return result


class EvaluationTests(unittest.TestCase):
    def test_stable_whole_trial(self):
        result = audit(trace())
        self.assertEqual(result['strict_successes'], 2)
        self.assertEqual(result['median_completion_s'], 7.)

    def test_success_then_walkoff_not_hidden(self):
        data = trace()
        data['planar_speed'][500:, 0] = .2
        result = audit(data)
        self.assertEqual(result['strict_successes'], 1)
        self.assertEqual(result['rise_then_lost_stand'], 1)

    def test_failure_before_reset_stays_failed(self):
        data = trace()
        data['terminated'][10, 0] = True
        self.assertEqual(audit(data)['strict_successes'], 1)

    def test_not_seven_seconds(self):
        data = {k: v[:349] for k, v in trace().items()}
        self.assertEqual(audit(data)['strict_successes'], 0)

    def test_contact_cheat_rejected(self):
        data = trace()
        data['nonfoot_force'][:, 0] = 20.
        self.assertEqual(audit(data)['strict_successes'], 1)

    def test_slow_slide_not_hidden(self):
        data = trace()
        data['xy'][:, :, 0] = np.arange(1250)[:, None] * .001
        self.assertEqual(audit(data)['strict_successes'], 0)


if __name__ == '__main__': unittest.main(verbosity=2)
