"""Replay the planted-foot portion of the saved failing walker on CPU.

The old recording lacks per-foot positions. Use only continuously both-loaded
samples, whose zero completed steps were independently measured. No invented
foot swing or landing is scored. This validates incentive removal, not learning.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from walking_rewards import walking_terms, REWARD_VERSION

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--telemetry', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.telemetry)
    recorded = np.load(source)
    motion = np.load(HERE / 'assets/motion_reference.npz')
    case = next(i for i, x in enumerate(recorded['case_speeds']) if abs(x - .35) < 1e-5)
    labels = recorded['case_labels'] == case
    stats = recorded['metrics'][::5, labels]
    q = recorded['q'][:, labels]
    use = (stats[:, :, 0] > .5) & (stats[:, :, 12] > .5) & (stats[:, :, 10] == 0) & (stats[:, :, 11] == 0)
    order = [motion['names'].tolist().index(n) for n in recorded['joint_names']]
    times = (101 + 5 * np.arange(len(q))) * float(recorded['step_dt'])
    phase = (times * 1.075) % 1.
    ix = phase * 120
    a = ix.astype(int) % 120
    b = (a + 1) % 120
    w = (ix - ix.astype(int))[:, None]
    target = ((1 - w) * motion['q'][1, a][:, order] + w * motion['q'][1, b][:, order])[:, None]
    expected = np.stack([phase < .62, (phase + .5) % 1. < .62], 1)[:, None]
    expected = np.broadcast_to(expected, (*use.shape, 2))
    health = (np.exp(-6 * stats[:, :, 6] ** 2) * np.exp(-5 * stats[:, :, 7] ** 2)
              * np.exp(-((stats[:, :, 1] - stats[:, :, 2]) / .06) ** 2))
    old_tracking = 2.5 * np.exp(-((stats[:, :, 3] - .35) ** 2 + stats[:, :, 4] ** 2) / .20) * health
    error = ((q - target) ** 2).mean(-1)
    old_reference = .70 * np.exp(-error / .10) * health
    old_gait = .55 * expected.mean(-1) * health
    legs = [i for i, n in enumerate(recorded['joint_names']) if any(x in n for x in ['_hip_', '_knee_', '_ankle_'])]
    leg_error = ((q[:, :, legs] - target[:, :, legs]) ** 2).mean(-1)
    count = int(use.sum())
    tensor = lambda x: torch.tensor(np.asarray(x)[use], dtype=torch.float32)
    velocity = torch.stack([tensor(stats[:, :, 3]), tensor(stats[:, :, 4]), torch.zeros(count)], 1)
    zeros_feet = torch.zeros((count, 2, 3))
    terms = walking_terms(torch.full((count,), .35), velocity, tensor(health), tensor(stats[:, :, 8]),
        torch.ones((count, 2), dtype=torch.bool), torch.tensor(expected[use], dtype=torch.bool),
        torch.zeros((count, 2)), zeros_feet, zeros_feet, tensor(leg_error), torch.zeros((count, 2)), .02)
    old = dict(track=float(old_tracking[use].mean()), reference=float(old_reference[use].mean()), gait=float(old_gait[use].mean()))
    new = {k: float(terms[k].mean()) for k in ['track', 'progress', 'reference', 'gait', 'missing_swing', 'completed_step']}
    old_total = sum(old.values())
    new_positive = float(sum(terms[k].clamp_min(0.) for k in ['track', 'progress', 'reference', 'gait', 'completed_step']).mean())
    report = dict(reward_version=REWARD_VERSION,source=str(source),samples=count,
        replay_scope='Recorded both-loaded/first-failure-free samples only; no missing foot-position data fabricated',
        old_positive_motion_rate=old_total,new_positive_motion_rate=new_positive,
        positive_motion_credit_reduction_fraction=1-new_positive/old_total,
        old_terms=old,new_terms=new,qualification='Incentive regression on recorded failure; NOT a walking promotion or convergence proof')
    assert new_positive < old_total * .20, report
    assert new['completed_step'] == 0. and new['gait'] == 0.
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
