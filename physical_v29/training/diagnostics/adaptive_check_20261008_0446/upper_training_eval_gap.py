"""Compare observed minibatch mean labels with closed-loop evaluation outputs."""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
review = json.loads((OUT / 'matched_shoulder_review.json').read_text())
rows = []
commands = [.15, .35, .65, 1., 1.5, 2., 2.5, 3., 3.5]
for row in review['rows']:
    with np.load(Path(row['nominal']) / 'gait_diagnostics_50hz.npz') as z:
        raw, time, speed, done = z['raw_action'], z['time'], z['commands'], z['done']
        names = z['joint_names'].tolist()
    ids = [names.index(n) for n in ['left_shoulder_pitch', 'right_shoulder_pitch']]
    minibatch = row['last_actual_minibatch']
    for group, command in enumerate(commands):
        group = 9 if command == 3.5 else group
        robots = np.flatnonzero(np.isclose(speed, command))
        end = len(time)
        terminal = np.flatnonzero(done[:, robots].any(1))
        if len(terminal):
            end = int(terminal[0])
        window = (time[:end] >= 5) & (time[:end] <= 17)
        observed = np.clip(raw[:end][window][:, robots][:, :, ids], -3., 3.) * .25 * 180 / np.pi
        rows.append(dict(branch=row['branch'], updates=row['updates'], command=command,
            training_group='[3.25,3.5]' if command == 3.5 else str(command),
            training_phase_balanced_mean_deg=minibatch['phase_balanced_mean_deg'][group],
            training_phase_counts=minibatch['counts'][group],
            evaluation_time_mean_effective_offset_deg=observed.mean((0, 1)).tolist(),
            representative_robots=len(robots)))
result = dict(rows=rows,
    scope='Actual last logged PPO minibatch phase-balanced outputs versus final deterministic nominal representative closed-loop raw outputs after the existing clamp. Both use native0.25rad/action offsets. These are different states and sampling/optimization times;the frontier training group is a speed band,not exact3.5.',
    conclusion='A small population/minibatch objective does not establish small individual closed-loop held bias. This comparison identifies a distribution/feature-response hypothesis,not a reproduced production bug or a causal attribution. Per-state parameter and physical coupling tests remain necessary.',
    production_change=False)
(OUT / 'upper_training_eval_gap.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps([r for r in rows if r['command'] in [2.5,3.5]], indent=2))
