"""CPU analysis of truthful physical coupling/labels; no production edits."""
import json, sys
from pathlib import Path
import numpy as np
OUT = Path(__file__).resolve().parent
folder = OUT / 'arm_target_response'
r = json.loads((folder / 'result.json').read_text())
with np.load(folder / 'telemetry_50hz.npz') as z:d={k:z[k] for k in z.files}
with np.load(folder / 'teacher_targets_10hz.npz') as z:l={k:z[k] for k in z.files}
names = d['joint_names'].tolist(); per = int(d['robots_per_group'])
lower = l['lower_indices']; upper = [names.index(x) for x in ['head_yaw', 'head_pitch', 'left_shoulder_pitch',
    'right_shoulder_pitch', 'left_shoulder_roll', 'right_shoulder_roll', 'left_elbow_pitch', 'right_elbow_pitch']]
shoulder = [names.index('left_shoulder_pitch'), names.index('right_shoulder_pitch')]
groups = r['groups']; index = {(g['seed'], g['initial_command'], g['mode']): i for i, g in enumerate(groups)}
t = d['time']; pre = (t >= 2) & (t < 5); post = t >= 6
rows = []; maximum_replay_difference = 0.
for i, g in enumerate(groups):
    if g['mode'] in ['control', 'transition']: continue
    j = index[(g['seed'], g['initial_command'], 'control')]
    ids = slice(i * per, (i + 1) * per); control = slice(j * per, (j + 1) * per)
    good = d['alive'][:, ids] & d['alive'][:, control]
    window = post[:, None] & good
    def paired_metric(key, transform=lambda x: x):
        a, b = transform(d[key][:, ids]), transform(d[key][:, control])
        delta = a - b
        robot = np.array([delta[window[:, k], k].mean() if window[:, k].any() else np.nan for k in range(per)])
        finite=robot[np.isfinite(robot)]
        return dict(mean=float(finite.mean()) if len(finite) else None, p05=float(np.quantile(finite, .05)) if len(finite) else None,
            p95=float(np.quantile(finite, .95)) if len(finite) else None, robots=per,
            robots_with_paired_live_samples=len(finite),paired_live_samples=int(window.sum()),
            scope='Paired live samples only. Falling robots and time lost to falls are reported separately; this conditional mean cannot establish overall improvement.')
    head = lambda q: np.rad2deg(np.sqrt(np.square(q[:, :, upper[:2]]).sum(-1)))
    raw_diff = d['applied_raw_action'][:, ids][:, :, lower] - d['applied_raw_action'][:, control][:, :, lower]
    own_diff = d['student_raw_action'][:, ids][:, :, lower] - d['student_raw_action'][:, control][:, :, lower]
    raw_max = float(np.abs(raw_diff).max())
    if g['mode'].startswith('body_replay'): maximum_replay_difference = max(maximum_replay_difference, raw_max)
    row = dict(name=g['name'], mode=g['mode'], seed=g['seed'], command=g['command'], sign=g['sign'],
        survival=g['survival'], resets=g['resets'], control_survival=groups[j]['survival'],
        pre_intervention_joint_RMS_difference_rad=float(np.sqrt(np.square(d['q'][pre, ids] - d['q'][pre, control]).mean())),
        applied_lower_action_max_difference=raw_max,
        truthful_own_lower_action_RMS_difference=float(np.sqrt(np.square(own_diff[window]).mean())) if window.any() else None,
        paired_mean_reward_rate_delta=paired_metric('reward_rate'),
        paired_absolute_lateral_speed_delta=paired_metric('root', lambda x: np.abs(x[:, :, 8])),
        paired_signed_speed_delta=paired_metric('root', lambda x: x[:, :, 7]),
        paired_head_angle_magnitude_delta_deg=paired_metric('q', head),
        post_actual_pitch_shift_deg=np.rad2deg((d['q'][:, ids][:, :, shoulder] - d['q'][:, control][:, :, shoulder])[window].mean(0)).tolist() if window.any() else None,
        post_physical_target_pitch_shift_deg=np.rad2deg((d['target'][:, ids][:, :, shoulder] - d['target'][:, control][:, :, shoulder])[window].mean(0)).tolist() if window.any() else None,
        own_lower_effective_action_RMS_difference=float(np.sqrt(np.square(np.clip(d['student_raw_action'][:, ids][:, :, lower], -3, 3)
            - np.clip(d['student_raw_action'][:, control][:, :, lower], -3, 3))[window].mean())) if window.any() else None,
        paired_live_sample_fraction_after6s=float(window[post].mean()),
        first_failed_time_s=float(t[np.flatnonzero(d['failures'][:,ids].any(1))[0]]) if d['failures'][:,ids].any() else None)
    rows.append(row)
assert maximum_replay_difference == 0.
(OUT / 'arm_coupling_analysis.json').write_text(json.dumps(dict(completed=True, checkpoint=r['checkpoint'],
    rows=rows, maximum_replayed_lower_action_difference=maximum_replay_difference,
    intervention='Smooth opposite5deg shoulder target offsets after5s, at original gains/caps. Paired unmodified body-action stream versus truthful closed-loop policy.',
    inference='Body replay isolates response to arm changes with prescribed nonshoulder actions; closed-loop response includes physical coupling and learned feedback. Nonlinearity and remaining independent shoulder feedback prevent an additive decomposition.',
    safe_production_repair_established=False, physical_coupling_proven_impossible=False, experimental_controllers_deployed=False), indent=2) + '\n')

target_rows = []; phase_rows = []
for i, g in enumerate(groups):
    if g['mode'] not in ['control', 'transition']: continue
    ids = slice(i * per, (i + 1) * per)
    times = l['time']; speed = l['command'][:, ids, 0]
    good = l['alive'][:, ids]
    teacher = np.clip(l['teacher_raw'][:, ids][:, :, lower], -3, 3)
    student = l['student_raw'][:, ids][:, :, lower]
    projected_error = l['student_projected'][:, ids][:, :, lower] - l['teacher_projected'][:, ids][:, :, lower]
    for label, mask in [('settled', (times >= (8 if g['transition'] else 5))[:, None] & good),
            ('command_switch_first_second', ((times >= 6) & (times < 7))[:, None] & good)]:
        if label.startswith('command_switch') and not g['transition']: continue
        if not mask.any():
            target_rows.append(dict(name=g['name'],mode=g['mode'],seed=g['seed'],command=g['command'],window=label,
                teacher_command_in_production_retention_contract=g['command']<=r['teacher_maximum_certified_command_m_s'],
                samples=0,scope='No paired live label samples in this window; do not infer successful response.'))
            continue
        data = dict(name=g['name'], mode=g['mode'], seed=g['seed'], command=g['command'], window=label,
            teacher_command_in_production_retention_contract=g['command'] <= r['teacher_maximum_certified_command_m_s'],
            certified_label_fraction=float(l['teacher_within_certified_command_range'][:, ids][mask].mean()),
            samples=int(mask.sum()), raw_lower_MSE=float(np.square(student - teacher)[mask].mean()),
            clipped_endpoint_lower_MSE=float(np.square(np.clip(student, -3, 3) - teacher)[mask].mean()),
            projected_lower_target_RMS_deg=float(np.rad2deg(np.sqrt(np.square(projected_error)[mask].mean()))),
            teacher_lower_joint_clip_fraction=float(l['teacher_joint_clipped'][:, ids][:, :, lower][mask].mean()),
            teacher_lower_slew_fraction=float(l['teacher_slew_limited'][:, ids][:, :, lower][mask].mean()),
            teacher_lower_CAD_rejection_fraction=float(l['teacher_CAD_invalid'][:, ids][:, :, lower][mask].mean()),
            existing_head_context_eligibility_fraction=float(l['context_eligible'][:, ids][mask].mean()),
            context_vs_truthful_teacher_lower_RMS_action=float(np.sqrt(np.square(l['teacher_raw'][:, ids][:, :, lower]
                - l['truthful_teacher_raw'][:, ids][:, :, lower])[mask].mean())),
            per_joint_projected_target_RMS_deg={names[int(k)]: float(np.rad2deg(np.sqrt(np.square(projected_error[:, :, jj])[mask].mean()))) for jj, k in enumerate(lower)})
        target_rows.append(data)
    mask = (times >= (8 if g['transition'] else 5))[:, None] & good
    p = l['phase'][:, ids]
    actual_force = d['force'][::5, ids]; actual_clearance = d['clearance'][::5, ids]
    actual = (actual_force < 1).all(-1) & (actual_clearance > .001).all(-1)
    bins = []
    for b in range(16):
        selected = mask & (p >= b / 16) & (p < (b + 1) / 16)
        if not selected.any(): continue
        bins.append(dict(beginning=b / 16, samples=int(selected.sum()),
            projected_target_lower_RMS_deg=float(np.rad2deg(np.sqrt(np.square(projected_error)[selected].mean()))),
            mean_sole_force_N=actual_force[selected].mean(0).tolist(),
            mean_sole_clearance_m=actual_clearance[selected].mean(0).tolist(),
            actual_flight_fraction=float(actual[selected].mean())))
    phase_rows.append(dict(name=g['name'], bins=bins,
        scope='Pre-action phase versus next20ms measured force/clearance.16 descriptive phase bins,10Hz label sampling; no fixed-latency causal claim.'))
(OUT / 'teacher_effective_target_analysis.json').write_text(json.dumps(dict(completed=True,
    checkpoint=r['checkpoint'], teacher=r['qualified_teacher_checkpoint'], rows=target_rows, phase_rows=phase_rows,
    maximum_actual_projection_error_rad=r['maximum_physical_target_reconstruction_error_rad'],
    production_labels_changed=False, defect_or_gradient_repair_proven=False,
    scope='Passive original phase-conditioned teacher on truthful current histories.3.5m/s teacher output is explicitly unqualified extrapolation outside the production retention contract; it cannot supply a safe new label. Targets use the same previous physical targets and native clamp,joint,slew,CAD map. Does not measure full teacher trajectories or demonstrate safe new gradients.'), indent=2) + '\n')

with np.load(folder / 'baseline_motors_200hz.npz') as z:m={k:z[k] for k in z.files}
motor_rows = []; baseline_groups = [g for g in groups if g['mode'] == 'control']
for i, g in enumerate(baseline_groups):
    ids = slice(i * per, (i + 1) * per)
    alive = m['alive'][1000:, ids]
    driving = m['driving_clip'][1000:, ids] & alive[:, :, None]
    ratios = m['cap_ratio'][1000:, ids]; req = m['requested'][1000:, ids]; applied = m['applied'][1000:, ids]
    v = m['velocity'][1000:, ids]
    if not alive.any():
        motor_rows.append(dict(name=g['name'],command=g['command'],motors=[],scope='No alive motor samples after5s.'))
        continue
    streak = np.zeros(driving.shape[1:], dtype=int); longest = streak.copy()
    for tick in driving: streak = np.where(tick, streak + 1, 0); longest = np.maximum(longest, streak)
    motors = []
    for j, name in enumerate(m['motor_names'].tolist()):
        good = alive
        motors.append(dict(name=name, p99_cap_ratio=float(np.quantile(ratios[:, :, j][good], .99)),
            at_or_above_cap_fraction=float((ratios[:, :, j][good] >= 1).mean()),
            driving_clip_fraction=float(driving[:, :, j][good].mean()),
            maximum_driving_clip_seconds=float(longest[:, j].max() * .005),
            overspeed_braking_fraction=float(((ratios[:, :, j] >= 1) & (req[:, :, j] * v[:, :, j] <= 0))[good].mean()),
            requested_torque_absolute_p99=float(np.quantile(abs(req[:, :, j])[good], .99)),
            applied_torque_absolute_p99=float(np.quantile(abs(applied[:, :, j])[good], .99))))
    motor_rows.append(dict(name=g['name'], command=g['command'], motors=motors))
(OUT / 'current_substep_motor_attribution.json').write_text(json.dumps(dict(completed=True,
    checkpoint=r['checkpoint'], sample_rate_Hz=200, substeps=len(m['cap_ratio']), groups=motor_rows,
    source='Passive original CAD actuator Jacobians/reductions/native computed and applied effort on unmodified control cohorts only.',
    relevant_actuator_ceiling_established=False, endpoint_verified=False,
    scope='This actor,commands.15/1.5/3.5,two seeds. Isolated overspeed/braking and brief clips do not prove limits block5m/s. No4/5 high-speed qualification.'), indent=2) + '\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 2, figsize=(12, 10), sharex='col')
for speed, row in zip([.15, 1.5, 3.5], ax):
    for mode, color in [('control', 'black'), ('body_replay_positive', 'tab:blue'), ('body_replay_negative', 'tab:cyan'),
            ('closed_loop_positive', 'tab:red'), ('closed_loop_negative', 'tab:orange')]:
        gi = index[(4321, speed, mode)]; ids = slice(gi * per, (gi + 1) * per)
        for a, key in [(row[0], 'q'), (row[1], 'root')]:
            value = np.rad2deg(d[key][:, ids, shoulder[0]]) if key == 'q' else d[key][:, ids, 8]
            value = np.where(d['alive'][:, ids], value, np.nan)
            a.plot(t, np.nanmean(value, 1), color=color, lw=1, label=mode)
            a.axvline(5, color='.7', lw=.5)
    row[0].set_ylabel(f'{speed}m/s left pitch(deg)'); row[1].set_ylabel('World lateral speed(m/s)')
    for a in row: a.grid(alpha=.2)
ax[0, 0].legend(fontsize=8); ax[-1, 0].set_xlabel('Physical time(s)'); ax[-1, 1].set_xlabel('Physical time(s)')
fig.suptitle('Paired arm-target/base response:32 independent states,seed4321; existing gains and native caps')
fig.tight_layout(); fig.savefig(OUT / 'arm_coupling_response.png', dpi=150); plt.close(fig)
print(json.dumps(dict(completed=True, coupling_rows=len(rows), target_rows=len(target_rows), motor_groups=len(motor_rows))))
