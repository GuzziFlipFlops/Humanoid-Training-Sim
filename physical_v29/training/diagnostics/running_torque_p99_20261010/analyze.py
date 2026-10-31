"""CPU-only p99/max clarification from preserved, synchronized 200 Hz captures."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import numpy as np

TRAIN = Path('E:/Aditya/robot_sim/physical_v29/training')
FINAL = TRAIN / 'diagnostics/final_check_20261010_0905'
RUN = TRAIN / 'runs/locomotion_20261007_r40_goal_budget_full_state'
OUT = Path(__file__).resolve().parent
ASSET = RUN / 'source/physical_v29/assets/actuator_limits.json'
ACTOR = RUN / '1064_run_350_train_352/candidate.pt'
ASSET_RECORDS = json.loads(ASSET.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.size and np.isfinite(values).all()
    a = np.abs(values)
    return dict(samples=int(values.size), torque_p99_abs_Nm=float(np.quantile(a, .99, method='linear')),
                torque_max_abs_Nm=float(a.max()), torque_RMS_Nm=float(np.sqrt(np.mean(values ** 2))),
                signed_min_Nm=float(values.min()), signed_max_Nm=float(values.max()))


def analyze(folder, commands, scope):
    mapping = json.loads((folder / 'motor_mapping.json').read_text())
    original = json.loads((folder / 'motor_population_or_episode_summary.json').read_text())
    with np.load(folder / 'native_motor_200hz.npz', allow_pickle=False) as z:
        t = z['applied_torque_Nm']
        requested = z['requested_torque_Nm']
        pivot = z['applied_pivot_torque_Nm']
        pivot_requested = z['requested_pivot_torque_Nm']
        valid = z['valid'].astype(bool)
        alive = z['alive'].astype(bool)
        terminated = z['terminated'].astype(bool)
        truncated = z['truncated'].astype(bool)
        times = z['interval_start_s']
        env_ids = z['environment_ids']
        command = z['command_m_s']
        rpm = z['velocity_RPM']
        assert t.shape == requested.shape == pivot.shape == pivot_requested.shape == valid.shape
        assert t.shape[-1] == len(mapping['motor_names']) == 23
        assert np.allclose(np.diff(times), .005, rtol=0, atol=2e-6)
        assert np.allclose(command, mapping['command'])
        assert not terminated.any() and not truncated.any(), 'Unexpected reset/terminal requires segmentation'
        assert alive.all() and valid.all(), 'Unexpected unhealthy samples require explicit review'
        assert np.isfinite(t).all() and np.isfinite(pivot).all()
        checks = []
        for i, ref in enumerate(mapping['shaft_references']):
            if 'reduction' in ref:
                name = ref['joint_names'][0]
                j = mapping['joint_names'].index(name)
                record = ASSET_RECORDS[name]
                r = float(record.get('physical_external_reduction', record.get('reduction', 1.)))
                eta = float(record.get('transmission_efficiency', 1.)) if r != 1 else 1.
                assert r == ref['reduction'] and eta == ref['modeled_external_efficiency']
                np.testing.assert_allclose(t[..., i], pivot[..., j] / (r * eta), rtol=2e-7, atol=2e-6)
                np.testing.assert_allclose(requested[..., i], pivot_requested[..., j] / (r * eta), rtol=2e-7, atol=2e-6)
                checks.append(dict(motor=ref['name'], kind='frozen reduction and efficiency verified against raw pivot effort',
                                   reduction=r, efficiency=eta,
                                   max_pivot_reconstruction_error_Nm=float(np.max(np.abs(t[..., i].astype(float)*r*eta-pivot[..., j])))))
            else:
                pair = ref['name'].rsplit('_motor_', 1)[0]
                m = int(ref['name'].rsplit('_', 1)[1]) - 1
                np.testing.assert_array_equal(t[..., i], z[f'cad_{pair}_motor_applied'][..., m])
                np.testing.assert_array_equal(requested[..., i], z[f'cad_{pair}_motor_requested'][..., m])
                checks.append(dict(motor=ref['name'], kind='bitwise equality to exact individual CAD motor effort'))
        rows = []
        trials = []
        for cmd in commands:
            selected = np.isclose(command, cmd)
            assert int(selected.sum()) == (32 if scope == 'nominal population' else 1)
            mask = (times[:, None] >= 2.) & selected[None, :] & alive & ~terminated & ~truncated
            for i, ref in enumerate(mapping['shaft_references']):
                v = mask & valid[..., i]
                base = stats(t[..., i][v])
                row = dict(scope=scope, command_m_s=cmd, motor=ref['name'], shaft=ref['shaft'], **base)
                req = stats(requested[..., i][v])
                row['requested_torque_p99_abs_Nm'] = req['torque_p99_abs_Nm']
                row['requested_torque_max_abs_Nm'] = req['torque_max_abs_Nm']
                row['speed_p99_abs_RPM'] = float(np.quantile(np.abs(rpm[..., i][v]).astype(float), .99))
                row['speed_max_abs_RPM'] = float(np.abs(rpm[..., i][v]).max())
                peak_flat = int(np.argmax(np.where(v, np.abs(t[..., i]), -1)))
                time_index, env_index = np.unravel_index(peak_flat, v.shape)
                row['maximum_event'] = dict(interval_start_s=float(times[time_index]), environment_id=int(env_ids[env_index]),
                                            signed_applied_torque_Nm=float(t[time_index, env_index, i]),
                                            simultaneous_signed_RPM=float(rpm[time_index, env_index, i]))
                if 'reduction' in ref:
                    j = mapping['joint_names'].index(ref['joint_names'][0])
                    p = stats(pivot[..., j][v])
                    row['pivot_joint'] = ref['joint_names'][0]
                    row['pivot_torque_p99_abs_Nm'] = p['torque_p99_abs_Nm']
                    row['pivot_torque_max_abs_Nm'] = p['torque_max_abs_Nm']
                    row['reduction'] = ref['reduction']
                    row['modeled_external_efficiency'] = ref['modeled_external_efficiency']
                old = next(x for x in original['summaries'][str(cmd)] if x['motor'] == ref['name'])
                assert np.isclose(base['torque_max_abs_Nm'], old['torque_peak_abs_Nm'], rtol=0, atol=1e-6)
                assert np.isclose(np.quantile(np.abs(t[..., i][v]), .95), old['torque_p95_abs_Nm'], rtol=0, atol=1e-5)
                row['prior_table_p95_and_peak_reproduced'] = True
                rows.append(row)
                for ei in np.flatnonzero(selected):
                    vv = v[:, ei]
                    trials.append(dict(command_m_s=cmd, motor=ref['name'], environment_id=int(env_ids[ei]),
                                       **stats(t[:, ei, i][vv])))
        return dict(scope=scope, capture=str(folder / 'native_motor_200hz.npz'), rows=rows, per_trial=trials,
                    validation=dict(source_shape=list(t.shape), shaft_mapping_checks=checks,
                                    resets_or_terminals=int(terminated.sum()+truncated.sum()),
                                    non_alive_samples=int((~alive).sum()), invalid_motor_samples=int((~valid).sum()),
                                    warmup_excluded_s=2, sample_period_s=.005,
                                    statistics='Pooled valid post-settling absolute applied torques; NumPy linear quantile'))


population = analyze(FINAL / 'selected_diverse_nominal', [1., 3.5], 'nominal population')
episode = analyze(TRAIN / 'videos/final_motor_capture_20261010_0905/locomotion_350', [3.5], 'single synchronized video episode')
eval_data = json.loads((FINAL / 'selected_diverse_nominal/evaluation.json').read_text())
running_case = next(x for x in eval_data['cases'] if x['statistics']['command_m_s'] == 3.5)
assert running_case['passed'] and running_case['statistics']['trials'] == 32
actor_hash = sha(ACTOR)
assert actor_hash == '0ad6a1cc1c018951b82d05c67d02ca7a2663eb93845030e255a04503c60a61d0'
result = dict(created_utc=datetime.now(timezone.utc).isoformat(), actor=str(ACTOR), actor_sha256=actor_hash,
              measured_signed_fixed_frame_speed_m_s=running_case['statistics']['reset_frame_forward_speed_m_s'],
              current_running_case_passed=True, complete_curriculum_qualified=False,
              population=population, video_episode=episode,
              limits='Simulation applied effort and derived installed shaft torque, not hardware measurements. Factory-geared outputs are not inaccessible internal rotor torques.',
              training='CPU-only analysis; no learning, evaluation worker, source edit or automation resume')
(OUT / 'torque_p99.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
running = [x for x in population['rows'] if x['command_m_s'] == 3.5]
columns = ['motor', 'torque_p99_abs_Nm', 'torque_max_abs_Nm', 'pivot_joint', 'pivot_torque_p99_abs_Nm',
           'pivot_torque_max_abs_Nm', 'requested_torque_p99_abs_Nm', 'requested_torque_max_abs_Nm', 'samples', 'shaft']
with (OUT / 'running_torque_p99.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore'); w.writeheader(); w.writerows(running)
lines = ['# Running torque: 99th percentile and maximum', '',
         f"Measured signed fixed-frame speed: **{result['measured_signed_fixed_frame_speed_m_s']:.6f} m/s**; command 3.5 m/s.", '',
         '32 independently initialized nominal-physics trials, 200 Hz. Statistics pool 115,200 valid samples per motor over 2–20 seconds; the first 2 seconds are settling and excluded. No falls, resets or invalid linkage samples occur. P99 is the 99th percentile of absolute applied torque, not a confidence limit. Maximum is the largest absolute applied torque in this capture. Pushed evaluations have no corresponding native torque population here.', '',
         'The old table provided p95, not p99. Its 4.25 N·m left-knee value was the **1 m/s motor-shaft maximum**, not the running knee-pivot torque. Knee shaft values are before the 8:1 planetary × 2:1 chain, using the frozen model’s 95% total external efficiency: T_pivot = T_motor × 16 × 0.95. This mapping has now been checked against the raw named pivot-effort arrays, independently of the summary table.', '',
         '| Knee | Motor shaft p99 (N·m) | Motor shaft max (N·m) | Knee pivot p99 (N·m) | Knee pivot max (N·m) |',
         '|---|---:|---:|---:|---:|']
for row in running:
    if 'knee' in row['motor']:
        lines.append(f"| {row['pivot_joint']} | {row['torque_p99_abs_Nm']:.3f} | {row['torque_max_abs_Nm']:.3f} | {row['pivot_torque_p99_abs_Nm']:.3f} | {row['pivot_torque_max_abs_Nm']:.3f} |")
lines += ['', '## Installed motor shafts', '',
          'DJI hip/knee values refer to bare shafts before their added reductions. Ankle/waist differentials use each exact CAD motor state at the factory-geared crank output; other geared/servo motors are reported at the installed output before any added transmission. These are simulation values, not measured hardware or continuous-duty ratings.', '',
          '| Motor | p99 absolute torque (N·m) | Maximum absolute torque (N·m) |', '|---|---:|---:|']
for row in running:
    lines.append(f"| {row['motor']} | {row['torque_p99_abs_Nm']:.3f} | {row['torque_max_abs_Nm']:.3f} |")
lines += ['', '## Single video episode, kept separate', '',
          'The synchronized 3.5-command video has one robot and 2,000 analyzed native samples over its 10-second recorded window. Its torque results are stored separately in torque_p99.json; they are not pooled into the 32-trial population above.', '',
          f'Actor: `{ACTOR}`; SHA256 `{actor_hash}`.', '',
          '[CSV](running_torque_p99.csv), [full statistics and validation](torque_p99.json), [reproducible CPU analysis](analyze.py).', '',
          'Training and automation remain stopped. No sources, checkpoints, caps, rewards or raw captures were modified.']
(OUT / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
sources = [population['capture'], episode['capture'], ASSET, ACTOR,
           FINAL / 'selected_diverse_nominal/motor_mapping.json', FINAL / 'selected_diverse_nominal/evaluation.json',
           FINAL / 'selected_diverse_nominal/motor_population_or_episode_summary.json',
           TRAIN / 'automation/motor_shaft_metrics.py', RUN / 'source/physical_v29/hardware_actuators.py']
files = [Path(p) for p in sources] + [OUT / 'analyze.py', OUT / 'torque_p99.json', OUT / 'running_torque_p99.csv', OUT / 'REPORT.md']
(OUT / 'artifact_hashes.json').write_text(json.dumps({str(p):sha(p) for p in files}, indent=2), encoding='utf-8')
print(json.dumps(dict(report=str(OUT/'REPORT.md'), measured_speed=result['measured_signed_fixed_frame_speed_m_s'],
                     running_rows=running, knee_1_m_s=[x for x in population['rows'] if x['command_m_s']==1 and 'knee' in x['motor']],
                     video_knees=[x for x in episode['rows'] if 'knee' in x['motor']]), indent=2))
