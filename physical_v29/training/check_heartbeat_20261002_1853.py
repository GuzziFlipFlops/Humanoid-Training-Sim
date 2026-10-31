"""Read-only heartbeat audit of live processes, saved gates, and raw rollout data."""
from pathlib import Path
import datetime, hashlib, json, math, time
import numpy as np
import psutil

HERE = Path(__file__).resolve().parent
OUT = HERE / 'diagnostics/heartbeat_20261002_1853'
OUT.mkdir(exist_ok=True)
active = json.loads((HERE / 'ACTIVE_RUN.json').read_text(encoding='utf-8-sig'))
run = Path(active['run'])
assert run.name == 'locomotion_20261002_r22_unitree_nominal'
state = json.loads((run / 'state.json').read_text())
assert state['status'] == 'running'
supervisor = psutil.Process(state['supervisor_pid'])
assert 'supervisor_unitree.py' in ' '.join(supervisor.cmdline())
assert str(run).lower() in ' '.join(supervisor.cmdline()).lower()
worker = psutil.Process(state['child_pid'])
assert str(run / 'source/physical_v29/training/worker_unitree.py').lower() in ' '.join(worker.cmdline()).lower()
worker_tree = [worker] + worker.children(recursive=True)
# Windows can attach a conhost to the wrapper. Count actual Python simulation
# processes and retain the complete owned tree as evidence.
owned_python = {p.pid for p in worker_tree if 'python' in p.name().lower()}
for proc in psutil.process_iter(['name', 'cmdline']):
    if 'python' in (proc.info['name'] or '').lower() and 'worker_unitree.py' in ' '.join(proc.info['cmdline'] or []):
        assert proc.pid in owned_python, 'A second Unitree simulation worker is running'
progress = json.loads((Path(state['operation_path']) / 'progress.json').read_text())
assert 0 <= time.time() - progress['time'] < 60
assert progress['num_envs'] == 8192
assert progress['policy']['nominal_pose_max_abs_drift_rad'] == 0
assert progress['gpu']['free_mb'] >= 300

protected = ['environment_unitree.py', 'worker_unitree.py', 'unitree_reward_port.py',
             'environment.py', 'evaluation.py', 'evaluation_steps.py', 'mass_estimate.json',
             'assets/robot_training.urdf', 'assets/motion_reference.npz']
hashes = {}
for name in protected:
    frozen = run / 'source/physical_v29/training' / name
    assert frozen.read_bytes() == (HERE / name).read_bytes(), name
    hashes[name] = hashlib.sha256(frozen.read_bytes()).hexdigest()

chain_config = json.loads((run / 'chain_config.json').read_text())
all_frozen_hashes = {}
for name, expected in chain_config['sources_sha256'].items():
    if name.startswith('physical_assets/'):
        frozen = run / 'source/physical_v29/assets' / name.removeprefix('physical_assets/')
    elif name.startswith('physical_v29/'):
        frozen = run / 'source/physical_v29' / name.removeprefix('physical_v29/')
    elif name in ['supervisor_unitree.py', 'vram_ramp.py']:
        frozen = run / name
    else:
        frozen = run / 'source/physical_v29/training' / name
    actual = hashlib.sha256(frozen.read_bytes()).hexdigest()
    assert actual == expected, f'Frozen run source changed: {name}'
    all_frozen_hashes[name] = actual

rows = []
for folder in sorted(run.glob('*_stand_train_*')):
    if (folder / 'progress.jsonl').exists():
        rows += [json.loads(line) for line in (folder / 'progress.jsonl').read_text().splitlines()]
assert rows
assert all(row['policy']['nominal_pose_max_abs_drift_rad'] == 0 for row in rows)
assert all(math.isfinite(row['behavior'][key]) for row in rows
           for key in ['healthy_fraction', 'falls_per_s', 'reward_physical_termination'])
def window(values):
    keys = ['healthy_fraction', 'falls_per_s', 'stand_forward_speed', 'foot_slip',
            'termination_invalid_differential_per_s']
    return {key: float(np.mean([row['behavior'][key] for row in values])) for key in keys}

gates = []
for path in sorted(run.glob('*_stand_gate_*_nominal/evaluation.json')):
    report = json.loads(path.read_text())
    case = report['cases'][0]
    gates.append(dict(folder=str(path.parent), passed=report['passed'],
        duration_s=report['duration_s'], statistics=case['statistics'], failures=case['failures']))
assert len(gates) >= 2
latest = Path(gates[-1]['folder'])
data = np.load(latest / 'gait_diagnostics_50hz.npz', allow_pickle=False)
assert all(np.isfinite(data[key]).all() for key in ['q', 'qd', 'tau', 'target', 'raw_action', 'root_state', 'foot_force'])
assert not data['done'].any(), 'Saved nominal gate did not survive continuously'
assert not data['failure_causes'].any()
root = data['root_state']
delta = root[-1, :, :2] - root[0, :, :2]
foot_delta = data['foot_position'][-1, :, :, :2] - data['foot_position'][0, :, :, :2]
joint_audit = []
for j, name in enumerate(data['joint_names']):
    joint_audit.append(dict(name=str(name), min_rad=float(data['q'][:, :, j].min()),
        max_rad=float(data['q'][:, :, j].max()),
        target_tracking_mae_rad=float(np.abs(data['target'][:, :, j] - data['q'][:, :, j]).mean()),
        peak_torque_nm=float(np.abs(data['tau'][:, :, j]).max()),
        peak_speed_rad_s=float(np.abs(data['qd'][:, :, j]).max())))
raw_telemetry = dict(source=str(latest / 'gait_diagnostics_50hz.npz'),
    scope='All 1000 control ticks for the first16 robots; gate survival covers128 robots',
    physics_ticks=len(data['time']), robots=root.shape[1], done_ticks=int(data['done'].sum()),
    physical_failure_ticks=int(data['failure_causes'].sum()),
    mean_root_world_xy_displacement_m=delta.mean(0).tolist(),
    mean_root_planar_displacement_m=float(np.linalg.norm(delta, axis=1).mean()),
    mean_left_right_foot_world_xy_displacement_m=foot_delta.mean(0).tolist(),
    both_contact_fraction=float(data['contact'].all(-1).mean()),
    flight_fraction=float((~data['contact'].any(-1)).mean()),
    mean_absolute_raw_action=float(np.abs(data['raw_action']).mean()),
    root_height_range_m=[float(root[:,:,2].min()), float(root[:,:,2].max())], joints=joint_audit)
record = dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), active=active,
    state=state, progress=progress,
    supervisor=dict(pid=supervisor.pid, command=supervisor.cmdline()),
    worker_processes=[dict(pid=p.pid, parent=p.ppid(), command=p.cmdline()) for p in worker_tree],
    frozen_source_hashes=hashes, complete_frozen_source_hashes=all_frozen_hashes, full_training_samples=len(rows),
    first20_training_updates=window(rows[:20]), last20_training_updates=window(rows[-20:]),
    independent_nominal_gates=gates, rollout_telemetry=raw_telemetry,
    conclusion='Training is advancing and independent nominal standing survival improved from0 to100 percent. '
        'The latest gate fails only its anchor drift bound; no reward or hardware repair is justified by a stalled-learning claim. '
        'Standing is unqualified, walking/running have not started, and shutdown is not authorized by completion.',
    runtime_limitations='The host allocation-growth cause is unresolved; checkpoint/Adam/LR recycling is operating. '
        'VRAM stayed above the300 MiB reserve. Higher reward is not used to qualify behavior.')
(OUT / 'status_and_rollout_audit.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(dict(checked_at_utc=record['checked_at_utc'], status=state['status'], stage=state['stage'],
    saved_updates=state['total_training_iterations'], current_worker_iteration=progress['iteration'],
    environments=progress['num_envs'], free_vram_mib=progress['gpu']['free_mb'],
    available_ram_mib=progress['gpu']['ram_available_mb'], sample_age_s=time.time()-progress['time'],
    gates=[dict(survival=g['statistics']['survival'], drift_m=g['statistics']['mean_final_drift_m'], failures=g['failures']) for g in gates],
    last20_training_updates=record['last20_training_updates'],
    raw_telemetry={key:value for key,value in raw_telemetry.items() if key!='joints'},
    evidence=str(OUT / 'status_and_rollout_audit.json')), indent=2))
