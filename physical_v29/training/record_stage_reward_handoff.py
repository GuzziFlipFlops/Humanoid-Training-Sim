"""Record R23 provenance and live evidence without creating an Isaac worker."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

HERE = Path(__file__).resolve().parent
DIAG = HERE / 'diagnostics/stage_rewards_20261002'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


active = read(HERE / 'ACTIVE_RUN.json')
run = Path(active['run'])
assert run.name == 'locomotion_20261002_r23_standing_curriculum'
frozen = run / 'source/physical_v29'
config = read(run / 'chain_config.json')
integrity = {}
for name, expected in config['sources_sha256'].items():
    if name in ('supervisor_unitree.py', 'vram_ramp.py'):
        path = run / name
    elif name.startswith('physical_assets/'):
        path = frozen / 'assets' / name.removeprefix('physical_assets/')
    elif name.startswith('physical_v29/'):
        path = frozen / name.removeprefix('physical_v29/')
    else:
        path = frozen / 'training' / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    integrity[name] = actual == expected
assert all(integrity.values()), integrity

# Reproduce the saved support flags using actual simulator COM, not URDF FK.
audit_path = DIAG / 'direct_sim_com_support_audit.json'
if not audit_path.exists():
    import numpy as np
    import torch

    sys.path.insert(0, str(frozen / 'training'))
    from support_geometry import horizontal_support

    folder = run / '003_checkpoint_reload'
    with np.load(folder / 'gait_diagnostics_50hz.npz') as trace, np.load(folder / 'telemetry.npz') as telemetry:
        dt = float(telemetry['step_dt'])
        selected = np.arange(len(trace['time'])) >= int(2.0 / dt)
        roots = trace['root_state'][selected]
        feet = trace['foot_position'][selected]
        com = trace['com'][selected]
        contact = trace['contact'][selected]
        batch = roots.shape[0] * roots.shape[1]
        outside, excess = horizontal_support(
            torch.from_numpy(roots[..., 3:7].reshape(batch, 4)),
            torch.from_numpy(roots[..., :3].reshape(batch, 3)),
            torch.from_numpy(feet.reshape(batch, 2, 3)),
            torch.from_numpy(com.reshape(batch, 3)),
            torch.from_numpy(contact.reshape(batch, 2).astype(bool)),
        )
        recorded = telemetry['metrics'][:, :roots.shape[1], 15].reshape(-1).astype(bool)
        reproduced = outside.numpy()
        assert np.array_equal(recorded, reproduced)
        audit = dict(
            passed=True, source=str(folder), ticks=roots.shape[0], robots=roots.shape[1],
            samples=batch, mismatched_flags=int(np.count_nonzero(recorded != reproduced)),
            outside_fraction=float(reproduced.mean()), mean_excess_m=float(excess.mean()),
            scope='CPU replay of actual simulator COM for the first 16 robots after 2s; '
                  '4-second checkpoint load smoke, not a standing qualification',
        )
    save(audit_path, audit)

state = read(run / 'state.json')
processes = []
for proc in psutil.process_iter(['pid', 'ppid', 'name', 'create_time', 'cmdline']):
    try:
        info = proc.info
        command = ' '.join(info['cmdline'] or [])
        if str(run) in command and ('worker_unitree.py' in command or 'supervisor_unitree.py' in command):
            processes.append(info)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
assert any(p['pid'] == state['supervisor_pid'] for p in processes), 'Supervisor ownership unverified'
if state.get('child_pid'):
    assert any(p['pid'] == state['child_pid'] for p in processes), 'Worker ownership unverified'
worker_processes = [p for p in processes if 'worker_unitree.py' in ' '.join(p['cmdline'] or [])]
worker_pids = {p['pid'] for p in worker_processes}
worker_parents = {p['ppid'] for p in worker_processes}
worker_leaves = sorted(worker_pids - worker_parents)
assert len(worker_leaves) <= 1, 'More than one actual worker process'
operation = Path(state['operation_path']) if state.get('operation_path') else None
worker_progress = read(operation / 'progress.json') if operation and (operation / 'progress.json').exists() else None
local_updates = len((operation / 'progress.jsonl').read_text().splitlines()) if operation and (operation / 'progress.jsonl').exists() else 0

benchmarks = read(run / 'benchmarks.json')
by_count = {b['environments']: b for b in benchmarks}
comparison = None
if 6144 in by_count and 8192 in by_count:
    lower, upper = by_count[6144], by_count[8192]
    comparison = dict(
        raw_8192_over_6144=upper['environment_steps_per_second'] / lower['environment_steps_per_second'],
        effective_8192_over_6144=upper['effective_environment_steps_per_second'] / lower['effective_environment_steps_per_second'],
        scope='Bounded PPO capacity tests including process startup and exit; '
              'actual update counts differ if a host-RAM guard fires; not a long-run convergence test',
    )
gates = []
for path in sorted(run.glob('*_gate_*/evaluation.json')):
    report = read(path)
    gates.append(dict(folder=str(path.parent), report=report))
memory = psutil.virtual_memory()
gpu = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu',
     '--format=csv,noheader'], text=True, timeout=10).strip()
handoff = dict(
    recorded_utc=datetime.now(timezone.utc).isoformat(), recorded_unix=time.time(),
    active_run=active, state=state, processes=processes, actual_worker_pids=worker_leaves,
    worker_progress=worker_progress, local_updates_recorded=local_updates, gpu=gpu,
    system_memory=dict(available_mb=memory.available // 2**20, total_mb=memory.total // 2**20,
                       used_percent=memory.percent),
    frozen_source_integrity=integrity, benchmarks=benchmarks, throughput_comparison=comparison,
    independent_gates=gates, direct_sim_com_support_audit=read(audit_path),
    smoke_scope='Physics preflight, four PPO updates, checkpoint reload and reconstructed visuals '
                'validate execution only. Only full independent gates can qualify stages.',
    qualification_reused=False,
)
save(DIAG / 'live_handoff.json', handoff)
print(json.dumps(dict(
    recorded_utc=handoff['recorded_utc'], status=state['status'], stage=state.get('stage'),
    operation_path=state.get('operation_path'), environments=state.get('environment_count'),
    training_iterations=state.get('total_training_iterations'),
    iteration=(state.get('progress') or {}).get('iteration'),
    local_updates_recorded=local_updates, actual_worker_pids=worker_leaves,
    gpu=gpu, frozen_files_verified=len(integrity),
    support_audit=handoff['direct_sim_com_support_audit'], comparison=comparison,
    gates=[dict(folder=g['folder'], passed=g['report']['passed'],
                failures=[c.get('failures') for c in g['report']['cases']]) for g in gates],
), indent=2))
