"""Restart the diagnosed balance repair from the independently qualified prior.

The failed walking candidates remain archived. No CAD joint, collision envelope,
mass or hardware limit is changed. Running corrections activate by command speed.
"""
import json, subprocess, time
from pathlib import Path
import psutil
from status_io import atomic_json

here = Path(__file__).resolve().parent
prior = here/'runs/locomotion_20261001_r11_net_travel'
old = json.loads((prior/'state.json').read_text())
assert old['status']=='superseded_for_diagnostic_repair' and old['child_pid'] is None
for p in psutil.process_iter(['name', 'cmdline']):
    command = ' '.join(p.info['cmdline'] or []).replace('\\','/')
    assert not ('python' in (p.info['name'] or '').lower() and
                ('training/worker.py' in command or 'training/supervisor.py' in command)), command
run = here/'runs/locomotion_20261002_r12_balance_events'
run.mkdir(exist_ok=True)
assert not (run/'state.json').exists()
initial = here/'checks/reward_repair_20261001/stand_normalization_repaired.pt'
command = [str(here.parents[1]/'env_isaaclab/Scripts/python.exe'), '-u', '-X', 'utf8',
    str(here/'supervisor.py'), '--output', str(run), '--reserve_mb', '300',
    '--max_hours', '18', '--iterations_per_block', '64',
    '--validated_checks', str(here/'checks/reward_repair_20261001/reused_checks'),
    '--initial_checkpoint', str(initial), '--train_envs', '8192']
with (run/'supervisor_stdout.log').open('w', encoding='utf-8') as out, (run/'supervisor_stderr.log').open('w', encoding='utf-8') as err:
    proc = subprocess.Popen(command, stdout=out, stderr=err,
        cwd=str(here.parents[1]), creationflags=subprocess.CREATE_NO_WINDOW)
atomic_json(run/'launch.json', dict(command=command, pid=proc.pid, time=time.time(),
    prior_run=str(prior), source_checkpoint=str(initial), target_environments=8192,
    reason='Weak fall event cost, lost balance incentive, reference/control envelope mismatch, running hop credit',
    preserved_prior_checkpoint=old['checkpoint'], diagnosis=str(here/'checks/balance_audit_20261002/diagnosis.json')))
atomic_json(here/'ACTIVE_RUN.json', dict(run=str(run), supervisor_pid=proc.pid,
    started_unix=time.time(), prior_run=str(prior),
    note='Balance-event reward v4; running flight/alternating gates; bounded reference, named joint diagnostics; 8192 target, 300MiB reserve, 180s ramp'))
print(json.dumps(dict(run=str(run), launcher_pid=proc.pid)))
