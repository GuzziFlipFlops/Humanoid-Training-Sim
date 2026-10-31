"""Authorized net-travel repair restart; saved CAD is separate from training."""
import json, subprocess, time
from pathlib import Path
import psutil
from status_io import atomic_json

here = Path(__file__).resolve().parent
prior = here/'runs/locomotion_20261001_r10_auto_vram_duration'
old = json.loads((prior/'state.json').read_text())
assert old['status']=='superseded_for_reward_v3' and old['child_pid'] is None
for p in psutil.process_iter(['name', 'cmdline']):
    command = ' '.join(p.info['cmdline'] or [])
    assert not ('python' in (p.info['name'] or '').lower() and
                ('training/worker.py' in command.replace('\\','/') or
                 'training/supervisor.py' in command.replace('\\','/'))), command
run = here/'runs/locomotion_20261001_r11_net_travel'
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
    prior_run=str(prior), reason='Verified stalled walking, rectified rocking credit and repeated short-block rollback',
    source_checkpoint=str(initial), target_environments=8192,
    cad='V32 archived locally and Fusion closed; unchanged frozen physical_v29 training asset'))
atomic_json(here/'ACTIVE_RUN.json', dict(run=str(run), supervisor_pid=proc.pid,
    started_unix=time.time(), prior_run=str(prior),
    note='Net-travel reward v3; fresh smoke and capacity benchmarks; 8192 target, 300MiB reserve, 180s ramp'))
print(json.dumps(dict(run=str(run), launcher_pid=proc.pid)))
