"""Publish verified owned continuation status after the real startup tests."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import psutil

here=Path(__file__).resolve().parent
training=here.parents[1]
sys.path.insert(0,str(training))
from status_io import atomic_json
experiment=json.loads((here/'experiment.json').read_text())
run=training/'runs/locomotion_20261004_r30_exact_graph_execution'
state=json.loads((run/'state.json').read_text())
assert state['status']=='running'
supervisor=psutil.Process(state['supervisor_pid'])
assert str(run) in ' '.join(supervisor.cmdline())
assert any(Path(arg).name=='supervisor_unitree.py' for arg in supervisor.cmdline())
assert json.loads((run/'001_physics_preflight/result.json').read_text())['passed']
assert json.loads((run/'002_resource_ppo_smoke/result.json').read_text())['completed']
original=Path(experiment['checkpoint'])
assert hashlib.sha256(original.read_bytes()).hexdigest()==experiment['checkpoint_sha256']
pointer=training/'ACTIVE_RUN.json'
before=json.loads(pointer.read_text())
assert Path(before['run']).resolve()==Path(experiment['run']).resolve()
atomic_json(here/'ACTIVE_RUN_before_continuation.json',before)
timestamp=datetime.now(timezone.utc).isoformat()
atomic_json(pointer,dict(run=str(run),started_unix=state['started_unix'],supervisor_pid=state['supervisor_pid'],
                        initial_full_checkpoint=str(original),prior_run=experiment['run'],prior_main_updates=256,
                        status='running',started_utc=timestamp,optimization_iteration='04',
                        optimization_proof=str(here/'continuation_proof.json'),
                        note='Exact graph execution continuation: actor/critic/Adam/exploration/adaptive LR/standing teacher retained; fresh physical/PPO/load/capacity and current independent gates; target8192 reserve300 fallback180s healthy ramp; four-hour automation stays paused'))
old=Path(experiment['run'])/'state.json'
old_state=json.loads(old.read_text())
assert old_state['status']=='paused_for_performance_review' and old_state.get('child_pid') is None
atomic_json(here/'R29_state_before_continuation.json',old_state)
old_state.update(status='superseded_for_optimization',continued_run=str(run),continued_utc=timestamp,
                 continuation_scope='Implementation-only exact math optimization, full checkpoint retained, current independent gates rerun')
atomic_json(old,old_state)
atomic_json(here/'continuation_status.json',dict(recorded_utc=timestamp,run=str(run),state=state,
             supervisor_command=supervisor.cmdline(),original_checkpoint_untouched=True,
             preflight_passed=True,full_ppo_smoke_completed=True,automation_reenabled=False))
print(json.dumps(dict(active_run=str(run),status=state['status'],operation=state.get('operation'),supervisor_pid=state['supervisor_pid'])))
