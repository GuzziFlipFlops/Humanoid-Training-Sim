"""Record the completed in-flight block without fabricating qualification."""
import datetime
import hashlib
import json
from pathlib import Path
import time
import psutil
import torch

DIAG = Path(__file__).resolve().parent
audit = json.loads((DIAG / 'pause_request.json').read_text())
run = Path(audit['run'])
operation = Path(audit['operation'])
result = json.loads((operation / 'result.json').read_text())
assert result['completed'] and not result.get('recycled'), result
for identity in audit['owned_worker_identities']:
    try:
        p = psutil.Process(identity['pid'])
        assert p.create_time() != identity['created'] or not p.is_running(), 'Owned worker still running'
    except psutil.NoSuchProcess:
        pass
state_path = run / 'state.json'
state = json.loads(state_path.read_text())
assert state['status'] == 'draining_for_performance_review'
checkpoint = Path(result['checkpoint']) if audit['mode'] == 'train' else Path(audit['prior_checkpoint'])
saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
assert saved['actor_state_dict'] and saved['critic_state_dict'] and saved['optimizer_state_dict']['state']
assert saved['infos']['contract'] == 'physical_v29_history5_unitree_defaultoffset_v2'
assert not any(torch.isnan(v).any() for v in saved['actor_state_dict'].values() if isinstance(v, torch.Tensor))
added = result.get('iterations', 0) if audit['mode'] == 'train' else 0
state.update(status='paused_for_performance_review', checkpoint=str(checkpoint), child_pid=None,
             total_training_iterations=audit['prior_iterations'] + added, updated_unix=time.time(),
             pause_reason='Four-hour heartbeat stopped at user request; full current block saved; simulator profiling and optimization assessment')
state['attempts'].append(dict(operation=audit['mode'], folder=str(operation),
                            environments=result.get('num_envs', state['environment_count']),
                            speed=state['stage_speed'], seconds=(operation/'result.json').stat().st_mtime-audit['owned_worker_identities'][0]['created'],
                            completed=True, failure=None, memory_failure=False,
                            recorded_after_supervisor_drain=True,
                            time_scope='Worker creation to completed result publication; excludes later review delay'))
state_path.write_text(json.dumps(state, indent=2))
audit.update(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             all_owned_workers_drained=True, iterations_saved=added,
             total_training_iterations=state['total_training_iterations'], checkpoint=str(checkpoint),
             checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             full_actor_critic_optimizer_saved=True, optimizer_lrs=[g['lr'] for g in saved['optimizer_state_dict']['param_groups']],
             qualification='Standing preserved. Latest saved walking actor has not yet been independently evaluated.')
(DIAG / 'pause_completion.json').write_text(json.dumps(audit, indent=2))
print(json.dumps({k:audit[k] for k in ['all_owned_workers_drained','total_training_iterations','checkpoint','checkpoint_sha256','full_actor_critic_optimizer_saved','qualification']}, indent=2))
