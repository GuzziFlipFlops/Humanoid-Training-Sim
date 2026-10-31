"""Read-only final provenance, owned-process, source and documentation checks."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import tomllib
import psutil

here=Path(__file__).resolve().parent
training=here.parents[1]
sys.path.insert(0,str(training))
import supervisor_unitree
pointer=json.loads((training/'ACTIVE_RUN.json').read_text())
run=Path(pointer['run']);state=json.loads((run/'state.json').read_text())
assert state['status']=='running'
assert pointer['supervisor_pid']==state['supervisor_pid']
assert str(run) in ' '.join(psutil.Process(state['supervisor_pid']).cmdline())
proof=json.loads((here/'continuation_proof.json').read_text())
prior=Path(proof['prior_run'])
supervisor_unitree.verify_resource_migration(prior,run/'source/physical_v29',str(here/'continuation_proof.json'))
config=json.loads((run/'chain_config.json').read_text())
matched=0
for relative,digest in config['sources_sha256'].items():
    if relative.startswith('physical_assets/'):
        path=run/'source/physical_v29/assets'/relative.removeprefix('physical_assets/')
    elif relative.startswith('physical_v29/'):
        path=run/'source/physical_v29'/relative.removeprefix('physical_v29/')
    elif relative in ('supervisor_unitree.py','vram_ramp.py'):
        path=run/relative
    else:path=run/'source/physical_v29/training'/relative
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,relative
    matched+=1
assert config['reserve_mb']==300 and config['automatic_upward_ramp']['hold_seconds']==180
assert config['automatic_upward_ramp']['max_envs']==8192
assert not config['actor_warm_start']
automation=Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml')
assert tomllib.loads(automation.read_text(encoding='utf-8'))['status']=='PAUSED'
linked=0
for path in [here/'REPORT.md',here/'README.md',training/'ITERATION_HISTORY.md']:
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',path.read_text()):
        if '://' in target or target.startswith('#'):continue
        link=target.split('#')[0]
        assert (path.parent/link).exists(),(path,target)
        linked+=1
workers=[]
for proc in psutil.process_iter(['pid','ppid','name','cmdline']):
    args=proc.info['cmdline'] or []
    if any(Path(s).name=='worker_unitree.py' for s in args):
        assert str(run) in ' '.join(args),proc.info
        workers.append(proc.info)
assert len(workers)<=2,'At most one venv worker launcher and its actual interpreter'
record=dict(passed=True,recorded_utc=datetime.now(timezone.utc).isoformat(),run=str(run),
            fresh_status=state['status'],operation=state.get('operation'),environments=state.get('environment_count'),
            frozen_hashes_matched=matched,relative_links_checked=linked,proof_verified=True,
            full_checkpoint_continuation=True,automation_paused=True,owned_workers=workers,
            current_gpu=state.get('gpu'),current_host=state.get('system_memory'))
(here/'HANDOFF_VERIFICATION.json').write_text(json.dumps(record,indent=2))
print(json.dumps({k:v for k,v in record.items() if k!='owned_workers'},indent=2))
