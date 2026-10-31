"""Retire suspended original supervisor after healthy replacement full-load startup."""
import hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pause=read(OUT/'diagnostic_pause.json');deployment=read(OUT/'deployment.json');root=Path(deployment['new_run'])
s=read(root/'state.json');assert s['status']=='running'
new=psutil.Process(s['supervisor_pid']);assert any(Path(x).name=='supervisor_unitree.py' for x in new.cmdline())
assert Path(new.cmdline()[new.cmdline().index('--output')+1]).resolve()==root.resolve()
config=read(root/'chain_config.json');verified={}
for key,expected in config['sources_sha256'].items():
 if key in ('supervisor_unitree.py','vram_ramp.py'):p=root/key
 elif key.startswith('physical_assets/'):p=root/'source/physical_v29/assets'/key.removeprefix('physical_assets/')
 elif key.startswith('physical_v29/'):p=root/'source/physical_v29'/key.removeprefix('physical_v29/')
 else:p=root/'source/physical_v29/training'/key
 verified[key]=sha(p)==expected
assert all(verified.values())
full=read(root/'002_resource_ppo_smoke/full_load_proof.json')
assert all(full[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
assert full['source_sha256']==pause['checkpoint_sha256']
assert read(root/'001_physics_preflight/result.json')['passed']
assert read(root/'003_checkpoint_reload/result.json')['completed']
old=psutil.Process(pause['supervisor_pid']);assert old.create_time()==pause['supervisor_create_time'] and old.status()==psutil.STATUS_STOPPED
oldstate=read(Path(pause['run'])/'state.json');assert oldstate['child_pid'] is None
assert oldstate['checkpoint']==pause['checkpoint'] and sha(oldstate['checkpoint'])==pause['checkpoint_sha256']
old.terminate();old.wait(timeout=10)
sys_now=time.time();oldstate.update(status='superseded_for_commanded_travel_guard',superseded_by=str(root),
 superseded_unix=sys_now,child_pid=None,termination_guard_proof=str(OUT/'termination_continuation_proof.json'))
import sys;sys.path.insert(0,str(HERE));from status_io import atomic_json
atomic_json(Path(pause['run'])/'state.json',oldstate)
active=read(HERE/'ACTIVE_RUN.json');active.update(status='running',last_verified_utc=datetime.now(timezone.utc).isoformat(),
 supervisor_pid=s['supervisor_pid'],launcher_pid=deployment['new_supervisor']['pid'])
atomic_json(HERE/'ACTIVE_RUN.json',active)
start=read(OUT/'throughput_and_loads.json')['measurement_endpoint_unix']
receipt=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),old_supervisor_retired=True,old_pid=pause['supervisor_pid'],
 new_supervisor_pid=s['supervisor_pid'],source_hashes_verified=len(verified),all_hashes_match=True,
 source_freeze_handoff_unix=read(root/'state.json')['started_unix'],
 old_saved_boundary_pause_start_unix=start,new_start_unix=s['started_unix'],
 old_learning_to_new_supervisor_gap_seconds=s['started_unix']-start,
 old_paused_until_retirement_seconds=sys_now-start,
 healthy_new_full_checkpoint_startup=True,diagnostic_actors_deployed=False,
 scope='New run startup/capacity clocks are separate. No main learning or3.5 qualification inferred from these startup receipts.')
atomic_json(OUT/'handoff_completion.json',receipt)
deployment.update(old_supervisor_retired=True,actual_new_supervisor_pid=s['supervisor_pid'],startup_integrity=receipt)
atomic_json(OUT/'deployment.json',deployment)
print(json.dumps(receipt,indent=2))
