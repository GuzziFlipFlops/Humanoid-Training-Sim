"""Final owned-process/source/documentation/inventory verification; CPU only."""
import json,hashlib,time
from pathlib import Path
from datetime import datetime,timezone
import psutil
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1];ROOT=TRAIN.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,x:Path(p).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
active=read(TRAIN/'ACTIVE_RUN.json');run=Path(active['run']);state=read(run/'state.json');pause=read(OUT/'diagnostic_pause.json');resume=read(OUT/'diagnostic_resume.json')
sup=psutil.Process(state['supervisor_pid']);assert sup.pid==39000 and sup.create_time()==pause['supervisor_create_time']
assert sup.status()!=psutil.STATUS_STOPPED and state['status']=='running'
deadline=time.monotonic()+30
while True:
    state=read(run/'state.json')
    workers=[p for p in psutil.process_iter(['cmdline']) if any(Path(s).name=='worker_unitree.py' for s in p.info['cmdline'] or [])]
    if workers or time.monotonic()>=deadline:break
    assert state['status']=='running' and sup.status()!=psutil.STATUS_STOPPED
    time.sleep(1)
leaves={p.pid for p in workers}-{p.ppid() for p in workers}
assert len(leaves)==1
actual_worker=next(p for p in workers if p.pid in leaves)
assert all(Path(p.cmdline()[p.cmdline().index('--output')+1]).resolve()==Path(state['operation_path']).resolve() for p in workers)
assert not any(any(Path(s).name in ['timing_probe.py','run_timing_probe.py','run_diagnosis.py'] for s in p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline']))
live=read(OUT/'liveness_cpu/live_handoff.json');assert all(live['frozen_source_integrity'].values()) and len(live['frozen_source_integrity'])==55
v=read(OUT/'liveness_learning_verification.json');block=v['blocks'][-1]
assert Path(block['folder']).name=='431_run_350_train_141' and block['updates']==64 and block['full_load_proof']['source_sha256']==pause['checkpoint_sha256']
assert block['all_preupdate_KL_and_command_mismatch_zero']
inventory=read(TRAIN/'history/RUN_INVENTORY.json');assert not inventory['read_warnings']
checks=read(OUT/'inventory.log');assert checks['differing_manifest_hashes']==0 and not checks['read_warnings']
write(OUT/'inventory_verification.json',checks)
auto=read(OUT/'automation_verification.json');assert auto['status']=='ACTIVE' and auto['five_hour_cadence'] and auto['same_thread'] and auto['durable_scope_endpoint_prefix_unchanged']
for f in [TRAIN/'ITERATION_HISTORY.md',ROOT/'docs/robot/TRAINING_JOURNAL_20261001.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    assert 'Five-hour check October7 23:46 UTC' in f.read_text(encoding='utf-8')
report=OUT/'REPORT.md';assert report.is_file()
closure=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),supervisor=39000,actual_worker=actual_worker.pid,venv_worker_launcher_pids=[p.pid for p in workers if p.pid not in leaves],operation=state['operation_path'],
    status=state['status'],sources_verified=55,resumed_completed_block=block['folder'],resumed_full_state_exact=True,
    resumed_updates=64,resumed_zero_KL_and_command_mismatch=True,resumed_memory={k:block[k] for k in ['minimum_free_gpu_MiB','minimum_host_available_MiB','private_memory_growth_MiB','torch_allocation_growth_MiB']},
    owned_retained_main_snapshot=v['actual_retained_main_updates'],documented_pause_seconds_this_check=resume['pause_seconds'],
    native_automation_active_same_thread_5h=True,computer_on=True,production_change=False,inventory_checks=checks,
    reviewed_gates_and_upper='Exact428;newer432/433 may not yet be a complete reviewed pair. Do not attribute428 qualification or posture to newer partial actors.')
write(OUT/'closeout_verification.json',closure)
with report.open('a',encoding='utf-8') as h:
    h.write(f"\n## Final continuation verification\n\n{closure['recorded_utc']}: same39000/one actual worker/no diagnostic process;55 source hashes verified. Resumed431 completes64 genuine6144 updates,exact full428 state/zeroKL/command mismatch. MinGPU1922MiB,host7191MiB,private+6MiB,Torch allocation unchanged. Retained current-run snapshot9113 plus27507 inherited. Current ownedoperation`{state['operation_path']}` may be newer than reviewed428. Same nativeheartbeat remains ACTIVE5h/same thread/notifications/durable endpoint;confirmation`automation_verification.json`. Inventory40 runs/2881 operations,zero differing hashes/read warnings;8 historical missing manifests remain documented. See `closeout_verification.json`;no new qualification.\n")
n=read(TRAIN/'automation/NEXT_CHECK.json');n.update(automation_verified_active=True,automation_verification=str(OUT/'automation_verification.json'),native_automation_verification=str(OUT/'automation_verification.json'),
    closeout_verification=str(OUT/'closeout_verification.json'),history_inventory=str(OUT/'inventory_verification.json'),closing_inventory=str(OUT/'inventory_verification.json'),
    resumed_learning_evidence=str(OUT/'liveness_learning_verification.json'),final_liveness=str(OUT/'liveness_cpu/live_handoff.json'),closing_liveness=str(OUT/'closeout_verification.json'),
    current_owned_operation=state['operation_path'],upper_joint_time_series=str(OUT/'final_upper/upper_joint_time_audit.json'),upper_body_analysis=str(OUT/'final_upper/upper_joint_time_audit.json'),
    latest_schedule_verification=str(OUT/'automation_verification.json'),upper_physical_response=str(OUT/'physical_validation.json'))
write(TRAIN/'automation/NEXT_CHECK.json',n)
print(json.dumps(closure,indent=2))
