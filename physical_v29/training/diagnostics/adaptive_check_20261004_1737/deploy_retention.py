"""Deploy validated training-only sources and continue a preserved full actor."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,sys,time
import psutil
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1];ROOT=HERE.parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
pause=read(OUT/'diagnostic_pause.json');old=Path(pause['run']);proof=read(OUT/'retention_continuation_proof.json')
assert read(OUT/'cpu_tests.json')['passed'] and read(OUT/'physical_parity.json')['cases_bitwise_equal']
smoke=OUT/'retention_fullstate_smoke_02';r=read(smoke/'result.json');assert r['completed'] and r['iterations']==4
loaded=read(smoke/'full_load_proof.json');assert all(loaded[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
rows=[json.loads(x) for x in (smoke/'progress.jsonl').read_text().splitlines()]
assert len(rows)==4 and all(x['ppo']['pre_update_kl']==0. and x['behavior']['action_command_difference']==0. and x['locomotion_retention']['batches']>0 for x in rows)
reload=read(OUT/'retention_checkpoint_reload/result.json');assert reload['completed'] and all(c['statistics']['survival']==1. for c in reload['cases'])
supervisor=psutil.Process(pause['supervisor_pid'])
assert supervisor.create_time()==pause['supervisor_create_time'] and supervisor.status()==psutil.STATUS_STOPPED
for p in psutil.process_iter(['pid','cmdline']):
 assert not any(Path(a).name=='worker_unitree.py' for a in p.info['cmdline'] or []),'Another physical worker is active'
state=read(old/'state.json');assert state['checkpoint']==pause['checkpoint'] and sha(Path(state['checkpoint']))==proof['drained_checkpoint_sha256']
write(OUT/'state_preserved_before_supersede.json',state)
supervisor.kill();supervisor.wait(timeout=10)
state.update(status='superseded_for_diagnostic_repair',child_pid=None,updated_unix=time.time(),
 pause_reason='Preserved final candidate1352; continue stronger running checkpoint269 with qualified walking gradient retention')
write(old/'state.json',state)
new=OUT/'validation_source/physical_v29'
backup=OUT/'canonical_before';backup.mkdir(exist_ok=True)
files={'standing_retention.py':new/'training/standing_retention.py','worker_unitree.py':new/'training/worker_unitree.py',
       'supervisor_unitree.py':OUT/'candidate_supervisor_unitree.py'}
changes={}
for name,source in files.items():
 target=HERE/name;expected=old/name if name=='supervisor_unitree.py' else old/'source/physical_v29/training'/name
 assert target.read_bytes()==expected.read_bytes(),name
 shutil.copy2(target,backup/name);changes[name]=dict(before=sha(target),after=sha(source));shutil.copy2(source,target)
sys.path.insert(0,str(HERE))
from standing_retention import verify_retention_migration
verified=verify_retention_migration(old,new,OUT/'retention_continuation_proof.json')
assert verified['checkpoint']==proof['selected_checkpoint']
run=HERE/'runs/locomotion_20261004_r33_qualified_walking_retention'
assert not run.exists(),'Preserve all prior runs'
run.mkdir()
argv=[sys.executable,'-u','-X','utf8',str(HERE/'supervisor_unitree.py'),'--output',str(run),
 '--reserve_mb','300','--max_hours','18','--iterations_per_block','64','--train_envs','8192',
 '--learning_rate','0.0001','--migrate_run',str(old),'--initial_checkpoint',proof['selected_checkpoint'],
 '--retention_proof',str(OUT/'retention_continuation_proof.json')]
write(OUT/'launch_args.json',dict(argv=argv))
with (run/'supervisor_stdout.log').open('w') as stdout,(run/'supervisor_stderr.log').open('w') as stderr:
 process=subprocess.Popen(argv,stdout=stdout,stderr=stderr,cwd=str(ROOT),creationflags=subprocess.CREATE_NO_WINDOW)
write(HERE/'ACTIVE_RUN.json',dict(run=str(run),status='starting',recorded_utc=datetime.now(timezone.utc).isoformat(),
 prior_run=str(old),selected_full_checkpoint=proof['selected_checkpoint'],reason=proof['reason']))
deployment=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),run=str(run),supervisor_launcher_pid=process.pid,
 old_run=str(old),old_final_checkpoint=pause['checkpoint'],old_final_updates=state['total_training_iterations'],
 selected_checkpoint=proof['selected_checkpoint'],selected_updates=proof['selected_main_updates'],changes=changes,
 full_load_proof=loaded,smoke_pre_update_KL=[x['ppo']['pre_update_kl'] for x in rows],
 current_and_prior_qualification_required=True,teacher_in_inference=False,learning_rate_ceiling=.0001)
write(OUT/'deployment.json',deployment);print(json.dumps(deployment,indent=2))
