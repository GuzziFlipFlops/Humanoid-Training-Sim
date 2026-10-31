"""Deploy the tested narrow objective repair to a fresh source-frozen chain."""
import hashlib,json,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1];ROOT=HERE.parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2)+'\n');temp.replace(p)
for name in ['bilateral_counterfactuals.json','candidate_cycle_tests.json','candidate_proof.json','validation_proof.json','physical_objective_audit.json','gradient_analysis.json']:
    assert read(OUT/name)['passed'],name
pause=read(OUT/'diagnostic_pause.json');old=Path(pause['run']);seed=read(OUT/'selected_seed.json')
assert sha(Path(seed['checkpoint']))==seed['sha256']
assert read(OUT/'validation_proof.json')['same_actor_cases_bitwise_equal']
supervisor=psutil.Process(pause['supervisor_pid'])
assert supervisor.create_time()==pause['supervisor_create_time'] and supervisor.status()==psutil.STATUS_STOPPED
for p in psutil.process_iter(['cmdline']):
    assert not any(Path(a).name in ['worker_unitree.py','physical_policy_comparison.py','capture_gradient.py'] for a in p.info['cmdline'] or []),'Another actual worker is live'
state=read(old/'state.json');assert state['checkpoint']==pause['checkpoint'] and sha(Path(state['checkpoint']))==pause['checkpoint_sha256']
write(OUT/'state_preserved_before_supersede.json',state)
staged=OUT/'validation_source/physical_v29';backup=OUT/'canonical_before';backup.mkdir()
changes={}
for name in ['stage_rewards.py','reward_warm_start.py']:
    target=HERE/name;before=old/'source/physical_v29/training'/name
    assert target.read_bytes()==before.read_bytes(),name
    shutil.copy2(target,backup/name);changes[name]={'before':sha(target),'after':sha(staged/'training'/name)}
supervisor.kill();supervisor.wait(timeout=10)
state.update(status='superseded_for_diagnostic_repair',child_pid=None,updated_unix=time.time(),
    pause_reason='1401 new main updates preserved; reproduced intended-flight load redistribution incentive; fresh bilateral-unloading objective with unchanged physics and walking retention')
write(old/'state.json',state)
for name in changes:
    shutil.copy2(staged/'training'/name,HERE/name);assert sha(HERE/name)==changes[name]['after']
run=HERE/'runs/locomotion_20261004_r34_bilateral_swing_unloading';assert not run.exists();run.mkdir()
args=[sys.executable,'-u','-X','utf8',str(HERE/'supervisor_unitree.py'),'--output',str(run),
    '--reserve_mb','300','--max_hours','18','--iterations_per_block','64','--train_envs','8192',
    '--learning_rate','0.0001','--initial_checkpoint',seed['checkpoint'],'--actor_warm_start']
write(OUT/'launch_args.json',{'argv':args})
with (run/'supervisor_stdout.log').open('w') as stdout,(run/'supervisor_stderr.log').open('w') as stderr:
    proc=subprocess.Popen(args,stdout=stdout,stderr=stderr,cwd=str(ROOT),creationflags=subprocess.CREATE_NO_WINDOW)
write(HERE/'ACTIVE_RUN.json',{'run':str(run),'status':'starting','recorded_utc':datetime.now(timezone.utc).isoformat(),
    'prior_run':str(old),'selected_checkpoint':seed['checkpoint'],'selected_sha256':seed['sha256'],
    'initial_actor_checkpoint':seed['checkpoint'],'initial_actor_sha256':seed['sha256'],
    'reason':seed['reason'],'repair_report':str(OUT/'REPORT.md'),'optimization_iteration':'04'})
r={'recorded_utc':datetime.now(timezone.utc).isoformat(),'run':str(run),'supervisor_launcher_pid':proc.pid,
    'old_run':str(old),'old_final_checkpoint':pause['checkpoint'],'old_final_updates':state['total_training_iterations'],
    'selected_checkpoint':seed['checkpoint'],'selected_sha256':seed['sha256'],'changes':changes,
    'actor_and_trained_exploration_preserved':True,'critic_and_Adam_reset_for_changed_objective':True,
    'qualification_reused':False,'current_and_all_prior_qualification_required':True,'teacher_in_inference':False,
    'target_envs':8192,'vram_reserve_mb':300,'healthy_memory_hold_s':180,'physical_and_gate_changes':False}
write(OUT/'deployment.json',r);print(json.dumps(r,indent=2))
