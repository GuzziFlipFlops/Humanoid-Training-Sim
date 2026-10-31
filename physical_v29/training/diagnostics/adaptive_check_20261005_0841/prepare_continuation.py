import json,hashlib,sys,shutil,copy,py_compile
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
pause=read(OUT/'diagnostic_pause.json');old=Path(pause['run']);source=OUT/'all_prior_source/physical_v29'
sys.path.insert(0,str(HERE))
from supervisor_unitree import verify_teacher_context_migration
from curriculum_provenance import training_reward_signature
comparison=read(OUT/'all_prior_comparison.json')
assert all(r['passed'] for kind in ['nominal','pushed'] for r in comparison['arms']['all_prior'][kind] if r['command']<=1.)
assert all(r['survival']==1. for kind in ['nominal','pushed'] for r in comparison['arms']['all_prior'][kind])
assert comparison['full_reload']['passed']
write(OUT/'all_prior_matched_validation.json',dict(passed=True,comparison_sha256=sha(OUT/'all_prior_comparison.json'),
  matched_updates=64,environments=4096,nominal_pushed_robots=192,duration=20,trials_per_command=32,
  stopped_and_all_priors_pass=True,running_qualified=False,diagnostic_actor_deployed=False,
  scope='One bounded matched PPO comparison; future learning stability is unproven.'))
proof=read(OUT/'all_prior_full_reload/full_load_proof.json')
rows=[json.loads(r) for r in (OUT/'all_prior_full_reload/progress.jsonl').read_text().splitlines()]
assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
assert len(rows)==4 and all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
write(OUT/'all_prior_full_state_smoke.json',dict(passed=True,full_load=proof,updates=4,zero_KL_command_mismatch=True,
  full_load_proof_sha256=sha(OUT/'all_prior_full_reload/full_load_proof.json'),progress_sha256=sha(OUT/'all_prior_full_reload/progress.jsonl')))
assert read(OUT/'all_prior_physical_parity.json')['passed'] and read(OUT/'cpu_all_prior_tests.json')['passed']
def inventory(root):return {p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and 'usd' not in p.parts}
a,b=inventory(old/'source/physical_v29'),inventory(source)
changed={k for k in a.keys()|b.keys() if k not in a or k not in b or a[k].read_bytes()!=b[k].read_bytes()}
assert changed=={'training/upper_body_refinement.py','training/teacher_head_context.py','training/teacher_head_context.json'}
write(OUT/'all_prior_source_inventory.json',dict(passed=True,changed=sorted(changed),protected_files=len(a)-1,
  implementation_hashes={k:sha(b[k]) for k in changed},protected_hashes={k:sha(a[k]) for k in a if k not in changed}))
p=dict(schema='posture_conditioned_walking_prior_v1',prior_run=str(old),checkpoint=pause['checkpoint'],
  checkpoint_sha256=pause['checkpoint_sha256'],supervisor_sha256=sha(HERE/'supervisor_unitree.py'),
  implementation_hashes={k:sha(b[k]) for k in changed},reward_signature=training_reward_signature(source/'training'),
  actor_critic_Adam_exploration_retained=True,student_observations_truthful=True,inference_teacher=False,qualifies_running=False,
  conditioned_commands=[.15,.35,.65,1.],same_stage='run_150',same_speed=1.5,
  retained_main_updates=4352,prior_main_updates_preserved=1473,
  reason='Reproduced teacher head-history mismatch; correct only walking teacher targets while retaining current forward-head full state')
for key,name in [('cpu_tests','cpu_all_prior_tests.json'),('physical_validation','all_prior_physical_parity.json'),
 ('matched_validation','all_prior_matched_validation.json'),('full_state_smoke','all_prior_full_state_smoke.json')]:
 p[key]=dict(path=name,sha256=sha(OUT/name))
write(OUT/'teacher_context_continuation_proof.json',p)
fixture=OUT/'migration_cpu_fixture';assert not fixture.exists();(fixture/'source').mkdir(parents=True)
shutil.copytree(old/'source/physical_v29',fixture/'source/physical_v29',ignore=shutil.ignore_patterns('__pycache__','*.pyc','usd'))
state=read(OUT/'state_before_diagnosis.json');state.update(status='superseded_for_diagnostic_repair',child_pid=None)
write(fixture/'state.json',state)
test=copy.deepcopy(p);test['prior_run']=str(fixture);testpath=OUT/'migration_test_proof.json';write(testpath,test)
checks=[]
assert verify_teacher_context_migration(fixture,source,testpath)['checkpoint']==pause['checkpoint'];checks.append('Exact tested source/full checkpoint accepted')
for key,value in [('schema','wrong'),('checkpoint_sha256','0'*64),('supervisor_sha256','0'*64),('reward_signature','0'*64),
 ('student_observations_truthful',False),('inference_teacher',True),('qualifies_running',True),('conditioned_commands',[.35,.65,1.]),('same_stage','stand')]:
 bad=copy.deepcopy(test);bad[key]=value;write(testpath,bad)
 try:verify_teacher_context_migration(fixture,source,testpath)
 except (AssertionError,KeyError):checks.append('Reject '+key)
 else:raise AssertionError('Accepted bad '+key)
write(testpath,test);state['status']='running';write(fixture/'state.json',state)
try:verify_teacher_context_migration(fixture,source,testpath)
except AssertionError:checks.append('Reject live prior state')
else:raise AssertionError('Accepted live prior')
state['status']='superseded_for_diagnostic_repair';write(fixture/'state.json',state)
bad=copy.deepcopy(test);bad['physical_validation']['sha256']='0'*64;write(testpath,bad)
try:verify_teacher_context_migration(fixture,source,testpath)
except AssertionError:checks.append('Reject corrupted evidence hash')
else:raise AssertionError('Accepted corrupt evidence')
write(testpath,test)
for name in ['teacher_head_context.py','upper_body_refinement.py']:
 py_compile.compile(str(source/'training'/name),doraise=True)
py_compile.compile(str(HERE/'supervisor_unitree.py'),doraise=True)
write(OUT/'migration_cpu_tests.json',dict(passed=True,count=len(checks),checks=checks,scope='CPU provenance checks; no production state mutation'))
print(json.dumps(dict(proof=str(OUT/'teacher_context_continuation_proof.json'),changed=sorted(changed),protected_files=len(a)-1,migration_tests=len(checks)),indent=2))
