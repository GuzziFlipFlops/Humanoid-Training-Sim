"""Stage a strictly proved full-state termination-only continuation supervisor."""
import hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
proof=read(OUT/'termination_load_proof.json')
source=OUT/'canonical_before_guard/supervisor_unitree.py' if (OUT/'canonical_before_guard/supervisor_unitree.py').exists() else HERE/'supervisor_unitree.py'
old=source.read_text();new=old
token="def verify_resource_migration(prior,new_source,optimization_proof='',curriculum_proof='',retention_proof=''):\n"
insert=" if optimization_proof and json.loads(Path(optimization_proof).read_text()).get('schema')=='commanded_travel_guard_v1':\n  assert not retention_proof and not curriculum_proof\n  return verify_commanded_travel_migration(prior,new_source,optimization_proof)\n"
anchor=' """Only maintenance changes may continue a candidate and its qualification."""\n'
assert new.count(anchor)==1;new=new.replace(anchor,anchor+insert)
function='''def verify_commanded_travel_migration(prior,new_source,proof_path):
 """Only the tested termination budget/signature bridge; requalify all priors."""
 from curriculum_provenance import training_reward_signature
 prior=Path(prior);new_source=Path(new_source);proof_path=Path(proof_path).resolve()
 proof=json.loads(proof_path.read_text());state=json.loads((prior/'state.json').read_text())
 assert proof['schema']=='commanded_travel_guard_v1' and not proof['qualifies_locomotion']
 assert state['status'] in ('paused_for_controlled_diagnosis','superseded_for_commanded_travel_guard') and state.get('child_pid') is None
 assert Path(proof['prior_run']).resolve()==prior.resolve()
 assert Path(proof['checkpoint']).resolve()==Path(state['checkpoint']).resolve()
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 assert sha(state['checkpoint'])==proof['checkpoint_sha256']
 assert set(proof['allowed_checkpoint_sha256'])=={sha(state['checkpoint']),sha(state['promotions'][-1]['checkpoint'])}
 assert sha(__file__)==proof['supervisor_sha256']
 def inventory(root):
  return {p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file()
   and '__pycache__' not in p.parts and p.suffix!='.pyc' and 'usd' not in p.parts}
 old,new=inventory(prior/'source/physical_v29'),inventory(new_source)
 assert old.keys()==new.keys()
 changed={k for k in old if old[k].read_bytes()!=new[k].read_bytes()}
 assert changed=={'training/environment.py','training/worker_unitree.py'},changed
 assert {k:sha(v) for k,v in new.items()}==proof['source_hashes']
 assert proof['environment_replacements']==EXPECTED_TRAVEL_ENV_REPLACEMENTS
 assert proof['worker_replacements']==EXPECTED_TRAVEL_WORKER_REPLACEMENTS
 for name,key in [('training/environment.py','environment_replacements'),('training/worker_unitree.py','worker_replacements')]:
  data=new[name].read_bytes();nl='\\r\\n' if b'\\r\\n' in data else '\\n'
  for before,after in reversed(proof[key]):
   a=after.replace('\\n',nl).encode();b=before.replace('\\n',nl).encode()
   assert data.count(a)==1;data=data.replace(a,b)
  assert data==old[name].read_bytes(),name
 assert training_reward_signature(prior/'source/physical_v29/training')==proof['old_reward_signature']
 assert training_reward_signature(new_source/'training')==proof['new_reward_signature']
 for key in ['cpu_tests','physical_validation','physical_parity','full_state_smoke']:
  entry=proof[key];path=(proof_path.parent/entry['path']).resolve()
  assert path.is_relative_to(proof_path.parent) and sha(path)==entry['sha256']
  assert json.loads(path.read_text())['passed'],key
 assert state['total_training_iterations']==proof['completed_operation_main_updates']
 assert proof['verified_recovered_main_updates']==58
 state=json.loads(json.dumps(state))
 state['preserved_completed_operation_main_updates']=state['total_training_iterations']
 state['total_training_iterations']+=proof['verified_recovered_main_updates']
 state['termination_guard_requalification_required']=True
 return state

'''
function='EXPECTED_TRAVEL_ENV_REPLACEMENTS = '+repr(proof['environment_replacements'])+'\nEXPECTED_TRAVEL_WORKER_REPLACEMENTS = '+repr(proof['worker_replacements'])+'\n\n'+function
assert new.count(token)==1;new=new.replace(token,function+token)
anchor="  if checkpoint:\n   command+=['--checkpoint',str(checkpoint)]\n"
insert="  if self.optimization_proof and json.loads(Path(self.optimization_proof).read_text()).get('schema')=='commanded_travel_guard_v1':\n   command+=['--termination_proof',self.optimization_proof]\n"
assert new.count(anchor)==1;new=new.replace(anchor,insert+anchor)
anchor="    prior_run=Path(prior['checkpoint']).parent.parent\n"
replacement=anchor+"    if self.optimization_proof and json.loads(Path(self.optimization_proof).read_text()).get('schema')=='commanded_travel_guard_v1':prior_run=self.root\n"
assert new.count(anchor)==1;new=new.replace(anchor,replacement)
anchor="   if self.retention_proof:\n    baseline=previous['promotions'][-1]\n"
replacement="   termination_requalification=bool(self.optimization_proof and json.loads(Path(self.optimization_proof).read_text()).get('schema')=='commanded_travel_guard_v1')\n   if self.retention_proof or termination_requalification:\n    baseline=previous['promotions'][-1]\n"
assert new.count(anchor)==1;new=new.replace(anchor,replacement)
anchor="   self.state['promotions']=previous['promotions'][:]\n   trusted=previous['promotions'][-1]['checkpoint'] if previous['promotions'] else None\n"
replacement=anchor+"   if termination_requalification:atomic(self.root/'termination_requalification.json',dict(checkpoint=trusted,all_prior_gates_fresh=True,no_old_qualification_inferred=True,termination_only=True))\n"
assert new.count(anchor)==1;new=new.replace(anchor,replacement)
# Label the continuation accurately rather than silently calling changed dones
# identical-runtime resource maintenance.
anchor="('training-only exact-command rehearsal, original reward/observation/action signature and prior gates preserved; ' if self.curriculum_proof else 'qualified prior-command gradient repair; ' if self.retention_proof else 'resource maintenance; ')"
replacement="('commanded-travel termination correction, numerical reward/action/history/motor contract unchanged; ' if termination_requalification else 'training-only exact-command rehearsal, original reward/observation/action signature and prior gates preserved; ' if self.curriculum_proof else 'qualified prior-command gradient repair; ' if self.retention_proof else 'resource maintenance; ')"
assert new.count(anchor)==1;new=new.replace(anchor,replacement)
anchor="    reports=self.gates(baseline['speed'],baseline['checkpoint'],'retention_teacher_requalification')\n"
replacement="    teacher_label='run_300_gate_termination_teacher_requalification' if termination_requalification else 'retention_teacher_requalification'\n    reports=self.gates(baseline['speed'],baseline['checkpoint'],teacher_label)\n"
assert new.count(anchor)==1;new=new.replace(anchor,replacement)
path=OUT/'tested_supervisor_unitree.py';path.write_text(new);compile(new,str(path),'exec')
(OUT/'supervisor_patch.json').write_text(json.dumps({'before_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'after_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
 'changes':'Strict termination-only migration/schema, explicit worker proof, full learner retention including58 recovered updates, fresh3m teacher/all-prior gates, fresh certificates from new run. No classifier/curriculum/optimizer change.'},indent=2)+'\n')
print('Prepared tested supervisor; not deployed')
