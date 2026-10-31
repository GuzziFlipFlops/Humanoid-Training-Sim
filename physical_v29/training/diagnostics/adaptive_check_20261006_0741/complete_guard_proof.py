"""Bind tested full-state continuation to exact sources, checkpoints and evidence."""
import copy,hashlib,importlib.util,json,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=read(OUT/'termination_load_proof.json');counts=read(OUT/'throughput_and_loads.json')
p.update(supervisor_sha256=sha(OUT/'tested_supervisor_unitree.py'),
 completed_operation_main_updates=counts['completed_operation_main_updates'],
 verified_recovered_main_updates=counts['verified_recovered_main_updates'],
 preserved_inherited_main_updates=counts['inherited_main_updates'],
 actual_preserved_total_main_updates=counts['actual_retained_main_updates']+counts['inherited_main_updates'],
 recovered_evidence=counts['recovered_evidence'],
 pause_started_unix=counts['measurement_endpoint_unix'],
 smoke_actors_deployed=False,full_actor_critic_Adam_exploration_continuation=True)
for key,name in [('cpu_tests','travel_guard_cpu_tests.json'),('physical_validation','physical_validation.json'),
                 ('physical_parity','physical_parity.json'),('full_state_smoke','physical_full_load_smoke.json')]:
 assert read(OUT/name)['passed'];p[key]=dict(path=name,sha256=sha(OUT/name))
path=OUT/'termination_continuation_proof.json';path.write_text(json.dumps(p,indent=2)+'\n')
sys.path.insert(0,str(HERE));spec=importlib.util.spec_from_file_location('tested_guard_supervisor',OUT/'tested_supervisor_unitree.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
root=OUT/'candidate_source/physical_v29'
state=module.verify_commanded_travel_migration(p['prior_run'],root,path)
assert state['total_training_iterations']==counts['actual_retained_main_updates']
assert state['checkpoint']==p['checkpoint'] and state['promotions'][-1]['speed']==3.
fixtures=[]
def reject(label,edit):
 q=copy.deepcopy(p);edit(q);file=OUT/('rejected_proof_'+label+'.json');file.write_text(json.dumps(q,indent=2)+'\n')
 try:module.verify_commanded_travel_migration(p['prior_run'],root,file)
 except (AssertionError,KeyError,FileNotFoundError):fixtures.append(dict(case=label,rejected=True));return
 raise AssertionError('Unsafe migration accepted:'+label)
reject('wrong_schema',lambda q:q.update(schema='generic'))
reject('qualification_claim',lambda q:q.update(qualifies_locomotion=True))
reject('wrong_checkpoint_hash',lambda q:q.update(checkpoint_sha256='0'*64))
reject('extra_checkpoint_allowed',lambda q:q['allowed_checkpoint_sha256'].append('0'*64))
reject('wrong_supervisor',lambda q:q.update(supervisor_sha256='0'*64))
reject('changed_gate',lambda q:q['source_hashes'].update({'training/evaluation.py':'0'*64}))
reject('changed_reward',lambda q:q['source_hashes'].update({'training/stage_rewards.py':'0'*64}))
reject('changed_geometry',lambda q:q['source_hashes'].update({'assets/differential_geometry.json':'0'*64}))
reject('different_guard_patch',lambda q:q['environment_replacements'].append(['x','y']))
reject('different_load_patch',lambda q:q['worker_replacements'].append(['x','y']))
reject('old_signature',lambda q:q.update(old_reward_signature='0'*64))
reject('new_signature',lambda q:q.update(new_reward_signature='0'*64))
reject('evidence_hash',lambda q:q['physical_validation'].update(sha256='0'*64))
reject('incomplete_update_count',lambda q:q.update(completed_operation_main_updates=9516))
reject('lost_recovery',lambda q:q.update(verified_recovered_main_updates=0))
r=dict(passed=True,positive_exact_full_state_test=True,negative_fixtures=fixtures,
 source_change_scope=['training/environment.py','training/worker_unitree.py'],
 preserves_actual_main_updates=p['actual_preserved_total_main_updates'],smoke_actors_deployed=False)
(OUT/'migration_cpu_tests.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
