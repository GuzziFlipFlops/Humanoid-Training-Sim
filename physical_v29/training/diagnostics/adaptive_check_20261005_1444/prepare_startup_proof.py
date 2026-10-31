import json,hashlib,sys,importlib.util,shutil
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
evidence=read(OUT/'initial_evidence.json');run=Path(evidence['active']['run']);source=OUT/'offline_source/physical_v29'
rows=[json.loads(s) for s in (OUT/'gradient_smoke/progress.jsonl').read_text().splitlines()]
load=read(OUT/'gradient_smoke/full_load_proof.json')
assert all(load[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
assert len(rows)==4 and all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
smoke=dict(passed=True,full_load=load,updates=4,source_checkpoint_sha256=evidence['checkpoint_sha256'],
 zero_pre_update_KL_command_difference=True,diagnostic_actor_deployed=False,
 minimum_GPU_MiB=min(r['gpu']['free_mb'] for r in rows),minimum_host_MiB=min(r['gpu']['ram_available_mb'] for r in rows))
(OUT/'offline_full_state_smoke.json').write_text(json.dumps(smoke,indent=2)+'\n')
def inventory(root):return {p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'usd' not in p.parts}
old,new=inventory(run/'source/physical_v29'),inventory(source)
changed={k for k in old.keys()|new.keys() if k not in old or k not in new or old[k].read_bytes()!=new[k].read_bytes()}
sys.path.insert(0,str(HERE));from curriculum_provenance import training_reward_signature
record=dict(schema='verified_offline_ground_v1',prior_run=str(run),checkpoint=evidence['checkpoint'],
 checkpoint_sha256=evidence['checkpoint_sha256'],supervisor_sha256=sha(OUT/'tested_supervisor_unitree.py'),
 implementation_hashes={k:sha(new[k]) for k in sorted(changed)},reward_signature=training_reward_signature(source/'training'),
 actor_critic_Adam_exploration_retained=True,training_gradient_repair=False,qualifies_running=False,
 same_stage='run_150',reason='Verified original NVIDIA ground cached locally after reproduced online resolver startup failure',
 tests={},cpu_tests={},physical_validation={},full_state_smoke={})
for key,name in [('cpu_tests','offline_ground_cpu_tests.json'),('physical_validation','ground_physical_parity.json'),('full_state_smoke','offline_full_state_smoke.json')]:
 assert read(OUT/name)['passed'];record[key]=dict(path=name,sha256=sha(OUT/name))
path=OUT/'offline_ground_continuation_proof.json';path.write_text(json.dumps(record,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('tested_offline_supervisor',OUT/'tested_supervisor_unitree.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
verified=module.verify_resource_migration(run,source,optimization_proof=str(path))
assert verified['checkpoint']==evidence['checkpoint']
checks=['exact_checkpoint_and_source_accepted','unchanged_reward_signature','same_actor_critic_Adam_exploration','all_7_startup_only_changes_pinned']
fixture=OUT/'migration_negative_source';assert not fixture.exists();shutil.copytree(source,fixture,ignore=shutil.ignore_patterns('__pycache__','*.pyc','usd'))
for name in ['training/stage_rewards.py','cad_differential.py','assets/actuator_limits.json','training/upper_body_refinement.py','training/worker_unitree.py','training/assets/ground/default_environment.usd']:
 p=fixture/name;data=p.read_bytes();p.write_bytes(data+b'\n# negative test modification')
 try:module.verify_resource_migration(run,fixture,optimization_proof=str(path))
 except AssertionError:checks.append('reject_changed_'+name)
 else:raise AssertionError(('Bad source accepted',name))
 p.write_bytes(data)
module.verify_resource_migration(run,fixture,optimization_proof=str(path));checks.append('exact_fixture_restoration_accepted')
(OUT/'offline_migration_cpu_tests.json').write_text(json.dumps(dict(passed=True,checks=checks,protected_original_files=len(old)-1,
 startup_only_changed_files=sorted(changed),proof_sha256=sha(path)),indent=2)+'\n')
print(json.dumps(dict(passed=True,checks=len(checks),runtime_changes=sorted(changed)),indent=2))
