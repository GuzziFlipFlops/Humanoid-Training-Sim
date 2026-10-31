"""Freeze the narrow training-update change; leave R40 immutable and running."""
import hashlib,json,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1]
OLD=TRAIN/'diagnostics/adaptive_check_20261010_0149'
RUN=TRAIN/'runs/locomotion_20261010_r41_command_conditioned_advantage'
SRC=TRAIN/'runs/locomotion_20261007_r40_goal_budget_full_state/source/physical_v29'
DST=RUN/'source/physical_v29'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert not RUN.exists(),'Never overwrite an earlier frozen iteration'
shutil.copytree(SRC,DST,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
worker=DST/'training/worker_unitree.py';original=worker.read_text(encoding='utf-8')
patches=[
 ("parser.add_argument('--actor_warm_start',action='store_true')", "parser.add_argument('--actor_warm_start',action='store_true')\nparser.add_argument('--command_advantage',action='store_true')"),
 ('from ppo_update_audit import PpoUpdateAudit', 'from ppo_update_audit import PpoUpdateAudit\nfrom command_advantage import install as install_command_advantage, VERSION as COMMAND_ADVANTAGE_VERSION'),
 ("   details['training_command_version']=TRAINING_COMMAND_VERSION", "   details['training_command_version']=TRAINING_COMMAND_VERSION\n   details['training_update_version']=COMMAND_ADVANTAGE_VERSION if args.command_advantage else 'original_global_advantage_ppo'\n   details['training_update_signature']=sha(HERE/'command_advantage.py')\n   details['command_advantage_applied']=bool(args.command_advantage)\n   details['training_update_manifest']=str(HERE.parents[2]/'training_update_manifest.json')"),
 ('   update_audit=PpoUpdateAudit(runner.alg)', '   update_audit=PpoUpdateAudit(runner.alg)\n   command_advantage=install_command_advantage(runner.alg,out,args.command_advantage)'),
 ("    state['command_coverage']=command_coverage(envraw.commands,args.speed)", "    state['command_coverage']=command_coverage(envraw.commands,args.speed)\n    state['command_advantage']=command_advantage.latest")]
data=original
for a,b in patches:
 assert data.count(a)==1,a
 data=data.replace(a,b)
inverse=data
for a,b in reversed(patches):
 assert inverse.count(b)==1
 inverse=inverse.replace(b,a)
assert inverse==original
worker.write_text(data,encoding='utf-8',newline='\n')
shutil.copy2(OUT/'command_advantage.py',DST/'training/command_advantage.py')
def hashes(root):
 return {p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and 'usd' not in p.parts}
old,new=hashes(SRC),hashes(DST)
changed={n for n in old.keys()|new.keys() if old.get(n)!=new.get(n)}
assert changed=={'training/worker_unitree.py','training/command_advantage.py'},changed
sys.path.insert(0,str(DST/'training'))
import command_advantage
test=command_advantage.cpu_tests();(OUT/'CPU_tests.json').write_text(json.dumps(test,indent=2)+'\n')
assert test['passed']
proof_path=TRAIN/'diagnostics/adaptive_check_20261006_0741/termination_continuation_proof_retry.json'
proof=read(proof_path)
assert proof['source_hashes']==old,'Original teacher migration source must match exactly'
proof['source_hashes']=new
proof['training_update_extension']=dict(version=command_advantage.VERSION,original_proof=str(proof_path),original_proof_sha256=sha(proof_path),
 allowed_changed_files=sorted(changed),literal_inverse_worker_patches=True,all_other_authored_files_identical=True,
 reward_and_inference_signature_unchanged=True,original_reward_signature_does_not_certify_changed_training_objective=True,
 actor_critic_Adam_exploration_and_teachers_preserved=True)
(OUT/'termination_teacher_update_proof.json').write_text(json.dumps(proof,indent=2)+'\n')
manifest=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),schema='command_conditioned_advantage_iteration_v1',
 run=str(RUN),prior_source=str(SRC),source=str(DST),source_hashes=new,original_source_hashes=old,
 changed_files=sorted(changed),literal_inverse_patches=patches,CPU_tests=test,
 algorithm_module_sha256=sha(DST/'training/command_advantage.py'),
 RSL_PPO_sha256=sha(Path(sys.executable).parents[1]/'Lib/site-packages/rsl_rl/algorithms/ppo.py'),
 source_freeze=True,reward_v8_and_GAE_and_critic_targets_unchanged=True,gate_v6_unchanged=True,
 inference_and_native_physics_caps_unchanged=True,one_actual_CUDA_worker=True,
 scope='Substantial task-conditioned PPO optimization scale. Not CPO/PCGrad, not original mixed-objective unbiasedness, not retention certification.',
 user_final_stop_utc='2026-10-10T09:03:00+00:00',training_seeds=[347,359],training_environments=4096,check_updates=[64,128],
 ordinary_seeds=[901,1301],independent_initial_states_seeds=[5011,5039],
 acceptance='Both learning seeds must improve persistent failures without losing any case the matched control passes; every full gate still required for qualification. Reject material head/gait degradation.',
 experimental_actors_not_deployed=True)
(RUN/'training_update_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(OUT/'source_freeze.json').write_text(json.dumps(manifest,indent=2)+'\n')
entry="import runpy,sys\nfrom pathlib import Path\nTRAIN=Path("+repr(str(DST/'training'))+")\nsys.path[:0]=[str(TRAIN),str(TRAIN.parent)]\nrunpy.run_path(str(TRAIN/'worker_unitree.py'),run_name='__main__')\n"
(OUT/'worker_entry.py').write_text(entry)
diverse=(OLD/'diverse_eval.py').read_text().replace("OUT=Path(__file__).resolve().parent;TRAIN=OUT/'control_source/physical_v29/training'", "OUT=Path(__file__).resolve().parent;TRAIN=Path("+repr(str(DST/'training'))+")")
(OUT/'diverse_eval.py').write_text(diverse)
for name in ('prepare_and_pause.py','resume_after_diagnosis.py'):
 text=(OLD/name).read_text()
 if name=='prepare_and_pause.py':
  text=text.replace("scope='Bounded complete-state continuation of earlier neutral-head267 versus unchanged current full state, two seeds and all native gates; no training-source/reward/physics modification or composite inference'", "scope='Versioned per-command PPO advantage normalization, exact current full-state control, two seeds and unchanged independent gates. R40 frozen source remains intact; candidate training update is separately certified.'")
 (OUT/name).write_text(text)
print(json.dumps(dict(frozen_run=str(RUN),files=len(new),changed_files=sorted(changed),CPU_checks=len(test['checks']),original_unchanged=hashes(SRC)==old)),flush=True)
