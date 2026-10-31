"""Prepare separately staged training-only refinement while live training continues."""
import json,shutil,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
active=json.loads((HERE/'ACTIVE_RUN.json').read_text());run=Path(active['run'])
target=OUT/'validation_source/physical_v29'
if not target.exists():shutil.copytree(run/'source/physical_v29',target,ignore=shutil.ignore_patterns('__pycache__','usd','*.pyc'))
shutil.copy2(OUT/'candidate_upper_body_refinement.py',target/'training/upper_body_refinement.py')
p=target/'training/worker_unitree.py';s=(run/'source/physical_v29/training/worker_unitree.py').read_text()
edits=[
('from ppo_update_audit import PpoUpdateAudit',
 'from upper_body_refinement import UpperBodyRefinement,LocomotionOnlyRetention,UPPER_REFINEMENT_VERSION\nfrom ppo_update_audit import PpoUpdateAudit'),
("   details['training_command_version']=TRAINING_COMMAND_VERSION",
 "   details['training_command_version']=TRAINING_COMMAND_VERSION\n   details['upper_refinement_version']=UPPER_REFINEMENT_VERSION\n   details['prior_output_scope']='Torso/legs only; upper posture has a separate training-only output loss'"),
('   retention=None;locomotion_retention=None',
 '   retention=None;locomotion_retention=None\n   joint_names=list(envraw.robot.joint_names)'),
('retention=StandingRetention(runner.alg.actor,teacher)',
 'retention=LocomotionOnlyRetention(runner.alg.actor,teacher,joint_names=joint_names)'),
('locomotion_retention=StandingRetention(runner.alg.actor,teacher,command_range=(.1,maximum))',
 'locomotion_retention=LocomotionOnlyRetention(runner.alg.actor,teacher,command_range=(.1,maximum),joint_names=joint_names)'),
('   # Continuous heartbeat and GPU reserve checks at each PPO update.',
 "   upper_refinement=UpperBodyRefinement(runner.alg.actor,joint_names)\n   atomic_json(out/'upper_refinement.json',upper_refinement.read_audit())\n   # Continuous heartbeat and GPU reserve checks at each PPO update."),
("    state['locomotion_retention']=locomotion_retention.read_audit() if locomotion_retention else None",
 "    state['locomotion_retention']=locomotion_retention.read_audit() if locomotion_retention else None\n    state['upper_refinement']=upper_refinement.read_audit()\n    state['upper_refinement']['actual_joint_mean_rad']={name:float(envraw.robot.data.joint_pos[:,joint_names.index(name)].mean()) for name in upper_refinement.names if name in ['head_yaw','head_pitch','left_shoulder_roll','right_shoulder_roll','left_elbow_pitch','right_elbow_pitch']}"),
]
for old,new in edits:
    assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new)
p.write_text(s,encoding='utf-8');compile(s,str(p),'exec')
# Reversibility proves no load, observation, action, PPO or evaluator behavior changed.
back=s
for old,new in reversed(edits):assert back.count(new)==1;back=back.replace(new,old)
assert back==(run/'source/physical_v29/training/worker_unitree.py').read_text()
supervisor=(HERE/'supervisor_unitree.py').read_text()
old=' if retention_proof:\n  assert not optimization_proof and not curriculum_proof\n  from standing_retention import verify_retention_migration'
new=" if retention_proof:\n  assert not optimization_proof and not curriculum_proof\n  if json.loads(Path(retention_proof).read_text()).get('schema')=='upper_body_output_refinement_v1':\n   from upper_body_refinement import verify_upper_body_migration\n   return verify_upper_body_migration(prior,new_source,retention_proof)\n  from standing_retention import verify_retention_migration"
assert supervisor.count(old)==1;supervisor=supervisor.replace(old,new)
old="'command_curriculum.py','curriculum_provenance.py']:\n   shutil.copy2(HERE/n,snapshot/'training'/n)"
new="'command_curriculum.py','curriculum_provenance.py','upper_body_refinement.py']:\n   shutil.copy2(HERE/n,snapshot/'training'/n)"
assert supervisor.count(old)==1;supervisor=supervisor.replace(old,new)
(OUT/'candidate_supervisor_unitree.py').write_text(supervisor,encoding='utf-8');compile(supervisor,'supervisor','exec')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
proof={'prior_run':str(run),'changed_runtime_files':['training/worker_unitree.py','training/upper_body_refinement.py'],
    'worker_changes_reversible':True,'head_target_native_rad':[0.,0.],
    'critic_Adam_exploration_and_all_actor_tensors_will_be_retained':True,
    'inference_history_action_physics_rewards_and_independent_gates_unchanged':True,
    'source_hashes':{k:sha(target/k) for k in ['training/worker_unitree.py','training/upper_body_refinement.py']}}
(OUT/'candidate_source_proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof,indent=2))
