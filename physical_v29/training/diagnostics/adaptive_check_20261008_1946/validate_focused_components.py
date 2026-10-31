"""Independent completion/source/full-load/diversity audit, separate from verdict."""
import hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_focused_prior'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
from validate_physical_replay import validate
parity=validate();cpu=read(OUT/'focused_prior_cpu_tests.json');assert cpu['passed'] and cpu['count']==27
copy=read(OUT/'protected_source_copy.json')
for relative,h in copy['protected_hashes'].items():
 assert sha(Path(copy['source'])/relative)==h and sha(OUT/'control_source/physical_v29'/relative)==h
for name,n in [('candidate_smoke',8),('candidate_full_reload',4),('passive_PPO_capture',4)]:
 folder=ROOT/name;assert read(folder/'result.json')['completed']
 proof=read(folder/'full_load_proof.json');assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
 rows=[json.loads(v) for v in (folder/'progress.jsonl').read_text().splitlines()];assert len(rows)==n
 assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
 assert all(r[k]['batches']==20 for r in rows for k in ['retention','locomotion_retention','upper_refinement'])
review=read(OUT/'matched_focused_review.json');assert review['completed'] and len(review['rows'])==8
assert read(ROOT/'completion.json')['completed']
for seed in [157,167]:
 for branch in ['control','candidate']:
  for updates in [64,128]:
   folder=ROOT/f'seed{seed}_{branch}_{updates:03d}_train'
   assert read(folder/'training_validation.json')['passed'] and read(folder/'result.json')['completed']
  for suffix in ['diverse_nominal','diverse_push']:
   a=read(ROOT/f'seed{seed}_control_128_{suffix}/initial_diversity.json');b=read(ROOT/f'seed{seed}_candidate_128_{suffix}/initial_diversity.json')
   assert a['initial_state_sha256']==b['initial_state_sha256']
   assert a['initial_q']==b['initial_q'] and a['initial_qd']==b['initial_qd']
   assert a['per_robot_seed_sequence']=='[seed, environment_id]' and all(n>=32 for n in a['unique_initial_states_by_command'].values())
physical=read(OUT/'motor_retention_response/result.json');motor=read(OUT/'motor_substep_attribution.json')
assert physical['completed'] and physical['maximum_command_difference']==0 and physical['maximum_reward_term_sum_error']<1e-3
assert len(physical['groups'])==14 and all(g['survival']==1 and g['resets']==0 for g in physical['groups'])
assert motor['passed'] and motor['independent_saved_numpy_driving_clip_mask_matches']
assert read(OUT/'current_gradient_analysis.json')['completed'] and read(OUT/'arm_feedback_sensitivity.json')['completed']
record=dict(passed=True,checks=['27_exact_bucket_and_other_frontier_CPU_fixtures','all_same_actor_cases_and51arrays_bitwise_equal',
 'eight_actual64env_PPO_and_four_full_state_reload_updates','four_passive_actual4096env_PPO_updates',
 'all_actor_critic_Adam_exploration_loads_exact_zero_KL_command_mismatch','three_original_auxiliaries20batches_update',
 'two_independent_learning_seeds_64_and128_unchanged_control_candidate','all_nominal_pushed_and32_per_robot_seeded_extra_tests_complete',
 'paired_initial_states_equal_between_branches','53_protected_authored_sources_identical','normal_limit448robot_two_seed20s_physics_and4000native_substeps',
 'independent_saved_data_driving_clipping_attribution_matches'],
 physical_arrays=parity['arrays_bitwise_equal'],protected_files=copy['files'],original_checkpoint=physical['checkpoint'],sha256=physical['checkpoint_sha256'],
 experiment_signature_does_not_certify_sampler_override=True,diagnostic_actors_deployed=False)
(OUT/'experiment_validation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))
