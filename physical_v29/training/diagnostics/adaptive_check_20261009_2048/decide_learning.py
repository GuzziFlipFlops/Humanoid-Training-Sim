"""Explicit diagnostic decision after current-state physical and gradient evidence."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_row_route'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(ROOT/'stage_ready.json')['smoke_and_reload']
assert read(OUT/'current_row_response.json')['passed_numerical_and_scope_checks']
assert read(ROOT/'current_physical_support.json')['passed']
for f,n in [('candidate_smoke',8),('candidate_full_reload',4)]:
 assert read(ROOT/f/'training_validation.json')['actual_updates']==n
 assert read(ROOT/f/'experimental_teacher_target.json')['final_output_rows_only']
checks=read(OUT/'final_row_CPU_checks.json');assert checks['passed']
response=read(OUT/'current_row_response_summary.json')
assert all(r['rows_head_max']==0 and r['rows_trunk_norm']==0 for r in response['records'])
physical=read(ROOT/'current_physical_support.json')['groups']
decision=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),proceed=True,diagnostic_only=True,
 evidence_hashes={n:sha(OUT/n) for n in ['current_row_response.json','final_row_CPU_checks.json','matched_row_route/current_physical_support.json']},
 reason='The current two-seed truthful physical comparison retains useful 267 coupled targets with the current head; real row-only PPO/load/reload passes. Restriction exactly removes the new selected target trunk/head path on fixed states. This is a narrow topology test against shared-route and unchanged exact full-state controls, not deployment or a guarantee of gait preservation.',
 critical_limit='Actual four minibatches show little fast-output insulation: final21-row changes affect frontier outputs almost as much as shared changes. Small target-only head differences do not establish the cause of neck or direction regression. Global clipping, Adam, original PPO and closed-loop feedback still interact. Reject unless every unchanged gate and independent state replication passes.',
 experimental_shared_route_is_topology_control=True,previous_shared_target_rejection_still_valid=True,
 learning_seeds=[251,263],updates=[64,128],required_all_prior_current_head_direction_flight_tests=True,
 selected_physical=[{k:g[k] for k in ['name','command','mean_forward_world_speed','mean_absolute_world_lateral_velocity','unloaded_geometric_flight_fraction','mean_reward_rate','head_RMS_deg']} for g in physical],
 production_source_reward_gain_cap_optimizer_actor_changed=False)
(ROOT/'learning_decision.json').write_text(json.dumps(decision,indent=2)+'\n');print(json.dumps(decision,indent=2))
