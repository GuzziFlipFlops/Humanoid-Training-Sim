"""Independently validate saved comparison after the harness's time-budget guard."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;root=OUT/'matched_heading'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
review=read(OUT/'matched_heading_review.json');boundary=read(OUT/'stopped_boundary.json')
assert review['completed'] and len(review['comparisons'])==4
for r in review['comparisons']:
 assert r['KL_command_mismatch_zero'] and r['all_upper_and_priors20_batches']
 parent=Path(r['checkpoint']).parent;load=read(parent/'full_load_proof.json');args=read(parent/'run_config.json')['arguments']
 expected=Path(boundary['checkpoint']) if r['updates']==64 else root/f"{r['branch']}_064_train/candidate.pt"
 assert Path(args['checkpoint']).resolve()==expected.resolve()
 assert hashlib.sha256(expected.read_bytes()).hexdigest()==load['source_sha256']
 for k in ['nominal','pushed']+(['push_seed1901','diverse_nominal'] if r['updates']==128 else []):
  folder=Path(r[k]['folder']);res=read(folder/'result.json');ev=read(folder/'evaluation.json')
  assert res['completed'] and ev['duration_s']==20.
  assert Path(read(folder/'run_config.json')['arguments']['checkpoint']).resolve()==Path(r['checkpoint']).resolve()
  assert len(ev['cases'])==10 and all(c['statistics']['trials']==32 for c in ev['cases'])
  assert all(c['statistics']['survival']==1. for c in ev['cases'])
 if r['updates']==128:
  receipt=r['initial_diversity'];assert all(n==32 for n in receipt['unique_initial_states_by_command'].values())
control=next(r for r in review['comparisons'] if r['branch']=='control' and r['updates']==128)
candidate=next(r for r in review['comparisons'] if r['branch']=='candidate' and r['updates']==128)
assert control['initial_diversity']['initial_state_sha256']==candidate['initial_diversity']['initial_state_sha256']
assert all(not candidate[k]['cases'][-1]['passed'] for k in ['nominal','pushed','push_seed1901','diverse_nominal'])
stderr=(OUT/'matched_heading_harness_stderr.log').read_text(encoding='utf-16' if (OUT/'matched_heading_harness_stderr.log').read_bytes().startswith(b'\xff\xfe') else 'utf-8-sig')
assert 'Matched diagnostic overall deadline exceeded' in stderr
validation=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),all_requested_learning_and_gate_outputs_complete=True,
 four_exact_full_state_training_blocks=True,smoke_and_reload_complete=all(read(root/n/'result.json')['completed'] for n in ['candidate_smoke','candidate_reload']),
 same_initial_states_between_branches=True,independent_unique_initial_states_per_command=32,
 harness_exit_success=False,harness_post_completion_budget_seconds=2500,actual_comparison_seconds=2581.892270088196,
 reason='Overall harness guard fired after every requested worker saved its complete result; outputs independently revalidated. No production failure or candidate deployment.',diagnostic_updates_not_counted_as_main=256)
(OUT/'comparison_validation.json').write_text(json.dumps(validation,indent=2))
focus=[
 'Preserve the same current full learner and compare newer exact3.5 direction plus0.15 net travel/1.5 flight/all prior gates. Course-feedback1 is REJECTED for immediate deployment:64 passes3.5 but128 loses it(0.185/0.179m/s lateral) and weakens1.5 flight; extra push and32 seeded initial states reproduce failure. Physical course response remains promising,not solved. Before another version measure learned128 versus current/original254 truthful yaw-response,command saturation and complete per-command objective/advantage/effective-target gradients,especially1.5 flight and0.15 travel. Test a bounded retention-preserving continuation against64/128 full-state control plus independent seeds; do not blindly increase gain,repeat identical feedback or change arbitrary weights.',
 'Mandatory head/shoulder pitch+roll/elbow angle-target-velocity timelines and arm-leg coordination by speed. Reuse upper_bias_audit.py alongside upper_joint_time_audit.py:0.5Hz descriptive separation,3s edges,post-warmup2.02s time origin,reset segmentation. Control128/2.5 totalpitch ranges~40/42deg but faster residual~10/13deg; much is slow bias. Quantify commanded bias versus actual response,auxiliary/PPO gradient and coupling before a narrow bias/phase refinement; preserve head/gait/full state,35/1 gains and all gates. No human/Unitree encoder or optical-gaze certification; source-matched encoder curves remain unavailable.',
 'Verify same39000 supervisor,one actual worker55 hashes,local ground,exact full actor/critic/Adam/exploration and both teachers. RecoveredR40 count is50 at this boundary,not old41;27507 inherited,32 unsaved R39 excluded. Exclude successful1918.3s and3502.1s pauses; nominal255 wrapper includes this pause.8192target/reserve300/hostfallback/180continuoushealthyTRAINING-second ramps remain; latest capacity is constrained by other GPU use. This384-robot5ms test had zero driving-speed clips,not a ceiling. Final5 or conditional4 still needs every unchanged gate and32 independently seeded initial states per command plus causal native limits for4.'
]
decision=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),decision='Reject immediate course-feedback deployment; resume same original full-state learner',production_change=False,diagnostic_actors_deployed=False,
 summary='High-speed course-feedback candidate is REJECTED for immediate deployment. It passes both3.5 cases at64 updates(0.132/0.130m/s lateral),but loses both at128(0.185/0.179),weakens retained1.5 flight and retains0.15 travel failures. Extra push and independent small-reset cohorts reproduce the tradeoff. Preserve its promising physical direction evidence without declaring the law permanently ineffective; resume the exact original learner.',
 exact_source_checkpoint=boundary['checkpoint'],source_sha256=boundary['checkpoint_sha256'],comparison_validation=str(OUT/'comparison_validation.json'),next_focused_tests=focus)
(OUT/'decision.json').write_text(json.dumps(decision,indent=2));print(json.dumps(dict(decision=decision['decision'],all_outputs_validated=True)))
