"""Decision binds complete independent control/candidate comparisons."""
import json
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
d=read(OUT/'matched_direction_review.json');assert d['complete']
control=next(r for r in d['rows'] if r['branch']=='control' and r['updates']==128)
candidate=next(r for r in d['rows'] if r['branch']=='candidate' and r['updates']==128)
for k in ['nominal','pushed']:
 assert 'world_lateral_speed' in next(c for c in candidate['cases'][k] if c['command']==3.5)['failures']
 assert 'no_hopping' in next(c for c in candidate['cases'][k] if c['command']==2.5)['failures']
 assert next(c for c in control['cases'][k] if c['command']==2.5)['passed']
focus=[
 'Direction correction is a promising but undeployed hypothesis: exact4096-env128 comparison lowers3.5 lateral0.590/0.571 to0.254/0.242m/s, yet world-lateral still fails and retained2.5 hop fraction2.299% exceeds the unchanged2% cap in both tests;1.5 flight also fails. First analyze per-command actual advantages/PPO versus prior gradients and2.5 landing events on these saved actors. Require a bounded retention-preserving candidate, independently seeded replication and full-state64/128 control/all gates before deployment; original signatures do not certify diagnostic reward overrides. Do not blindly repeat the same kernel, permanently infer the frame hypothesis false, arbitrarily change weights, or deploy diagnostic actors.',
 'Every check inspect actual upper angle/target/velocity timelines. Actual shoulder35/1 free-base tests show high-frequency attenuation; more damping worsens pitch and is not selected. Before an upper repair, test bounded normal-limit stiffness/damping or feasible target-velocity/phase response with truthful feedback and motor-effort/base coupling, then narrow full-state matched control/retention tests. Preserve current head/gait/full learner. Cross-check the CPU modeled world head axis against actual Isaac head-link orientation and original CAD face axis before any world-forward posture label; native head zero alone is insufficient. No cap raise, spoof, hybrid or fabricated human/Unitree curves.',
 'Verify same owned worker55 frozen hashes/local ground/exact full-state recovery, completed plus33 proven recovered R39 updates and16936 inherited, moving6144/8192 target/reserve300/host fallback/180continuoushealthyTRAINING-second ramp. Exclude this1599.9s diagnostic pause and earlier1823.6s pause from clocks. All10 current normal-limit physical cohorts have zero native5ms driving-speed clips, not proof of a high-speed ceiling. Before final5 or conditional4 endpoint establish32 independently seeded nominal-physics initial-state trials per command with all unchanged classifiers; nominal clones alone are insufficient. Preserve useful actors, unsaved Fusion and fulfilled videos.'
]
decision=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),action='Reject this fixed-frame kernel for immediate deployment; resume the same original full learner and existing upper refinement.',
 reason='At128 updates lateral improves about57%, but both3.5 direction tests still fail and a new2.5 no_hopping failure appears in both, while control passes2.5. Preserving gait outweighs isolated direction improvement. The frame hypothesis remains a research lead requiring focused retention/gradient evidence.',
 diagnostic_actors_deployed=False,production_source_reward_PD_caps_optimizer_or_actor_reset=False,current_stage_qualified=False,
 highest_existing_contract_qualified_speed_m_s=3.,velocity_ceiling_proven=False,human_like_upper_motion_certified=False,actual5m_s_proven=False,actual4m_s_fallback_proven=False,
 matched_test=str(OUT/'matched_direction_review.json'),next_focused_tests=focus)
if (OUT/'closing_gate_review.json').exists():
 pair=read(OUT/'closing_gate_review.json')['latest']
 if Path(pair['checkpoint']).parent.name=='267_run_350_train_086':
  assert all(c['passed'] for k in ['nominal_cases','pushed_cases'] for c in pair[k] if c['command']<3.5)
  focus[0]='Latest original267 now passes every stopped/prior case through3m/s in matching268/269; preserve that useful actor and full neutral-head learner.3.5 still fails direction at3.489/3.496 signed and0.516/0.488 lateral. '+focus[0]
  decision['postresume_all_prior_passing_actor']=dict(checkpoint=pair['checkpoint'],sha256=pair['sha256'],nominal=pair['nominal'],pushed=pair['pushed'],all_prior_pass=True,current3_5_qualified=False)
(OUT/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
print(json.dumps(dict(action=decision['action'],reason=decision['reason']),indent=2))
