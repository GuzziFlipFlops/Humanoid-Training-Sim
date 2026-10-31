"""Separate valid execution evidence from a failed behavioral candidate."""
import json
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
comparison=read(OUT/'matched_focused_review.json')
validation=read(OUT/'experiment_validation.json')
upper=read(OUT/'upper_matched_comparison.json')
assert comparison['completed'] and validation['passed'] and upper['completed']
assert all(r['visual_QA_saved'] for r in upper['records'])
selected=[]
for seed in [157,167]:
 rows={r['branch']:r for r in comparison['rows'] if r['seed']==seed and r['updates']==128}
 assert len(rows)==2
 for test in ['nominal','pushed']:
  get=lambda branch,cmd:next(c for c in rows[branch][test+'_cases'] if c['command']==cmd)
  assert get('candidate',1.5)['flight']<get('control',1.5)['flight']
  assert not get('candidate',3.5)['passed'] and not get('candidate',1.5)['passed']
  selected.append(dict(seed=seed,test=test,control={str(c):get('control',c) for c in [.15,1.,1.5,3.,3.5]},
   candidate={str(c):get('candidate',c) for c in [.15,1.,1.5,3.,3.5]}))
explanation=(
 'The focused failed-prior rehearsal candidate is rejected for deployment after two independent learning seeds and exact full-state64/128 controls. '
 'Only the existing24% prior bucket is reweighted; each0.15/1.5 anchor rises from expected3% to4.8%, while standing32% and frontier44% stay unchanged. '
 'At128, candidate1.5 geometric flight is1.021/0.969% versus control1.222/1.240% for seed157, and0.611/0.559% versus1.069/0.986% for seed167; all remain below1.5%. '
 'Seed157 candidate loses0.15 travel and both1m walking-support passes; seed167 recovers0.15 and3m direction but also fails both1m support tests, where its control pushed test passes. '
 'Extra32 per-robot seeded initial states reproduce these failures. Neither seed qualifies3.5 direction. Preserve the original learner; no diagnostic actor or sampler enters production. '
 'This rejects the tested allocation, not every retention method or a claim that training cannot converge.')
next_tests=[
 'Preserve current full-state R40 and compare newer exact3.5 direction plus0.15 net travel,1.5 genuine flight and every prior gate. The focused prior-bucket reweighting is rejected at64/128 across learning seeds157/167 and paired32 independent initial states; do not repeat it without new evidence. Native-limit physical349 still gives1.83-1.86% genuine1.5 flight and higher complete mean reward than exact773 with zero flight. Resolve per-command/phase teacher effective-target errors, actual post-clamp/joint/slew response and sole/contact timing on truthful settled and transitioning histories before one bounded dynamics-aware, command-local continuation with exact-current64/128 controls, two learning seeds and all unchanged gates. Positive normalized advantages, smaller teacher norms and mixed alignment do not prove a sampling defect or globally opposed gradients. Preserve605 current-case evidence,267 all-prior and separately certified349.',
 'Inspect mandatory actual head/shoulder pitch+roll/elbow angle-target-velocity timelines, bilateral amplitude/arm-leg phase and noncancelling slow bias. First actual773 minibatch same-command/phase-bin donor arm-history substitutions change effective lower outputs0.324 RMS at3.5, but this CPU sensitivity does not identify mechanical coupling or a safe repair. Before another shoulder derivative, run bounded truthful free-base arm-target/base-response cohorts and reset-safe sequential state coverage to separate physical coupling from learned feedback dependence. Existing upper20 batches, forward head, legs, both teachers and35/1 gains remain. The tested mean, stiffness, exposure, course and teacher candidates stay rejected absent new evidence. No production observation substitution, imposed reference curve, hybrid or cap increase; mapped human/Unitree encoder curves and optical gaze remain unverified.',
 'Verify same39000 supervisor, one actual worker,55 frozen hashes, local ground and exact actor/critic/Adam/exploration plus both teachers. Count27507 inherited plus fresh completed and50 verified R40 recoveries; exclude32 unsaved R39 and all diagnostic samples. Exclude the documented5405.4s pause only through each rate endpoint. Keep8192 target,300MiB reserve,host fallback and180 continuous healthy TRAINING seconds; Fusion stays closed. Exact773 native5ms request/application audit finds zero driving-speed clips through3.5; overspeed ankles are braking, not a proven ceiling. Later79410Hz samples have ankle overspeed but cannot inherit773 substep attribution. If a newer high-speed plateau persists, record actor-specific5ms requested/applied torque, CAD speed, available torque and sustained driving saturation. Only fully qualified5 or qualified4 with causal native limits blocking5 ends training/automation, with all gates and32 independently seeded states/cmd.'
]
record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),verdict='REJECT_FOCUSED_PRIOR_REHEARSAL',deploy=False,
 explanation=explanation,execution_validation_passed=True,behavioral_candidate_qualified=False,
 full_stage_qualified=False,velocity_ceiling_proven=False,production_change=False,diagnostic_actors_deployed=False,
 evidence_scope='Matched controls and candidate start exact773 actor/critic/Adam/exploration. Other protected sources, numerical rewards, caps, teachers and upper auxiliary are identical; original signature does not certify the experimental sampler.',
 two_learning_seed_128_comparisons=selected,next_tests=next_tests)
(OUT/'decision.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(record['verdict'])
