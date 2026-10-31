"""Behavioral selection is separate from successful experiment execution."""
from pathlib import Path
import json
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
d=json.loads((ROOT/'matched_reward_review.json').read_text())
assert d['completed'] and len(d['rows'])==8
rows=[]
for seed in [179,191]:
    control=next(r for r in d['rows'] if r['seed']==seed and r['branch']=='control' and r['updates']==128)
    candidate=next(r for r in d['rows'] if r['seed']==seed and r['branch']=='candidate' and r['updates']==128)
    for label,field in [('nominal','nominal_cases'),('pushed','pushed_cases'),('diverse_nominal',None),('diverse_pushed',None)]:
        if field: cc,tc=control[field],candidate[field]
        else:
            suffix='diverse_push' if label=='diverse_pushed' else label
            cc,tc=control[suffix]['cases'],candidate[suffix]['cases']
        current_control=next(c for c in cc if c['command']==3.5)
        current_candidate=next(c for c in tc if c['command']==3.5)
        flight_control=next(c for c in cc if c['command']==1.5)
        flight_candidate=next(c for c in tc if c['command']==1.5)
        rows.append(dict(seed=seed,test=label,control_current_pass=current_control['passed'],candidate_current_pass=current_candidate['passed'],
            control_3_5_lateral_m_s=current_control['lateral_speed'],candidate_3_5_lateral_m_s=current_candidate['lateral_speed'],
            candidate_current_failures=current_candidate['failures'],
            control_1_5_genuine_flight_fraction=flight_control['flight'],candidate_1_5_genuine_flight_fraction=flight_candidate['flight'],
            control_prior_failures=[dict(command=c['command'],failures=c['failures']) for c in cc[:-1] if not c['passed']],
            candidate_prior_failures=[dict(command=c['command'],failures=c['failures']) for c in tc[:-1] if not c['passed']]))
assert all(not r['candidate_current_pass'] for r in rows), 'Reassess direction selection if a replicated case improves'
assert all(r['candidate_1_5_genuine_flight_fraction']<.015 for r in rows)
selection=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),candidate_status='REJECTED',deploy=False,
    exact_two_seed_128_direction_and_prior_comparisons=rows,
    reason='At128 the narrow slow tracker loses current3.5 direction across both learning seeds and original/independent-state tests. Some slow tracking improves, but1.5 genuine flight remains unqualified and no whole-stage repair is achieved. Reject immediate deployment; this does not establish that all command-aware rewards or later learning are ineffective.',
    original_full_learner_preserved=True,diagnostic_actors_deployed=False)
(ROOT/'reward_selection.json').write_text(json.dumps(selection,indent=2)+'\n',encoding='utf-8')
print(json.dumps(selection,indent=2))
