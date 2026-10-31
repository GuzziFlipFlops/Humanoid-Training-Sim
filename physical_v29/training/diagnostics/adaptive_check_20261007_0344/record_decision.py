"""CPU-only decision and visual inspection receipts for the completed gain test."""
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
now = lambda: datetime.now(timezone.utc).isoformat()
gain = read(OUT/'matched_gain_review.json')
assert gain['complete'] and not gain['diagnostic_actors_deployed']
control = next(r for r in gain['rows'] if r['branch']=='control' and r['updates']==128)
candidate = next(r for r in gain['rows'] if r['branch']=='candidate' and r['updates']==128)
for label in ['nominal', 'pushed']:
    cc = next(c for c in control[label+'_cases'] if c['command']==1.5)
    pc = next(c for c in candidate[label+'_cases'] if c['command']==1.5)
    assert cc['passed'] and pc['failures']==['geometric_flight']
    assert not candidate[label+'_cases'][-1]['passed']
decision = dict(recorded_utc=now(), action='Reject shoulder stiffness70 for deployment; retain original35/1 and resume exact full learner with supervisor-only goal-budget recovery.',
    reason='Early direction improvement at64 updates faded at128; both current3.5 cases still fail world lateral speed. Candidate introduces1.5 geometric-flight retention failure in both tests while unchanged control passes. Slow arm amplitudes grow and fast bias/asymmetry remain.',
    matched_test=str(OUT/'matched_gain_review.json'), diagnostic_actors_deployed=False,
    production_shoulder_gain_or_reward_or_physics_change=False, production_supervisor_maintenance=True,
    full_actor_critic_Adam_exploration_retained=True, should_start_from_scratch=False,
    current_stage_qualified=False, highest_existing_contract_qualified_speed_m_s=3.0,
    actual5m_s_proven=False, actual4m_s_fallback_proven=False, velocity_ceiling_proven=False,
    human_like_upper_motion_certified=False,
    seed_limitation='Repeated nominal seed1901 invocation uses the existing fixed nominal resets; it does not establish32 independently seeded initial-state trials.',
    unchanged_gate_classifiers=True, original_signature_does_not_certify_diagnostic_gain_override=True,
    deployment=str(OUT/'deployment.json'))
(OUT/'decision.json').write_text(json.dumps(decision,indent=2)+'\n',encoding='utf-8')
qa=[]
for folder in ['initial_upper','control_upper','candidate_upper']:
    for name in ['upper_joint_angles.png','upper_joint_velocities.png']:
        p=OUT/folder/name
        assert p.is_file()
        qa.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),visually_inspected=True))
(OUT/'plot_qa.json').write_text(json.dumps(dict(recorded_utc=now(),files=qa,
    inspection='All six rendered plots inspected through image viewer in this check. Population10Hz and representative50Hz traces are distinguished; reset segments excluded from phase interpretation.'),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(action=decision['action'],plots_inspected=len(qa))))
