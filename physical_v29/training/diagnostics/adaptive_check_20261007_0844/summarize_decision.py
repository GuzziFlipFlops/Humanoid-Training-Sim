import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
review=read(OUT/'matched_context_review.json')
summaries=[]
for r in review['comparisons']:
    summary={'branch':r['branch'],'updates':r['updates'],'tests':{}}
    for key in ['nominal','pushed','push_seed1901','diverse_nominal']:
        if key not in r:continue
        cases=r[key]['cases'];cur=cases[-1]
        summary['tests'][key]={'signed_speed':cur['statistics']['reset_frame_forward_speed_m_s'],
            'lateral_speed':cur['statistics']['reset_frame_lateral_speed_m_s'],
            'true_flight':cur['statistics']['geometric_flight_fraction'],
            'head_RMS_deg':cur['head_RMS_deg'],
            'all_survive':all(c['statistics']['survival']==1 for c in cases),
            'failures':[{'command':c['command'],'failures':c['failures']} for c in cases if not c['passed']]}
    summaries.append(summary)
decision={'recorded_utc':datetime.now(timezone.utc).isoformat(),'action':'REJECT teacher-context removal for deployment; resume same original R40 full-state learner.',
    'reason':'Candidate recovers the slowest-speed gait but only slightly improves current drift at128 updates and introduces1.5 lateral and2.5 slip failures that unchanged control avoids. Current3.5 direction remains unqualified. Extra independently seeded initial states retain a comparison, not qualification.',
    'matched_test':str(OUT/'matched_context_review.json'),'comparisons':summaries,
    'diagnostic_actors_deployed':False,'production_source_reward_gain_caps_or_optimizer_reset':False,
    'full_actor_critic_Adam_exploration_preserved':True,'should_start_from_scratch':False,
    'current_stage_qualified':False,'highest_existing_contract_qualified_speed_m_s':3.,
    'actual5m_s_proven':False,'actual4m_s_fallback_proven':False,'velocity_ceiling_proven':False,
    'human_like_upper_motion_certified':False,
    'seed_scope':'Extra seed2301 provides32 independently generated small nominal-physics q/qd initial states per command on each diagnostic128 actor. It does not qualify production, failed current cases,5m/s,or4m/s. Seed1901 changes pushes only; ordinary nominal seeds still clone initial states.',
    'unchanged_gate_classifiers':True,'original_signature_does_not_certify_diagnostic_training_override':True,
    'truthful_teacher349_freshly_passes_both_tests':all(x['passed'] for x in review['truthful_teacher'].values()),
    'next_focused_tests':[
        'Compare fresh exact current3.5 direction and all prior gates. The old-head-context removal is rejected after64/128 full-state controls and diverse resets; preserve current learner,267 and349. Use the yaw-aligned objective audit to test truthful normal-limit heading-feedback response and complete objective versus fixed-frame drift. Verify actual yaw-channel sensitivity and target/heading timing before any narrowly versioned64/128 learning candidate. Do not blindly repeat rejected fixed-frame kernels,teacher changes,arbitrary weights or relax gates.',
        'Inspect mandatory upper angles,targets,velocities and arm-leg phase by speed. Separate slow shoulder bias drift from cycle swing. Existing upper20-batch refinement stays; both matched actors retain asymmetry. Before another upper change measure effective targets,auxiliary/PPO gradients and physical coupling. Original gains35/1 and native caps remain; no fabricated Unitree encoder curves,world-gaze certification,spoof,hybrid or tilted-head rollback.',
        'Verify same full-state supervisor,one worker55 hashes,local ground,27507 inherited plus current completed and41 hash-proven recovery updates. Keep8192 target,reserve300,host fallback and180 healthy training seconds. Exclude documented pause,exclude all diagnostic samples. Extend reproducible independent initial-state tests to eventual endpoint actors; final5 or conditional4 still requires every gate and causal limits for4.']}
(OUT/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
qa=read(OUT/'plot_qa.json')
for branch in ['control','candidate']:
    for name in ['upper_joint_angles.png','upper_joint_velocities.png']:
        p=OUT/(branch+'_upper')/name
        if not any(x['path']==str(p) for x in qa['files']):
            qa['files'].append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'visually_inspected':True})
qa['matched_findings']='All four matched128 plots visually inspected. Unequal shoulder biases/amplitudes,elbow coordination and fast head oscillation persist. At2.5/3 posture bias drifts over seconds; target excursions exceed actual fast shoulder response. Candidate does not establish natural upper motion.'
(OUT/'plot_qa.json').write_text(json.dumps(qa,indent=2)+'\n')
print(json.dumps({'action':decision['action'],'truthful_teacher_pass':decision['truthful_teacher349_freshly_passes_both_tests'],'comparisons':summaries},indent=2))
