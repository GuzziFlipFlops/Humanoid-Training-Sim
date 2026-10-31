"""Stage a truthful earlier-policy comparison outside production certification."""
import hashlib,json
from pathlib import Path
OUT=Path(__file__).resolve().parent;OLD=OUT.parent/'adaptive_check_20261009_1047';TRAIN=OUT.parents[1]
teacher=TRAIN/'runs/locomotion_20261006_r39_commanded_travel_teacher_certificate/267_run_350_train_086/candidate.pt'
assert teacher.is_file()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
scope='Bounded fresh qualification of earlier neutral-head actor267 and current-state/upper-controller compatibility; truthful physical feedback, unchanged native limits and all gates; diagnostic composites never enter production'
p=OUT/'prepare_and_pause.py';s=p.read_text(encoding='utf-8')
s=s.replace('Bounded slow-command reward precision experiment on the current exact full state versus unchanged control; all gates, weights and native limits retained',scope);p.write_text(s,encoding='utf-8')
p=OUT/'resume_after_diagnosis.py';s=p.read_text(encoding='utf-8').replace("'slow_reward_candidate.py'}","'slow_reward_candidate.py','neutral_teacher_probe.py','neutral_teacher_candidate.py'}");p.write_text(s,encoding='utf-8')
s=(OLD/'motor_retention_probe.py').read_text(encoding='utf-8')
def change(a,b):
    global s
    assert s.count(a)==1,a
    s=s.replace(a,b)
change("out=OUT/'motor_retention_response'","out=OUT/'teacher_compatibility_response'")
change("actor_files={'current':checkpoint,'qualified_prior':Path(certificate['checkpoint'])}",
    "neutral_certificate=json.loads((OUT/'neutral_teacher_certificate.json').read_text())\n    actor_files={'current':checkpoint,'qualified_prior':Path(certificate['checkpoint']),'neutral_prior':Path(neutral_certificate['checkpoint'])}")
change('for seed in [4491,4511]:','for seed in [4631,4651]:')
change("for actor in ['current','qualified_prior']:","for actor in ['current','qualified_prior','neutral_prior','neutral_body_current_upper','neutral_body_current_head']:")
change("for speed in ([.15,.35,.65,1.,1.5,3.,3.5] if actor=='current' else [.15,.35,.65,1.,1.5,3.]):","for speed in [.15,1.5]:")
change('validate_qualification(certificate,str(file),training_reward_signature(TRAIN),GATE_VERSION)',
    "active_certificate=neutral_certificate if label=='neutral_prior' else certificate\n            validate_qualification(active_certificate,str(file),training_reward_signature(TRAIN),GATE_VERSION)")
change("assert actor_hashes[label]==certificate['checkpoint_sha256']","assert actor_hashes[label]==active_certificate['checkpoint_sha256']")
change("(out/'qualified_teacher_receipt.json')", "(out/(label+'_teacher_receipt.json'))")
change('dict(certificate=certificate,','dict(certificate=active_certificate,')
change("    masks={label:torch.tensor(actor_labels==label,device=raw.device) for label in policies}",
    "    masks={label:torch.tensor(actor_labels==label,device=raw.device) for label in set(actor_labels)}\n    upper_names=['head_yaw','head_pitch','left_shoulder_pitch','right_shoulder_pitch','left_shoulder_roll','right_shoulder_roll','left_elbow_pitch','right_elbow_pitch']\n    upper_rows=[raw.lookup[x] for x in upper_names];head_rows=[raw.lookup[x] for x in upper_names[:2]]\n    from diagnostic_teacher_rows import combine\n    composite_tick_assertions=0")
change("            for label,policy in policies.items():action[masks[label]]=policy(observation)[masks[label]]", """            values={label:policy(observation) for label,policy in policies.items()}
            for label,value in values.items():action[masks[label]]=value[masks[label]]
            for label,rows in [('neutral_body_current_upper',upper_rows),('neutral_body_current_head',head_rows)]:
                mixed=combine(values['neutral_prior'],values['current'],rows)
                other=[j for j in range(23) if j not in rows]
                assert torch.equal(mixed[:,rows],values['current'][:,rows])
                assert torch.equal(mixed[:,other],values['neutral_prior'][:,other])
                action[masks[label]]=mixed[masks[label]]
            composite_tick_assertions+=1""")
change('initial_state_seeds=[4491,4511]','initial_state_seeds=[4631,4651]')
change("scope='832 physical robots,20s,32 paired independent initial states per cohort at each of two seeds. Exact current actor versus gate-certified prior; unchanged native fixed commands/heading feedback. Actual5ms requested/applied torque, motor speeds/driving clips and representative physical q/qd/target/phase. No action offsets,support,phase/input rewrite or learned diagnostic actor. This is controlled diagnosis, not complete independent locomotion qualification.'",
    "composite_tick_assertions=composite_tick_assertions,diagnostic_composites=True,scope='640 robots,20s,two seeds and32 paired independent states per cohort at0.15/1.5. Current,qualified349,fresh neutral267,and neutral267 body with current whole-upper or current head only. Every policy sees truthful physical states/history. This is closed-loop component diagnosis,not pure mechanical isolation,full independent qualification or a production controller. All native limits and original complete reward terms remain.'")
change("scope='Current actor and qualified prior, unchanged fixed commands at two initial-state seeds. Actual requested/applied torque, native speed ratio and driving clips at5ms. No command/action/observation intervention; one test cannot prove a high-speed ceiling.'",
    "scope='Passive native5ms attribution on two truthful initial-state seeds. Composite controller cohorts are diagnostic only and do not establish an actuator ceiling. Terminal/reset samples remain available for independently guarded analysis.'")
compile(s,str(OUT/'neutral_teacher_probe.py'),'exec');(OUT/'neutral_teacher_probe.py').write_text(s,encoding='utf-8')
(OUT/'teacher_diagnosis_preparation.json').write_text(json.dumps(dict(teacher_checkpoint=str(teacher),teacher_sha256=sha(teacher),
    source_probe=str(OLD/'motor_retention_probe.py'),source_probe_sha256=sha(OLD/'motor_retention_probe.py'),probe_sha256=sha(OUT/'neutral_teacher_probe.py'),
    commands=[.15,1.5],initial_seeds=[4631,4651],robots_per_cohort=32,population=640,seconds=20,
    no_production_change=True,strict_fresh_all_prior_qualification_required=True,scope=scope),indent=2)+'\n',encoding='utf-8')
print('Prepared exact earlier neutral-head checkpoint and bounded physical compatibility diagnosis.')
