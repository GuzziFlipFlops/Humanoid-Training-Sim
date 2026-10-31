"""Stage the previously untested inverse arm comparison for diagnosis only."""
from pathlib import Path
import json, hashlib
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'high_speed_extrapolation'
helper='''"""Diagnostic action-row composition; truthful closed-loop feedback only."""
def replace_arm_rows(current, teacher, rows, mask):
    assert current.shape==teacher.shape and current.ndim==2
    assert rows.shape==(current.shape[1],) and mask.shape==(current.shape[0],)
    assert rows.dtype==mask.dtype==__import__('torch').bool
    result=current.clone()
    result[mask[:,None].expand_as(result) & rows[None,:].expand_as(result)] = teacher[mask[:,None].expand_as(teacher) & rows[None,:].expand_as(teacher)]
    return result
'''
(OUT/'diagnostic_arm_rows.py').write_text(helper,encoding='utf-8')
text=(ROOT/'motor_retention_probe.py').read_text(encoding='utf-8')
text=text.replace("HERE=OUT.parents[1]","HERE=OUT.parents[2]")
text=text.replace("out=OUT/'motor_retention_response'","out=OUT/'inverse_arm_response'")
old="""for seed in [4491,4511]:
        for actor in ['current','qualified_prior']:
            for speed in ([.15,.35,.65,1.,1.5,3.,3.5] if actor=='current' else [.15,.35,.65,1.,1.5,3.]):"""
new="""for seed in [4591,4611]:
        for actor in ['current','qualified_prior','current_body_teacher_arms']:
            for speed in [.15,1.5]:"""
assert old in text;text=text.replace(old,new)
old="masks={label:torch.tensor(actor_labels==label,device=raw.device) for label in policies}"
new="""masks={label:torch.tensor(actor_labels==label,device=raw.device) for label in ['current','qualified_prior','current_body_teacher_arms']}
    arm_names=['left_shoulder_pitch','right_shoulder_pitch','left_shoulder_roll','right_shoulder_roll','left_elbow_pitch','right_elbow_pitch']
    arm_indices=[raw.robot.joint_names.index(name) for name in arm_names]
    arm_rows=torch.zeros(23,dtype=torch.bool,device=raw.device);arm_rows[arm_indices]=True
    assert not arm_rows[raw.lookup['head_yaw']] and not arm_rows[raw.lookup['head_pitch']]
    assert sum(m.to(torch.int32) for m in masks.values()).eq(1).all()
    from diagnostic_arm_rows import replace_arm_rows
    (out/'controller_manifest.json').write_text(json.dumps(dict(diagnostic_only=True,
        arms=arm_names,current_head_and_body_preserved=True,truthful_observations=True,
        teacher_used_only_commands=[.15,1.5],teacher_certificate_ceiling=3.,
        original_signature_does_not_certify_composite=True,production_composite_deployed=False,
        helper_sha256=hashlib.sha256((OUT/'diagnostic_arm_rows.py').read_bytes()).hexdigest()),indent=2))"""
assert old in text;text=text.replace(old,new)
old="""action=torch.empty((n,23),device=raw.device)
            for label,policy in policies.items():action[masks[label]]=policy(observation)[masks[label]]"""
new="""assert observed_commands[:,:2].norm(dim=1).max()<=3., 'Teacher inference above its certificate'
            current_action=policies['current'](observation)
            prior_action=policies['qualified_prior'](observation)
            action=replace_arm_rows(current_action,prior_action,arm_rows,masks['current_body_teacher_arms'])
            action[masks['qualified_prior']]=prior_action[masks['qualified_prior']]
            assert torch.equal(action[masks['current']],current_action[masks['current']])
            assert torch.equal(action[masks['current_body_teacher_arms']][:,~arm_rows],current_action[masks['current_body_teacher_arms']][:,~arm_rows])
            assert torch.equal(action[masks['current_body_teacher_arms']][:,arm_rows],prior_action[masks['current_body_teacher_arms']][:,arm_rows])"""
assert old in text;text=text.replace(old,new)
text=text.replace('initial_state_seeds=[4491,4511]','initial_state_seeds=[4591,4611]')
text=text.replace('832 physical robots','384 physical robots')
text=text.replace('Exact current actor versus gate-certified prior;','Exact current actor, gate-certified prior and current-body/current-head with prior arms diagnostic;')
text=text.replace("scope='832", "scope='384")
# Preserve failed/empty cohorts as null fields, never numeric success sentinels.
token="(out/'result.json').write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result,indent=2),flush=True)"
assert token in text
text=text.replace(token,"""def finite_json(value):
        if isinstance(value,float) and not np.isfinite(value):return None
        if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
        if isinstance(value,list):return [finite_json(v) for v in value]
        return value
    result=finite_json(result)
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\\n');print(json.dumps(result,indent=2,allow_nan=False),flush=True)""")
compile(text,str(OUT/'inverse_arm_probe.py'),'exec')
(OUT/'inverse_arm_probe.py').write_text(text,encoding='utf-8')
print('Prepared 384-robot two-seed current-body/current-head versus qualified arms diagnostic; no deployment.')
