"""Stage a normal-limit current/qualified-prior probe from tested diagnostic code."""
from pathlib import Path

OUT = Path(__file__).resolve().parent
source = OUT.parent / 'adaptive_check_20261007_2346/timing_probe.py'
text = source.read_text(encoding='utf-8')
text = text.replace("out=OUT/'timing_response'", "out=OUT/'motor_retention_response'")
text = text.replace("actor_files={'current':checkpoint}", "config_args=json.loads((checkpoint.parent/'run_config.json').read_text(encoding='utf-8'))['arguments']\n    certificate=json.loads(Path(config_args['locomotion_prior_certificate']).read_text(encoding='utf-8'))\n    actor_files={'current':checkpoint,'qualified_prior':Path(certificate['checkpoint'])}")
old = """    for seed in [4191,4211]:
        for speed in [.15,1.5]:groups.append((f'seed{seed}_{speed}_baseline',speed,0.,.5,None,'current',seed,'baseline'))
        for law in ['baseline','instant','lowpass120ms','cycle380ms']:
            groups.append((f'seed{seed}_3.5_{law}',3.5,0. if law=='baseline' else 1.,.5,None,'current',seed,law))"""
new = """    for seed in [4291,4311]:
        for actor in ['current','qualified_prior']:
            for speed in ([.15,1.5,3.,3.5] if actor=='current' else [.15,1.5,3.]):
                groups.append((f'seed{seed}_{actor}_{speed}',speed,0.,.5,None,actor,seed,'baseline'))"""
assert old in text
text = text.replace(old, new)
old = "        assert saved['infos']['reward_signature']==training_reward_signature(TRAIN)"
new = """        if label=='current':
            assert saved['infos']['reward_signature']==training_reward_signature(TRAIN)
        else:
            from standing_retention import validate_qualification
            from evaluation import GATE_VERSION
            validate_qualification(certificate,str(file),training_reward_signature(TRAIN),GATE_VERSION)
            assert actor_hashes[label]==certificate['checkpoint_sha256']
            (out/'qualified_teacher_receipt.json').write_text(json.dumps(dict(certificate=certificate,
                actor_sha256=actor_hashes[label],original_saved_signature=saved['infos']['reward_signature'],
                current_gate_signature=training_reward_signature(TRAIN),scope='Exact freshly gate-certified teacher; diagnostic inference only. This does not admit an experimental actor or authorize generic signature bypass.'),indent=2),encoding='utf-8')"""
assert old in text
text = text.replace(old, new)
text = text.replace("substep_ratios=[];substep_speed_clip=[];original_compute=raw.actuator.compute", "substep_ratios=[];substep_speed_clip=[];motor_requests=[];motor_applied=[];motor_velocities=[];original_compute=raw.actuator.compute")
text = text.replace("ratios=[];clipped=[]", "ratios=[];clipped=[];requested_all=[];applied_all=[];velocities_all=[]")
text = text.replace("ratios.append(velocity.abs()/cap)", "ratios.append(velocity.abs()/cap)\n            requested_all.append(requested);applied_all.append(applied);velocities_all.append(velocity)")
text = text.replace("applied=raw.actuator.applied_effort[:,ordinary]", "applied=raw.actuator.applied_effort[:,ordinary]\n        requested_all.append(request);applied_all.append(applied);velocities_all.append(velocity)")
text = text.replace("        return result\n    raw.actuator.compute", "        motor_requests.append(torch.cat(requested_all,1).detach().cpu().numpy().copy())\n        motor_applied.append(torch.cat(applied_all,1).detach().cpu().numpy().copy())\n        motor_velocities.append(torch.cat(velocities_all,1).detach().cpu().numpy().copy())\n        return result\n    raw.actuator.compute")
start = text.index("    substate={k:[]")
end = text.index("    reward_terms={}", start)
text = text[:start] + """    substate={k:[] for k in ['joint_position','joint_velocity','target','phase']}
    original_scene_update=raw.scene.update
    physics_tick=0
    representative_ids=torch.arange(0,n,per,device=raw.device)
    def post_physics_update(dt):
        nonlocal physics_tick
        original_scene_update(dt)
        assert abs(dt-.005)<1e-9
        physics_tick+=1
        values=dict(joint_position=raw.robot.data.joint_pos[representative_ids],
            joint_velocity=raw.robot.data.joint_vel[representative_ids],target=raw.targets[representative_ids],phase=raw.phase[representative_ids])
        for k,v in values.items():substate[k].append(v.detach().cpu().numpy().copy())
    raw.scene.update=post_physics_update
""" + text[end:]
text = text.replace("np.savez_compressed(out/'root_kinematics_200hz.npz'", "np.savez_compressed(out/'representative_joint_200hz.npz'")
text = text.replace("motor_names=np.asarray(motor_names),motor_units=np.asarray(motor_units),dt=.005,robots_per_group=per)", "motor_names=np.asarray(motor_names),motor_units=np.asarray(motor_units),dt=.005,robots_per_group=per,requested_torque=np.asarray(motor_requests),applied_torque=np.asarray(motor_applied),native_velocity=np.asarray(motor_velocities))")
text = text.replace("scope='Current3.5 actor,truthful heading/course/yaw command interventions. Driving clips are sampled exactly at5ms; one test cannot prove a high-speed ceiling.'", "scope='Current actor and qualified prior, unchanged fixed commands at two initial-state seeds. Actual requested/applied torque, native speed ratio and driving clips at5ms. No command/action/observation intervention; one test cannot prove a high-speed ceiling.'")
text = text.replace("initial_state_seeds=[4191,4211]", "initial_state_seeds=[4291,4311]")
text = text.replace("original_signature_does_not_certify_external_command_law=True", "original_signature_does_not_certify_external_command_law=False")
old_scope = "384 physical robots,20s,32 paired initial states per cohort at each of two independent seeds. Exact current actor, baseline,instantaneous course120ms velocity low-pass and380ms trailing-cycle mean course. Actual5ms post-physics root link quaternion/world COM velocity and direct PhysX parity captured. No action offsets,support,phase/input rewrite or learned actor. All external command laws diagnostic only; not qualification or certified inference."
new_scope = "448 physical robots,20s,32 paired independent initial states per cohort at each of two seeds. Exact current actor versus gate-certified prior; unchanged native fixed commands/heading feedback. Actual5ms requested/applied torque, motor speeds/driving clips and representative physical q/qd/target/phase. No action offsets,support,phase/input rewrite or learned diagnostic actor. This is controlled diagnosis, not complete independent locomotion qualification."
assert old_scope in text
text = text.replace(old_scope, new_scope)
(OUT / 'motor_retention_probe.py').write_text(text, encoding='utf-8')
print('Prepared 448-robot truthful current/qualified-prior motor and retention probe.')
