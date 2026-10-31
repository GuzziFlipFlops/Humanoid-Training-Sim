"""Prepare a bounded diagnostic without modifying production sources."""
import json,shutil
from pathlib import Path
OUT=Path(__file__).resolve().parent;PREV=OUT.parent/'adaptive_check_20261007_1344'
for name in ['prepare_and_pause.py','resume_after_diagnosis.py','run_heading_probe.py']:
    text=(PREV/name).read_text()
    if name=='prepare_and_pause.py':
        text=text.replace('Bounded truthful heading-command/physical-response diagnosis; original supervisor resumes','Bounded saved-actor command-response comparison and passive actual PPO audit; original supervisor resumes')
    if name=='resume_after_diagnosis.py':
        text=text.replace("'diverse_heading_eval.py'}", "'diverse_heading_eval.py','capture_current_gradients.py'}")
    (OUT/name).write_text(text)
for name in ['capture_current_gradients.py','analyze_current_gradients.py']:
    shutil.copy2(OUT.parent/'adaptive_check_20261007_0844'/name,OUT/name)
text=(PREV/'heading_probe.py').read_text()
text=text.replace('import argparse,hashlib,json,sys,time,traceback','import argparse,hashlib,json,sys,time,traceback,copy')
start=text.index('    # g=(name,speed,course_feedback_gain')
end=text.index('    cfg=LocomotionCfg()',start)
text=text[:start]+'''    # Same seeded initial joint states across actors and command laws.
    actor_files={
      'current':checkpoint,
      'original254':Path(OUT.parent/'adaptive_check_20261007_1344/stopped_boundary.json'),
      'control128':OUT.parent/'adaptive_check_20261007_1344/matched_heading/control_128_train/candidate.pt',
      'course128':OUT.parent/'adaptive_check_20261007_1344/matched_heading/candidate_128_train/candidate.pt'}
    actor_files['original254']=Path(json.loads(actor_files['original254'].read_text())['checkpoint'])
    actor_hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in actor_files.items()}
    groups=[]
    for label in actor_files:
        for speed in [.15,1.5,3.5]:groups.append((f'{label}_{speed}_baseline',speed,0.,.5,None,label))
        groups.append((f'{label}_3.5_course1',3.5,1.,.5,None,label))
    per=24;n=len(groups)*per
''' +text[end:]
start=text.index('    config=json.loads(')
end=text.index('    speeds=torch.tensor',start)
text=text[:start]+'''    policies={}
    for label,file in actor_files.items():
        config=json.loads((file.parent/'agent.json').read_text())
        runner=OnPolicyRunner(env,config,log_dir=None,device=raw.device)
        saved=torch.load(file,map_location=raw.device,weights_only=False)
        assert saved['infos']['reward_signature']==training_reward_signature(TRAIN)
        runner.load(str(file),load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False})
        assert all(torch.equal(v,runner.alg.actor.state_dict()[k]) for k,v in saved['actor_state_dict'].items())
        policies[label]=copy.deepcopy(runner.get_inference_policy(raw.device))
    actor_labels=np.array([g[5] for g in groups]).repeat(per)
    masks={label:torch.tensor(actor_labels==label,device=raw.device) for label in policies}
''' +text[end:]
old='    raw.reset();reference_heading=raw.heading_target.clone();observation=raw._get_observations();alive=torch.ones(n,dtype=torch.bool,device=raw.device)'
new='''    raw.reset()
    rng=np.random.default_rng(3071)
    offsets=rng.uniform(-.008,.008,(per,23));velocities=rng.uniform(-.01,.01,(per,23))
    q=raw.stand.expand(n,-1).clone()+torch.tensor(np.tile(offsets,(len(groups),1)),device=raw.device,dtype=raw.stand.dtype)
    qd=torch.tensor(np.tile(velocities,(len(groups),1)),device=raw.device,dtype=q.dtype)
    assert ((q>=raw.target_lo)&(q<=raw.target_hi)).all()
    raw.robot.write_joint_state_to_sim(q,qd)
    raw.targets[:]=q;raw.previous_targets[:]=q;raw.previous_vel[:]=qd
    raw.scene.update(raw.step_dt)
    raw.history.zero_();raw.critic_history.zero_()
    raw._command_tick=-1;raw._last_history_tick=-1
    observation=raw._get_observations();reference_heading=raw.heading_target.clone();alive=torch.ones(n,dtype=torch.bool,device=raw.device)
    initial_q=q.cpu().numpy().copy();initial_qd=qd.cpu().numpy().copy()
    assert len({r.tobytes() for r in np.concatenate([initial_q[:per],initial_qd[:per]],1)})==per
'''
assert old in text;text=text.replace(old,new)
text=text.replace('        with torch.inference_mode():action=policy(observation)', '''        with torch.inference_mode():
            action=torch.empty((n,23),device=raw.device)
            for label,policy in policies.items():action[masks[label]]=policy(observation)[masks[label]]''')
text=text.replace("heading_feedback_gain=g[3],course_feedback_gain=g[2],requested_yaw_rate=g[4],robots=per", "heading_feedback_gain=g[3],course_feedback_gain=g[2],requested_yaw_rate=g[4],actor=g[5],robots=per,\n            yaw_command_saturation_fraction=float((np.abs(data['action_command'][window,ids,2])>=.199999).mean()),\n            mean_absolute_yaw_command=float(abs(data['action_command'][window,ids,2]).mean()),")
# Remove the extra comma after the newly inserted last field.
text=text.replace("mean()),,survival", "mean()),survival")
text=text.replace("joint_names=np.array(raw.robot.joint_names),groups=", "initial_q=initial_q,initial_qd=initial_qd,joint_names=np.array(raw.robot.joint_names),groups=")
text=text.replace("result=dict(completed=True,checkpoint=str(checkpoint)", "result=dict(completed=True,actor_checkpoints={k:str(v) for k,v in actor_files.items()},actor_sha256=actor_hashes,\n        paired_initial_states=True,independent_initial_states_per_cohort=per,initial_state_seed=3071,\n        original_signature_does_not_certify_external_command_law=True,diagnostic_actors_deployed=False,checkpoint=str(checkpoint)")
text=text.replace('384 physical robots,20s,32 per cohort at1/3.5m/s', '384 physical robots,20s,24 independently seeded paired states per cohort, four saved actors at0.15/1.5/3.5m/s')
(OUT/'heading_probe.py').write_text(text)
compile(text,str(OUT/'heading_probe.py'),'exec')
print('Prepared saved-actor response, owned pause/resume, and passive PPO audit.')
