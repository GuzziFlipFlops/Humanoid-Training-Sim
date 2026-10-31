"""Build a bounded truthful diagnostic from preserved probe code."""
from pathlib import Path
import shutil
OUT=Path(__file__).resolve().parent;PREV=OUT.parent/'adaptive_check_20261007_1845'
for name in ['prepare_and_pause.py','resume_after_diagnosis.py']:
    shutil.copy2(PREV/name,OUT/name)
p=OUT/'resume_after_diagnosis.py';s=p.read_text().replace("'heading_probe.py','run_heading_probe.py'","'timing_probe.py','run_timing_probe.py'")
p.write_text(s)
s=(PREV/'heading_probe.py').read_text()
s=s.replace("out=OUT/'heading_response'","out=OUT/'timing_response'")
lo=s.index('    actor_files={');hi=s.index('    per=24;n=',lo)
s=s[:lo]+'''    actor_files={'current':checkpoint}
    actor_hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in actor_files.items()}
    groups=[]
    for seed in [4191,4211]:
        for speed in [.15,1.5]:groups.append((f'seed{seed}_{speed}_baseline',speed,0.,.5,None,'current',seed,'baseline'))
        for law in ['baseline','instant','lowpass120ms','cycle380ms']:
            groups.append((f'seed{seed}_3.5_{law}',3.5,0. if law=='baseline' else 1.,.5,None,'current',seed,law))
''' +s[hi:]
s=s.replace('per=24;n=','per=32;n=')
lo=s.index('    original_update=');hi=s.index('    raw.configure_fixed_commands',lo)
s=s[:lo]+'''    original_update=raw._update_observation_command
    law_kind=np.array([g[7] for g in groups]).repeat(per)
    low_mask=torch.tensor(law_kind=='lowpass120ms',device=raw.device)
    cycle_mask=torch.tensor(law_kind=='cycle380ms',device=raw.device)
    filtered_velocity=None;velocity_ring=[];law_tick=-1
    def truthful_requested_commands():
        nonlocal filtered_velocity,law_tick,velocity_ring
        original_update()
        velocity=raw.robot.data.root_lin_vel_w[:,:2]
        if law_tick!=raw.common_step_counter:
            if filtered_velocity is None:filtered_velocity=velocity.clone()
            else:filtered_velocity=filtered_velocity+(1.-__import__('math').exp(-raw.step_dt/.12))*(velocity-filtered_velocity)
            velocity_ring.append(velocity.clone())
            if len(velocity_ring)>19:velocity_ring.pop(0)
            law_tick=raw.common_step_counter
        average=torch.stack(velocity_ring).mean(0)
        measured=torch.where(low_mask[:,None],filtered_velocity,velocity)
        measured=torch.where(cycle_mask[:,None],average,measured)
        heading_error=wrap_angle(raw.heading_target-yaw_from_quat(raw.robot.data.root_quat_w))
        course_error=wrap_angle(raw.heading_target-torch.atan2(measured[:,1],measured[:,0]))
        course_error=torch.where(measured.norm(dim=1)>.2,course_error,torch.zeros_like(course_error))
        rate=(heading_gain*heading_error+course_gain*course_error).clamp(-.2,.2)
        baseline=course_gain==0
        assert torch.equal(raw.commands[baseline,2],rate[baseline]),'Baseline heading command parity failed'
        raw.commands[:,2]=torch.where(raw.commands[:,:2].norm(dim=1)>.1,rate,torch.zeros_like(rate))
    raw._update_observation_command=truthful_requested_commands
''' +s[hi:]
lo=s.index('    rng=np.random.default_rng');hi=s.index('    assert ((q>=',lo)
s=s[:lo]+'''    offsets=[];velocities=[]
    for g in groups:
        rng=np.random.default_rng(g[6])
        offsets.append(rng.uniform(-.008,.008,(per,23)))
        velocities.append(rng.uniform(-.01,.01,(per,23)))
    q=raw.stand.expand(n,-1).clone()+torch.tensor(np.concatenate(offsets),device=raw.device,dtype=raw.stand.dtype)
    qd=torch.tensor(np.concatenate(velocities),device=raw.device,dtype=q.dtype)
''' +s[hi:]
s=s.replace('    raw._command_tick=-1;raw._last_history_tick=-1','    raw._command_tick=-1;raw._last_history_tick=-1\n    filtered_velocity=None;velocity_ring=[];law_tick=-1')
point=s.index('    reward_terms={}')
s=s[:point]+'''    substate={k:[] for k in ['root','raw_pose','raw_velocity','com_quaternion','action_command','action_phase','counter','timestamp']}
    original_scene_update=raw.scene.update
    physics_tick=0
    def post_physics_update(dt):
        nonlocal physics_tick
        original_scene_update(dt)
        assert abs(dt-.005)<1e-9
        physics_tick+=1
        data=raw.robot.data
        raw_pose=data._root_physx_view.get_root_transforms().clone()
        raw_pose=raw_pose[:,[0,1,2,6,3,4,5]]
        raw_velocity=data._root_physx_view.get_root_velocities().clone()
        values=dict(root=data.root_state_w,raw_pose=raw_pose,raw_velocity=raw_velocity,
            com_quaternion=data.root_com_quat_w,action_command=raw.action_commands,action_phase=raw.phase,
            counter=raw._sim_step_counter,timestamp=data._sim_timestamp)
        for k,v in values.items():substate[k].append(v.detach().cpu().numpy().copy() if torch.is_tensor(v) else v)
    raw.scene.update=post_physics_update
''' +s[point:]
point=s.index("    np.savez_compressed(out/'substep_motor_200hz.npz'")
s=s[:point]+'''    assert physics_tick==4000
    np.savez_compressed(out/'root_kinematics_200hz.npz',**{k:np.asarray(v) for k,v in substate.items()},
        dt=.005,robots_per_group=per,groups=np.array([g[0] for g in groups]))
''' +s[point:]
s=s.replace('initial_state_seed=3071','initial_state_seeds=[4191,4211]')
s=s.replace("scope='384 physical robots,20s,24 independently seeded paired states per cohort, four saved actors at0.15/1.5/3.5m/s and truthful heading/course/yaw command intervention; no action offsets,support,phase/input rewrite or new trained actor. Native constraints retained. Not a qualification gate. Course feedback uses actual instantaneous planar velocity; it is a diagnostic command law,not deployed guidance or proof of robustness.'", "scope='384 physical robots,20s,32 paired initial states per cohort at each of two independent seeds. Exact current actor, baseline,instantaneous course120ms velocity low-pass and380ms trailing-cycle mean course. Actual5ms post-physics root link quaternion/world COM velocity and direct PhysX parity captured. No action offsets,support,phase/input rewrite or learned actor. All external command laws diagnostic only; not qualification or certified inference.'")
(OUT/'timing_probe.py').write_text(s)
s=(PREV/'run_heading_probe.py').read_text().replace('heading_probe.py','timing_probe.py').replace("'heading_response'","'timing_response'").replace('+300','+420')
(OUT/'run_timing_probe.py').write_text(s)
print('Truthful timing probe and owned pause/resume tools prepared.')
