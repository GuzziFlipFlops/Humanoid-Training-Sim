"""Matched archived-walker/candidate tests with first-episode, pre-reset telemetry."""
import json
import math
import numpy as np
import torch
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, yaw_quat
from contact_state import CompletedStances


def evaluate(env, policies, candidate_scale, target, output, seed=47, perturb=False, duration=20., replicas=24):
    raw = env.unwrapped
    robot = raw.scene['robot']
    sensor = raw.scene.sensors['contact_forces']
    cases = [('reference_0.5', 'reference', .5), ('candidate_0.5', 'candidate', .5),
             ('reference_1.0', 'reference', 1.), ('candidate_1.0', 'candidate', 1.),
             ('candidate_1.8', 'candidate', 1.8), ('candidate_target', 'candidate', target)]
    n, device = raw.num_envs, raw.device
    assert n == replicas * len(cases)
    speed = torch.tensor([s for _, _, s in cases], device=device).repeat_interleave(replicas)
    scales = torch.tensor([.35 if p == 'reference' else candidate_scale for _, p, _ in cases],
                          device=device).repeat_interleave(replicas)
    raw.action_manager.get_term('joint_pos')._scale = scales[:, None].expand(n, 22).clone()
    command = raw.command_manager.get_term('base_velocity')
    # Same physical initial states for every policy/speed; no startup domain randomization.
    generator = torch.Generator(device=device).manual_seed(seed)
    perturb_q = (torch.rand((replicas, 22), generator=generator, device=device) - .5) * .04
    rolls = (torch.rand(replicas, generator=generator, device=device) - .5) * .05
    qinit = robot.data.default_joint_pos + perturb_q.repeat(len(cases), 1)
    initial = robot.data.default_root_state.clone()
    initial[:, :3] += raw.scene.env_origins
    rr = rolls.repeat(len(cases))
    initial[:, 3:7] = quat_from_euler_xyz(rr, torch.zeros_like(rr), torch.zeros_like(rr))
    robot.write_root_state_to_sim(initial)
    robot.write_joint_state_to_sim(qinit, torch.zeros_like(qinit))
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    fi = [robot.body_names.index(s) for s in ('left_foot', 'right_foot')]
    ki = [robot.body_names.index(s) for s in ('left_shank', 'right_shank')]
    si = [sensor.body_names.index(s) for s in ('left_foot', 'right_foot')]
    ti, wi = robot.body_names.index('torso'), robot.joint_names.index('waist_pitch')
    nonfeet = [i for i, s in enumerate(sensor.body_names) if s not in ('left_foot', 'right_foot')]
    fields = {}; width = 0
    for name, size in [('q',22),('target',22),('qd',22),('torque_est',22),('velocity',3),
                       ('gravity_z',1),('torso_tilt',1),('fz',2),('slip',2),('separation',2),
                       ('target_oob',1),('waist_oob',1),('speed_saturation',1),('nonfoot',1)]:
        fields[name] = [width, width+size]; width += size
    frames = round(duration / raw.step_dt)
    records = np.zeros((frames, n, width), np.float32)
    alive_log = np.zeros((frames, n), bool)
    fell = torch.zeros(n, dtype=torch.bool, device=device)
    active = False; counter = 0; cached = None
    update = raw.scene.update

    def hook(dt):
        nonlocal counter, cached
        update(dt)
        if not active:
            return
        counter += 1
        if counter % raw.cfg.decimation:
            return
        d = robot.data
        forces = sensor.data.net_forces_w
        heading = yaw_quat(d.root_quat_w)
        pos = d.body_pos_w[:, fi + ki] - d.root_pos_w[:, None]
        local = quat_apply_inverse(heading[:,None].expand(-1,4,-1).reshape(-1,4),
                                   pos.reshape(-1,3)).reshape(n,4,3)
        up = torch.zeros((n,3), device=device); up[:,2] = 1.
        tilt = torch.acos(quat_apply_inverse(d.body_quat_w[:,ti],up)[:,2].clamp(-1.,1.))
        bounds = d.joint_pos_limits
        outside = (d.joint_pos_target < bounds[...,0]) | (d.joint_pos_target > bounds[...,1])
        cached = torch.cat([d.joint_pos, d.joint_pos_target, d.joint_vel, d.applied_torque,
            d.root_lin_vel_b, d.projected_gravity_b[:,2:3], tilt[:,None], forces[:,si,2].clamp_min(0.),
            d.body_lin_vel_w[:,fi,:2].norm(dim=-1),
            torch.stack((local[:,0,1]-local[:,1,1],local[:,2,1]-local[:,3,1]),1),
            outside.float().mean(1,keepdim=True), outside[:,wi:wi+1].float(),
            (d.joint_vel.abs() >= .95*d.soft_joint_vel_limits).float().mean(1,keepdim=True),
            (forces[:,nonfeet].norm(dim=-1).amax(1)>5.).float()[:,None]],1).clone()

    raw.scene.update = hook
    with torch.inference_mode():
        for k in range(frames):
            command.vel_command_b[:] = 0.
            command.vel_command_b[:,0] = speed
            command.is_standing_env[:] = False
            if perturb and k in (round(8./raw.step_dt), round(13./raw.step_dt)):
                vel = robot.data.root_vel_w.clone()
                # Equal external lateral kicks across the matched cases.
                vel[:,1] += .15 if k == round(8./raw.step_dt) else -.15
                robot.write_root_velocity_to_sim(vel)
            obs = raw.observation_manager.compute(update_history=False)
            actions = torch.zeros((n,22), device=device)
            for i, (_, policy, _) in enumerate(cases):
                sl = slice(i*replicas,(i+1)*replicas)
                actions[sl] = policies[policy]({key:value[sl] for key,value in obs.items()})
            active = True
            _, _, done, extras = env.step(actions)
            active = False
            terminated = done.bool() & ~extras.get('time_outs', torch.zeros_like(done)).bool()
            fell |= terminated
            records[k] = cached.cpu().numpy()
            alive_log[k] = (~fell).cpu().numpy()
    raw.scene.update = update
    np.savez_compressed(output/'telemetry.npz', samples=records, prefall_valid=alive_log,
                        initial_q=qinit.cpu().numpy(), initial_root=initial.cpu().numpy())
    assert np.isfinite(records).all(), 'Nonfinite evaluation telemetry'
    rows = {}
    velocity_limits = robot.data.soft_joint_vel_limits.detach().cpu().numpy()
    for i, (label, _, command_speed) in enumerate(cases):
        sl = slice(i*replicas,(i+1)*replicas)
        valid = alive_log[:,sl].copy(); valid[:round(3./raw.step_dt)] = False
        def f(name):
            a,b = fields[name]; return records[:,sl,a:b]
        contact = f('fz') > 5.
        tracker = CompletedStances(replicas, 'cpu', raw.step_dt)
        filtered = np.zeros_like(contact)
        durations = [[],[]]
        individual = [[[],[]] for _ in range(replicas)]
        for t in range(frames):
            complete, dur = tracker.update(torch.from_numpy(contact[t]))
            filtered[t] = tracker.stable.numpy()
            for leg in range(2):
                good = complete[:,leg].numpy() & valid[t]
                if t*raw.step_dt < 4.5: good[:] = False
                for idx in np.flatnonzero(good):
                    value=float(dur[idx,leg]); durations[leg].append(value); individual[idx][leg].append(value)
        asym = [abs(np.mean(x[0])-np.mean(x[1]))/(np.mean(x[0])+np.mean(x[1]))
                for x in individual if len(x[0])>=3 and len(x[1])>=3]
        def mean(name):
            values=f(name)[valid]; return float(values.mean()) if len(values) else None
        def aggregate(values):
            values=values[valid]; return float(values.mean()) if len(values) else None
        load=(f('fz')*valid[:,:,None]).sum((0,1)); impulse_share=float(load[0]/max(load.sum(),1e-8))
        slipden=(contact*valid[:,:,None]).sum()
        joint_saturation = {}
        joint_velocity_p95 = {}
        for joint_idx, joint_name in enumerate(robot.joint_names):
            joint_values = np.abs(f('qd')[..., joint_idx])[valid]
            limit = float(np.mean(velocity_limits[sl, joint_idx]))
            joint_saturation[joint_name] = float(np.mean(joint_values >= .95 * limit)) if len(joint_values) else None
            joint_velocity_p95[joint_name] = float(np.percentile(joint_values, 95)) if len(joint_values) else None
        upright_cone = math.radians(8. + min(abs(command_speed), 6.))
        rows[label] = dict(target=command_speed, fall_fraction=float(1-alive_log[-1,sl].mean()),
            speed_mae=aggregate(np.abs(f('velocity')[...,0]-command_speed)),
            mean_speed=aggregate(f('velocity')[...,0]), lateral_speed=aggregate(np.abs(f('velocity')[...,1])),
            torso_tilt_deg=mean('torso_tilt')*180/np.pi if valid.any() else None,
            torso_excess_fraction=aggregate(f('torso_tilt')[...,0] > upright_cone),
            slip=float((f('slip')*contact*valid[:,:,None]).sum()/max(slipden,1)),
            foot_crossing=aggregate(f('separation')[...,0] < .08),
            knee_crossing=aggregate(f('separation')[...,1] < .075),
            nonfoot=mean('nonfoot'), target_oob=mean('target_oob'), waist_oob=mean('waist_oob'),
            speed_saturation=mean('speed_saturation'),
            filtered_flight_fraction=aggregate(filtered.sum(2)==0),
            stance_asymmetry=float(np.mean(asym)) if asym else None,
            valid_step_replicas=len(asym), stance_left=float(np.mean(durations[0])) if durations[0] else None,
            stance_right=float(np.mean(durations[1])) if durations[1] else None,
            left_impulse_share=impulse_share, evaluated_seconds=float(valid.sum()*raw.step_dt/replicas),
            joint_velocity_saturation=joint_saturation, joint_velocity_p95=joint_velocity_p95,
            joint_velocity_limits={name:float(np.mean(velocity_limits[sl, j]))
                                   for j,name in enumerate(robot.joint_names)})
    report = dict(rows=rows, seed=seed, lateral_kicks=perturb, replicas=replicas, duration=duration,
                  fields=fields, joints=robot.joint_names, candidate_scale=candidate_scale,
                  torque_note='Implicit drive estimates, not measured motor torques.')
    (output/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
