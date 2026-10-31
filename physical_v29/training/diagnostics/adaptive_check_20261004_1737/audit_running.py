"""Saved-physics running diagnosis, CPU only; never edits a frozen run.

Motor speeds use each saved pose's exact CAD Jacobian or the declared reduction.
Sampled speeds cannot certify a velocity ceiling or count substep clipping.
Command intervention holds a reconstructed observed state fixed, not physics.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

torch.set_num_threads(1)
OUT = Path(__file__).resolve().parent
HERE = OUT.parents[1]
pointer = json.loads((HERE/'ACTIVE_RUN.json').read_text())
RUN = Path(pointer['run'])
FROZEN = RUN/'source/physical_v29'
sys.path[:0] = [str(FROZEN), str(FROZEN/'training')]
from cad_differential import CadDifferential
from unitree_reward_port import gait_parameters


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def inverse(q, v):
    t = 2*torch.cross(q[..., 1:], v, dim=-1)
    return v-q[..., :1]*t+torch.cross(q[..., 1:], t, dim=-1)


def motor_stats(folder):
    records = read(FROZEN/'assets/actuator_limits.json')
    geometry = read(FROZEN/'assets/differential_geometry.json')
    data = np.load(folder/'telemetry.npz', allow_pickle=False)
    names = list(data['joint_names'])
    q = torch.from_numpy(data['q']).reshape(-1, len(names))
    qd = torch.from_numpy(data['qd']).reshape_as(q)
    case_speeds = data['case_speeds'][data['case_labels']]
    labels = np.tile(case_speeds, len(data['q']))
    motors = {}
    paired = set()
    for name, spec in geometry.items():
        ids = [names.index(j) for j in spec['joint_names']]
        paired.update(spec['joint_names'])
        _, J, valid, closure = CadDifferential(spec).state(q[:, ids])
        speed = torch.matmul(J, qd[:, ids, None]).squeeze(-1).numpy()
        record = records[spec['joint_names'][0]]
        for i in range(2):
            motors[f'{name}/motor_{i+1}'] = (speed[:, i], record['motor_speed_cap_RPM']*2*np.pi/60,
                                            'factory-geared output, theta_dot=J(q) q_dot')
        assert bool(valid.all()), (name, 'Invalid saved pose')
        assert float(closure.max()) < .01
    for i, name in enumerate(names):
        if name in paired:
            continue
        record = records[name]
        reduction = record.get('reduction', record.get('physical_external_reduction', 1.))
        motors[name] = (qd[:, i].numpy()*reduction, record['velocity_rad_s']*reduction,
                       f'joint speed x declared reduction {reduction}; same stage as cap')
    summaries = []
    for command in sorted(set(labels)):
        mask = labels == command
        items = []
        for name, (speed, cap, units) in motors.items():
            ratio = np.abs(speed[mask])/cap
            items.append(dict(name=name, cap_rad_s=cap, cap_RPM=cap*60/(2*np.pi),
                              max_speed_RPM=float(np.abs(speed[mask]).max()*60/(2*np.pi)),
                              p99_cap_fraction=float(np.quantile(ratio, .99)), maximum_cap_fraction=float(ratio.max()),
                              fraction_above_90_percent=float((ratio>=.9).mean()),
                              fraction_at_or_above_cap=float((ratio>=1.).mean()), units=units))
        summaries.append(dict(command_m_s=float(command), samples_per_motor=int(mask.sum()),
                              motors=sorted(items, key=lambda x:x['p99_cap_fraction'], reverse=True)))
    # Finite-difference theta along measured q_dot validates units and Jacobian use.
    derivative_checks=[]
    for name,spec in geometry.items():
        ids=[names.index(j) for j in spec['joint_names']]
        sample=torch.arange(0,len(q),max(1,len(q)//64))[:64]
        pose=q[sample][:,ids].double();velocity=qd[sample][:,ids].double()
        model=CadDifferential(spec);h=1e-6
        _,J,valid,_=model.state(pose)
        plus=model.state(pose+h*velocity)[0];minus=model.state(pose-h*velocity)[0]
        diff=torch.atan2(torch.sin(plus-minus),torch.cos(plus-minus))/(2*h)
        analytic=(J@velocity[...,None]).squeeze(-1)
        error=float((diff-analytic).abs().max())
        assert error<1e-5,(name,error)
        derivative_checks.append(dict(pair=name,poses=len(sample),max_velocity_error_rad_s=error,passed=True))
    return dict(folder=str(folder), commands=summaries,jacobian_directional_checks=derivative_checks,
                scope='All robots, saved 10Hz physical q/qd after 2s; exact pose-dependent reductions. Not substep clipping/torque attribution or a controlled ceiling experiment.')


def trace_stats(folder):
    data = np.load(folder/'gait_diagnostics_50hz.npz', allow_pickle=False)
    t = data['time']
    commands = data['commands']
    result = []
    for command in sorted(set(commands)):
        mask = commands == command
        contact = data['contact'][t>=2.][:, mask]
        force = data['foot_force'][t>=2.][:, mask]
        clear = data['clearance'][t>=2.][:, mask]
        both_unloaded = ~contact.any(-1)
        period = 1/(.9+.5*command) if command>=1.5 else .8
        duty = .43 if command>=1.5 else .55
        phase = np.remainder(t[t>=2.]/period, 1.)
        expected_stance = np.remainder(phase[:, None]+np.array([0., .5]), 1.) < duty
        expected_flight = ~expected_stance.any(-1)
        item = dict(command_m_s=float(command), sampled_robots=int(mask.sum()),
                    force_contact_consistency=bool(np.array_equal(contact,force > 34.46334144435157*9.81*.05)),
                    flight_fraction=float(both_unloaded.mean()),
                    both_force_below_1N_fraction=float((np.abs(force)<1.).all(-1).mean()),
                    both_soles_above_1mm_fraction=float((clear>.001).all(-1).mean()),
                    both_soles_above_5mm_fraction=float((clear>.005).all(-1).mean()),
                    stance_fraction_left=float(contact[..., 0].mean()), stance_fraction_right=float(contact[..., 1].mean()),
                    both_contact_fraction=float(contact.all(-1).mean()),
                    expected_flight_fraction=float(expected_flight.mean()),
                    loaded_leg_in_expected_flight_fraction=float(contact[expected_flight].any(-1).mean()) if expected_flight.any() else None,
                    lower_foot_force_in_expected_flight_N=float(force[expected_flight].min(-1).mean()) if expected_flight.any() else None,
                    greater_foot_force_in_expected_flight_N=float(force[expected_flight].max(-1).mean()) if expected_flight.any() else None,
                    observed_action_command_max_difference=float(np.abs(data['observed_command'][:,mask]-data['action_command'][:,mask]).max()),
                    raw_action_above_runtime_clip_3_fraction=float((np.abs(data['raw_action'][:,mask])>3).mean()),
                    action_clip_3_max_difference=float(np.abs(np.clip(data['raw_action'][:,mask],-3,3)-data['applied_action'][:,mask]).max()),
                    joint_action_statistics=[dict(name=str(name),raw_abs_max=float(np.abs(data['raw_action'][:,mask,k]).max()),
                        fraction_above_clip_3=float((np.abs(data['raw_action'][:,mask,k])>3).mean()),
                        actual_q_min_rad=float(data['q'][:,mask,k].min()),actual_q_max_rad=float(data['q'][:,mask,k].max()),
                        target_min_rad=float(data['target'][:,mask,k].min()),target_max_rad=float(data['target'][:,mask,k].max()))
                        for k,name in enumerate(data['joint_names'])])
        result.append(item)
    return dict(folder=str(folder), commands=result, scope='First16 robots, every 50Hz control tick; flight measures use three physical force/clearance formulations. Expected phase uses saved elapsed time, not an inference success criterion.')


def actor_response(folder):
    config = read(folder/'run_config.json')
    checkpoint = Path(config['arguments']['checkpoint'])
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['actor_state_dict']
    actor = torch.nn.Sequential(torch.nn.Linear(400,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),
                                torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,23))
    actor.load_state_dict({k.removeprefix('mlp.'):v for k,v in state.items() if k.startswith('mlp.')})
    actor.eval()
    data = np.load(folder/'gait_diagnostics_50hz.npz',allow_pickle=False)
    motion = np.load(FROZEN/'training/assets/motion_reference.npz',allow_pickle=False)
    names = list(data['joint_names'])
    stand = torch.from_numpy(motion['stand'][[list(motion['names']).index(n) for n in names]])
    roots = torch.from_numpy(data['root_state'][:-1])
    quat = roots[...,3:7]
    commands = torch.from_numpy(data['observed_command'][1:])
    period, _ = gait_parameters(commands.reshape(-1,3))
    period = period.reshape(commands.shape[:2])
    # These independent trials have no reset: sample i's action observes sample i-1's post-step state.
    assert not data['done'][2:].any(), 'Reconstruction requires uninterrupted traces'
    ticks = torch.arange(1,len(data['time']),dtype=torch.float32)[:,None]
    phase = torch.remainder(ticks*.02/period,1.)
    ph = torch.stack([torch.sin(2*torch.pi*phase),torch.cos(2*torch.pi*phase)],-1)*(commands[...,0:1]>.1)
    gravity = torch.zeros_like(roots[...,10:13]);gravity[...,2]=-1.
    frames = torch.cat([inverse(quat,roots[...,10:13])*.2, inverse(quat,gravity),commands,
                        torch.from_numpy(data['q'][:-1])-stand,torch.from_numpy(data['qd'][:-1])*.05,
                        torch.from_numpy(data['applied_action'][:-1]),ph],-1)
    # Exact five complete observed frames; do not reconstruct the initial reset zeros.
    history = torch.stack([frames[i-4:i+1].transpose(0,1).reshape(16,400) for i in range(4,len(frames))])
    selected = history[20::7].reshape(-1,400)
    expected = torch.from_numpy(data['raw_action'][5:])[20::7].reshape(-1,23)
    with torch.no_grad():pred = actor(selected)
    error = float((pred-expected).abs().max())
    reconstruction = dict(max_abs_action_error=error, rms_action_error=float((pred-expected).square().mean().sqrt()),
                          samples=len(selected), passed=error<.0005)
    result = dict(folder=str(folder), checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  reconstruction=reconstruction, scope='CPU actor at fixed reconstructed physical states. Command derivatives do not prove closed-loop acceleration or solve running.')
    if not reconstruction['passed']:
        result['intervention']='Withheld because reconstructed observations do not match saved policy actions'
        return result
    interventions=[]
    for command in [.15,.35,.65,1.,1.5,2.]:
        counter = selected.clone().reshape(-1,5,80)
        counter[:,:,6] = command
        # Keep phase/state/yaw feedback fixed to isolate the velocity-command channel.
        counter=counter.reshape(-1,400).requires_grad_(True)
        action=actor(counter)
        grad=[]
        for joint in ['left_hip_pitch','right_hip_pitch','left_knee_pitch','right_knee_pitch','left_ankle_pitch','right_ankle_pitch']:
            g=torch.autograd.grad(action[:,names.index(joint)].sum(),counter,retain_graph=True)[0]
            clipped_g=torch.autograd.grad(action[:,names.index(joint)].clamp(-3,3).sum(),counter,retain_graph=True)[0]
            grad.append(dict(joint=joint,mean_abs_derivative_per_m_s=float(g.reshape(-1,5,80)[:,:,6].sum(1).abs().mean()),
                             clipped_mean_abs_derivative_per_m_s=float(clipped_g.reshape(-1,5,80)[:,:,6].sum(1).abs().mean()),
                             fraction_raw_mean_outside_clip=float((action[:,names.index(joint)].abs()>3).float().mean())))
        interventions.append(dict(command_m_s=command,mean_action_RMS_difference=float((action.detach()-pred).square().mean().sqrt()),
                                  clipped_action_RMS_difference=float((action.detach().clamp(-3,3)-pred.clamp(-3,3)).square().mean().sqrt()),
                                  joint_gradients=grad))
    result['fixed_state_command_interventions']=interventions
    return result


def tracking_sensitivity(folder):
    cases=read(folder/'result.json')['cases']
    rows=[]
    for case in cases:
        s=case['statistics'];command=s['command_m_s']
        if not command:continue
        v=s['forward_speed_m_s'];l=s['lateral_speed_m_s']
        delta=v-command
        # Pinned G1 exp tracking kernel, before posture scaling. Width .5m/s.
        rate=4*np.exp(-(delta**2+l**2)/.5**2)
        rows.append(dict(command_m_s=command,actual_body_forward_m_s=v,signed_forward_m_s=s['reset_frame_forward_speed_m_s'],
                         kernel_rate_at_mean=rate, derivative_at_mean_per_m_s=-2*delta/.5**2*rate,
                         perfect_speed_gain_before_posture=4-rate, note='At mean velocity, not mean per-tick reward; nonlinear averaging matters.'))
    return rows


if __name__=='__main__':
    latest=Path(sys.argv[1]) if len(sys.argv)>1 else sorted(p.parent for p in RUN.glob('*_gate_*_nominal/result.json') if read(p).get('completed'))[-1]
    assert 'run_150' in latest.name
    motors=motor_stats(latest)
    traces=trace_stats(latest)
    actor=actor_response(latest)
    result=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),run=str(RUN),motor_audit=motors,trace_audit=traces,
                actor_audit=actor,tracking_sensitivity=tracking_sensitivity(latest),
                conclusion='Diagnostic only; no motor limit, reward, source, policy or independent gate was changed. No velocity-ceiling qualification.')
    (OUT/'running_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(folder=latest.name,trace={k:v for k,v in traces['commands'][-1].items() if k!='joint_action_statistics'},reconstruction=actor['reconstruction'],
                         running_motor_top=motors['commands'][-1]['motors'][:5],tracking=result['tracking_sensitivity'][:2]),indent=2))
