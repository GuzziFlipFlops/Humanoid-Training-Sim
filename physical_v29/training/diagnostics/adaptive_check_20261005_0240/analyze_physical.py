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
    group_labels=np.repeat(np.arange(len(data['groups'])),int(data['robots_per_group']))
    labels=np.tile(group_labels,len(data['q']))
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
        summaries.append(dict(group=str(data['groups'][int(command)]), samples_per_motor=int(mask.sum()),
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
                scope='All robots, saved 50Hz physical q/qd, whole ten seconds; exact pose-dependent reductions. Not substep clipping/torque attribution or a controlled ceiling experiment.')



results={}
for folder in [OUT/'physical_response',OUT/'hybrid_response']:
 z=np.load(folder/'telemetry.npz');names=list(z['joint_names']);per=int(z['robots_per_group']);post=z['time']>=3.;groups=[]
 for i,name in enumerate(z['groups']):
  idx=slice(i*per,(i+1)*per);phase=z['phase'][post,idx];flight_window=np.remainder(phase,.5)>=.43
  force=z['force'][post,idx];contact=z['contact'][post,idx];root=z['root'][post,idx];head=z['q'][post,idx][:,:,[names.index('head_yaw'),names.index('head_pitch')]]
  w,x,y,v=[root[:,:,j] for j in [3,4,5,6]];yaw=np.arctan2(2*(w*v+x*y),1-2*(y*y+v*v))
  groups.append(dict(group=str(name),head_combined_rms_deg=float(np.rad2deg(np.sqrt(np.mean(np.sum(head**2,axis=-1))))),head_both_within5deg_fraction=float((np.abs(head)<np.deg2rad(5)).all(-1).mean()),absolute_lateral_world_speed=float(np.abs(root[:,:,8]).mean()),final_heading_abs_mean_deg=float(np.abs(np.rad2deg(yaw[-1])).mean()),loaded_leg_in_intended_flight_fraction=float(contact[flight_window].any(-1).mean()) if flight_window.any() else None,greater_force_in_intended_flight_N=float(force[flight_window].max(-1).mean()) if flight_window.any() else None))
 results[folder.name]=dict(behavior=groups,motors=motor_stats(folder))
(OUT/'physical_timing_and_motors.json').write_text(json.dumps(results,indent=2))
print(json.dumps({k:v['behavior'] for k,v in results.items()},indent=2))
