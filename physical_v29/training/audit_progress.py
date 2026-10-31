"""CPU-only throughput, gate trends and recorded joint-angle audit."""
from pathlib import Path
from collections import defaultdict
import json
import math
import statistics
import time
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
pointer = json.loads((HERE/'ACTIVE_RUN.json').read_text())
RUN = Path(pointer['run_directory'])
OUT = RUN/'progress_audit'
OUT.mkdir(exist_ok=True)
state = json.loads((RUN/'state.json').read_text())
groups = defaultdict(list)
for folder in sorted(RUN.glob('*_train_*')):
    result_path = folder/'result.json'
    progress_path = folder/'progress.jsonl'
    if not result_path.exists() or not progress_path.exists():
        continue
    result = json.loads(result_path.read_text())
    lines = [json.loads(line) for line in progress_path.read_text().splitlines() if line.strip()]
    if not result.get('completed') or not lines:
        continue
    stage = 'stand' if '_stand_' in folder.name else 'walk'
    n = result['num_envs']
    warm = lines[1:] or lines
    config = json.loads((folder/'run_config.json').read_text())
    groups[(stage,n)].append(dict(folder=folder.name, updates=len(lines),
        mean_update_s=statistics.mean(x['collect_seconds']+x['learn_seconds'] for x in warm),
        mean_collect_s=statistics.mean(x['collect_seconds'] for x in warm),
        seconds=result['seconds'], stage_speed=config['arguments']['speed']))

timing=[]
for (stage,n),rows in groups.items():
    update_s=statistics.mean(row['mean_update_s'] for row in rows)
    timing.append(dict(stage=stage,environments=n,blocks=len(rows),
        mean_update_seconds=update_s,policy_steps_per_update=n*24,
        environment_steps_per_second=n*24/update_s,
        simulation_seconds_per_wall_second=n*24*.02/update_s,
        wall_steps_per_second=sum(r['updates']*n*24 for r in rows)/sum(r['seconds'] for r in rows),
        mean_worker_seconds=statistics.mean(r['seconds'] for r in rows)))

trends=[]
for folder in sorted(RUN.glob('*_gate_*')):
    path=folder/'evaluation.json'
    if not path.exists():continue
    report=json.loads(path.read_text())
    for case in report['cases']:
        s=case['statistics']
        if '_walk_' in folder.name or (state['promotions'] and folder.name.startswith(('028_','029_'))):
            trends.append(dict(folder=folder.name,perturb=report['perturb'],command=s['command_m_s'],
                passed=case['passed'],survival=s['survival'],speed=s['forward_speed_m_s'],
                left_steps=s['steps_left'],right_steps=s['steps_right'],slip=s['slip_m_s'],
                support_outside=s['support_outside_fraction'],nonfoot=s['nonfoot_fraction'],
                crossing=s['crossing_fraction'],failures=case['failures']))

joint_audits=[]
latest_nominal=sorted([p for p in RUN.glob('*walk*nominal') if (p/'telemetry.npz').exists()])
standing=RUN/'028_stand_gate_014_nominal'
paths=([standing] if (standing/'telemetry.npz').exists() else [])+latest_nominal[-3:]
sys.path.insert(0,str(HERE.parent))
from cad_differential import CadDifferential
import torch
torch.set_num_threads(2)
geometry=json.loads((HERE.parent/'assets/differential_geometry.json').read_text())
plot_data=[]
for folder in paths:
    data=np.load(folder/'telemetry.npz')
    names=data['joint_names'].tolist();labels=data['case_labels'];speeds=data['case_speeds']
    metrics=data['metrics'];q=data['q'];qd=data['qd'];tau=data['tau']
    # q/tau sampled every fifth control tick; metrics every tick from t=2s.
    alive=metrics[::5,:,0]>.5
    assert alive.shape==q.shape[:2],(alive.shape,q.shape)
    for ci,speed in enumerate(speeds):
        if '_walk_' in folder.name and speed==0:continue
        mask=(labels==ci)[None,:]&alive
        sample_q=q[mask];sample_qd=qd[mask];sample_tau=tau[mask]
        rows=[]
        for j,name in enumerate(names):
            if any(k in name for k in ['hip_pitch','knee_pitch','ankle_pitch','hip_roll','ankle_roll','waist_']):
                values=np.degrees(sample_q[:,j]) if len(sample_q) else np.array([np.nan])
                rows.append(dict(joint=name,min_deg=float(np.nanmin(values)),median_deg=float(np.nanmedian(values)),
                    max_deg=float(np.nanmax(values)),p05_deg=float(np.nanpercentile(values,5)),p95_deg=float(np.nanpercentile(values,95)),
                    max_abs_rpm=float(np.abs(sample_qd[:,j]).max()*60/(2*np.pi)) if len(sample_q) else None))
        pairs=[]
        for key,spec in geometry.items():
            if not len(sample_q):continue
            ids=[names.index(n) for n in spec['joint_names']]
            tensor=torch.as_tensor(sample_q[:,ids],dtype=torch.float64)
            _,J,valid,error=CadDifferential(spec).state(tensor)
            valid_np=valid.numpy()
            peak=30. if key=='waist' else 45.
            if valid.any():
                # Post-step q is slightly later than the applied effort; reconstruction
                # is diagnostic, not exact actuator-clipping evidence.
                torque=torch.as_tensor(sample_tau[:,ids],dtype=torch.float64)[valid]
                motor=torch.linalg.solve(J[valid].transpose(1,2),torque[...,None]).squeeze(-1)/.95
                omega=(J[valid]@torch.as_tensor(sample_qd[:,ids],dtype=torch.float64)[valid,:,None]).squeeze(-1)
                pairs.append(dict(pair=key,valid_fraction=float(valid_np.mean()),
                    reconstructed_peak_motor_Nm=float(motor.abs().max()),assumed_peak_Nm=peak,
                    p99_abs_motor_Nm=float(torch.quantile(motor.abs().flatten(),.99)),
                    max_motor_rpm=float(omega.abs().max()*60/(2*math.pi)),
                    caveat='Approximate reconstruction from post-step pose; no logged motor-space current/thermal history'))
        all_alive=metrics[:,:,0]>.5
        case_mask=labels==ci
        survival_curve=all_alive[:,case_mask].mean(1)
        first_loss=int(np.flatnonzero(survival_curve<1.)[0]) if np.any(survival_curve<1.) else None
        joint_audits.append(dict(folder=folder.name,command=float(speed),joint_samples_before_first_failure=int(mask.sum()),
            first_trial_failure_time_s=2.+first_loss*.02 if first_loss is not None else None,
            final_survival=float(survival_curve[-1]),finite=bool(np.isfinite(sample_q).all()),joints=rows,differential_pairs=pairs))
        if speed>0:
            env_id=int(np.flatnonzero(case_mask)[0])
            plot_data.append((folder.name,names,q[:,env_id],alive[:,env_id]))

wall=dict()
for stage in ['stand','walk']:
    attempts=[a for a in state['attempts'] if ('_stand_' if stage=='stand' else '_walk_') in Path(a['folder']).name]
    wall[stage]=dict(completed_train_blocks=sum(a['operation']=='train' and a['completed'] for a in attempts),
        training_seconds=sum(a['seconds'] for a in attempts if a['operation']=='train'),
        evaluation_seconds=sum(a['seconds'] for a in attempts if a['operation']=='eval'))
summary=dict(created_unix=time.time(),run=str(RUN),live_status=state['status'],live_stage=state['stage'],
    completed_iterations=state['total_training_iterations'],timing=timing,trends=trends,joint_audits=joint_audits,
    stage_wall_time=wall,remaining_chain_budget_hours=max(0,(state['started_unix']+16*3600-time.time())/3600),
    reward_note='At command 0.35, stationary velocity receives exp(-0.35**2/0.20)=0.542 tracking fraction before posture gating; standing without walking can still earn substantial reward.',
    limits='Recorded joint data begins at 2s and ends at 20s; fall/reset samples excluded from angle statistics. Simplified contacts, no video-style or measured motor thermal qualification.')
(OUT/'audit.json').write_text(json.dumps(summary,indent=2))

if plot_data:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    title,names,q,alive=plot_data[-1]
    times=2.+np.arange(len(q))*.1
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for ax,kind in zip(axes,['hip_pitch','knee_pitch','ankle_pitch']):
        for side,color in [('left','#1265b0'),('right','#e17915')]:
            name=next(n for n in names if n.startswith(side+'_') and kind in n)
            values=np.degrees(q[:,names.index(name)]).copy();values[~alive]=np.nan
            ax.plot(times,values,label=side,color=color)
        ax.set_ylabel(kind.replace('_',' ')+' (deg)');ax.grid(alpha=.25);ax.legend(loc='upper right')
    axes[-1].set_xlabel('Rollout time (s); joint traces stop at first failure')
    fig.suptitle('Latest completed nominal walking check: '+title,fontsize=11)
    fig.tight_layout();fig.savefig(OUT/'walking_joint_angles.png',dpi=150);plt.close(fig)
print(json.dumps({k:summary[k] for k in ['live_status','live_stage','completed_iterations','timing','stage_wall_time','remaining_chain_budget_hours']},indent=2))
print('WALK TRENDS',json.dumps([x for x in trends if x['command']>0],indent=2))
print('JOINT AUDIT',json.dumps([dict(folder=x['folder'],command=x['command'],first_failure_s=x['first_trial_failure_time_s'],final_survival=x['final_survival'],joint_angles=x['joints'],differential_pairs=x['differential_pairs']) for x in joint_audits][-1:],indent=2))
