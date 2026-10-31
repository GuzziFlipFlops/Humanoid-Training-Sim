"""Analyze physics-rate diagnostic recordings without launching Isaac or training."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

def intervals(x):
    edges = np.flatnonzero(np.diff(np.r_[False,x,False].astype(int)))
    return list(zip(edges[::2],edges[1::2]))

def cleaned_contacts(x, samples):
    x = x.copy()
    # Bridge sub-30ms force dropouts, then discard sub-30ms isolated contact bursts.
    for lo,hi in intervals(~x):
        if lo > 0 and hi < len(x) and hi-lo < samples:
            x[lo:hi] = True
    for lo,hi in intervals(x):
        if hi-lo < samples:
            x[lo:hi] = False
    return x

def safe_mean(x):
    return float(np.mean(x)) if len(x) else None

def write_csv(path, rows):
    if rows:
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def euler(q):
    w,x,y,z = np.moveaxis(q,-1,0)
    return np.stack([np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),
        np.arcsin(np.clip(2*(w*y-z*x),-1,1)),
        np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))],axis=-1)

def main(root):
    meta=json.loads((root/'manifest.json').read_text())
    assert meta['completed']
    x=np.load(root/'physics.npy',mmap_mode='r')
    par=np.load(root/'runtime_parameters.npz')
    fall=np.load(root/'episodes.npz')['first_fall_s']
    dt=meta['physics_dt']; ts=(np.arange(len(x))+1)*dt
    names=meta['joint_names']; fields=meta['fields']
    def field(name,ids):
        a,b=fields[name]
        return np.asarray(x[:,ids,a:b])
    summaries=[]; joint_rows=[]; leg_rows=[]; phase_rows=[]; parameter_rows=[]
    all_finite=True
    for case in meta['cases']:
        ids=case['env_ids']; count=len(ids)
        q=field('q',ids); qd=field('qd',ids); target=field('target',ids)
        tau=field('torque_est',ids); req=field('torque_requested_est',ids)
        fv=field('foot_vel_w',ids).reshape(len(x),count,2,3)
        fp=field('foot_pos_h',ids).reshape(len(x),count,2,3)
        force=field('foot_force_w',ids).reshape(len(x),count,2,3)
        fz=np.maximum(force[:,:,:,2],0)
        vel=field('root_vel_b',ids)
        cut=np.where(np.isnan(fall[ids]),meta['duration_s']+dt,fall[ids])
        valid=(ts[:,None]>3.) & (ts[:,None]<cut[None,:])
        all_finite &= bool(np.isfinite(q).all() and np.isfinite(qd).all() and np.isfinite(force).all())
        loaded=fz>5.
        contact=np.zeros_like(loaded)
        foot_data=[dict(stance=[],air=[],period=[],raw_brief=[],raw_n=0,clean_n=0) for _ in range(2)]
        curves={foot:{'q':[],'target':[]} for foot in range(2)}
        for env in range(count):
            for foot in range(2):
                raw_contact=loaded[:,env,foot]
                # Clean only within the first episode; never bridge across a reset.
                length=min(len(x),int(cut[env]/dt))
                contact[:length,env,foot]=cleaned_contacts(raw_contact[:length],round(.03/dt))
                stats=foot_data[foot]
                for a,b in intervals(raw_contact[:length]):
                    if a*dt>3. and b < length and b*dt<cut[env]:
                        stats['raw_n']+=1
                        stats['raw_brief'].append((b-a)*dt<.06)
                clean=contact[:length,env,foot]
                starts=[]
                for a,b in intervals(clean):
                    if a*dt>3. and b < length and b*dt<cut[env]:
                        stats['stance'].append((b-a)*dt); stats['clean_n']+=1; starts.append(a)
                for a,b in intervals(~clean):
                    if a*dt>3. and b < length and b*dt<cut[env]: stats['air'].append((b-a)*dt)
                for a,b in zip(starts[:-1],starts[1:]):
                    period=(b-a)*dt
                    if .15<period<2.:
                        stats['period'].append(period)
                        sample=np.linspace(a,b-1,101)
                        for key,arr in [('q',q),('target',target)]:
                            curves[foot][key].append(np.stack([np.interp(sample,np.arange(a,b),arr[a:b,env,j]) for j in range(22)],axis=-1))
        w=valid[:,:,None]
        impulse=(fz*w).sum((0,1))*dt
        support=(contact.sum(-1))
        rollpitch=euler(field('torso_quat_w',ids))*180/np.pi
        pelvis_angles=euler(field('root_quat_w',ids))*180/np.pi
        total_valid=int(valid.sum())
        sample_q=q[valid]; sample_qd=qd[valid]; sample_tau=tau[valid]; sample_req=req[valid]
        vmax=np.broadcast_to(par['velocity_limits'][ids],qd.shape)[valid]
        taumax=np.broadcast_to(par['effort_limits'][ids],tau.shape)[valid]
        plim=np.broadcast_to(par['position_limits'][ids],q.shape+(2,))[valid]
        # Evaluate current bad-order ankle proxy against intended within-leg grouping without changing policy.
        current_names=meta['reward_joint_resolution']['ankle_differential.asset_cfg']
        intended_names=['left_ankle_pitch','left_ankle_roll','right_ankle_pitch','right_ankle_roll']
        proxy={}
        for title,order in [('current',current_names),('within_leg',intended_names)]:
            ii=[names.index(a) for a in order]
            tt=sample_tau[:,ii].reshape(-1,2,2); vv=sample_qd[:,ii].reshape(-1,2,2)
            mixed_t=np.stack([tt[:,:,0]+tt[:,:,1],tt[:,:,0]-tt[:,:,1]],-1)
            mixed_v=np.stack([vv[:,:,0]+vv[:,:,1],vv[:,:,0]-vv[:,:,1]],-1)
            pen=(np.maximum(np.abs(mixed_t)-150.88,0)/150.88)**2+(np.maximum(np.abs(mixed_v)-9.56,0)/9.56)**2
            proxy[title+'_active_fraction']=float(np.mean(np.any(pen>1e-8,axis=(1,2)))) if len(pen) else None
            proxy[title+'_mean_penalty_unweighted']=safe_mean(pen.sum((1,2)))
        row=dict(case=case['name'], target_mps=case['target_mps'], envs=count,
            fell_count=int(np.isfinite(fall[ids]).sum()),
            first_fall_s_min=float(np.nanmin(fall[ids])) if np.isfinite(fall[ids]).any() else None,
            valid_prefall_env_seconds=total_valid*dt, mean_vx_mps=safe_mean(vel[:,:,0][valid]),
            speed_mae_mps=safe_mean(np.abs(vel[:,:,0][valid]-case['target_mps'])),
            lateral_speed_rms_mps=float(np.sqrt(np.mean(vel[:,:,1][valid]**2))) if total_valid else None,
            left_vertical_impulse_share=float(impulse[0]/impulse.sum()) if impulse.sum() else None,
            left_right_stance_s=[safe_mean(s['stance']) for s in foot_data],
            left_right_air_s=[safe_mean(s['air']) for s in foot_data],
            left_right_stride_period_s=[safe_mean(s['period']) for s in foot_data],
            left_right_contact_fraction=[safe_mean(contact[:,:,i][valid]) for i in range(2)],
            left_right_brief_raw_contact_fraction=[safe_mean(s['raw_brief']) for s in foot_data],
            flight_fraction=safe_mean((support==0)[valid]), double_support_fraction=safe_mean((support==2)[valid]),
            left_right_slip_mps=[safe_mean(np.linalg.norm(fv[:,:,i,:2],axis=-1)[valid&contact[:,:,i]]) for i in range(2)],
            torso_roll_pitch_mean_deg=np.mean(rollpitch[valid],axis=0)[:2].tolist() if total_valid else None,
            torso_roll_pitch_std_deg=np.std(rollpitch[valid],axis=0)[:2].tolist() if total_valid else None,
            pelvis_roll_pitch_mean_deg=np.mean(pelvis_angles[valid],axis=0)[:2].tolist() if total_valid else None,
            signed_foot_width_p01_m=float(np.percentile((fp[:,:,0,1]-fp[:,:,1,1])[valid],1)) if total_valid else None,
            com_lateral_mean_m=safe_mean(field('com_h',ids)[:,:,1][valid]),
            nonfoot_contact_fraction=safe_mean((field('nonfoot_force_max',ids)[:,:,0]>5)[valid]),
            ankle_proxy=proxy)
        summaries.append(row)
        for j,name in enumerate(names):
            if not total_valid: continue
            err=(target-q)[valid][:,j]*180/np.pi
            r=dict(case=case['name'],joint=name,q_mean_deg=float(sample_q[:,j].mean()*180/np.pi),
                q_p05_deg=float(np.percentile(sample_q[:,j],5)*180/np.pi),q_p95_deg=float(np.percentile(sample_q[:,j],95)*180/np.pi),
                target_mean_deg=float(target[valid][:,j].mean()*180/np.pi),
                target_rom90_deg=float(np.ptp(np.percentile(target[valid][:,j],[5,95]))*180/np.pi),
                tracking_mae_deg=float(np.abs(err).mean()), tracking_p95_deg=float(np.percentile(np.abs(err),95)),
                speed_abs_p99_rad_s=float(np.percentile(np.abs(sample_qd[:,j]),99)),
                speed_limit_fraction=float(np.mean(np.abs(sample_qd[:,j])>=.95*vmax[:,j])),
                torque_est_abs_p99_nm=float(np.percentile(np.abs(sample_tau[:,j]),99)),
                torque_est_rms_nm=float(np.sqrt(np.mean(sample_tau[:,j]**2))),
                torque_est_limit_fraction=float(np.mean(np.abs(sample_tau[:,j])>=.95*taumax[:,j])),
                torque_est_clipped_fraction=float(np.mean(np.abs(sample_req[:,j])>taumax[:,j]+1e-3)),
                target_outside_hard_limit_fraction=float(np.mean((target[valid][:,j]<plim[:,j,0])|(target[valid][:,j]>plim[:,j,1]))))
            joint_rows.append(r)
        for suffix in ['hip_yaw','hip_roll','hip_pitch','knee_pitch','ankle_pitch','ankle_roll']:
            li,ri=[names.index(f'{side}_{suffix}') for side in ['left','right']]
            sign=-1. if ('roll' in suffix or 'yaw' in suffix) else 1.
            rr=dict(case=case['name'],joint_pair=suffix,left_cycles=len(curves[0]['q']),right_cycles=len(curves[1]['q']))
            if curves[0]['q'] and curves[1]['q']:
                for key in ['q','target']:
                    left=np.mean(curves[0][key],axis=0)[:,li]
                    right=np.mean(curves[1][key],axis=0)[:,ri]*sign
                    rr[key+'_phase_rmse_deg']=float(np.sqrt(np.mean((left-right)**2))*180/np.pi)
                    rr[key+'_phase_mean_left_deg']=float(left.mean()*180/np.pi)
                    rr[key+'_phase_mean_right_mirrored_deg']=float(right.mean()*180/np.pi)
                    for k in range(101):
                        phase_rows.append(dict(case=case['name'],pair=suffix,signal=key,phase=k/100,left_deg=float(left[k]*180/np.pi),right_mirrored_deg=float(right[k]*180/np.pi)))
            else:
                for key in ['q','target']:
                    for metric in ['phase_rmse_deg','phase_mean_left_deg','phase_mean_right_mirrored_deg']: rr[key+'_'+metric]=None
            leg_rows.append(rr)
        for j,name in enumerate(names):
            parameter_rows.append(dict(case=case['name'],joint=name,
                kp_mean=float(par['stiffness'][ids,j].mean()),kd_mean=float(par['damping'][ids,j].mean()),
                armature=float(par['armature'][ids,j].mean()),effort_limit_nm=float(par['effort_limits'][ids,j].mean()),
                velocity_limit_rad_s=float(par['velocity_limits'][ids,j].mean())))
    result=dict(method='First episode only; 3s startup excluded. Contacts >5N with <30ms gaps/bursts filtered. Phase alignment is each foot own touchdown-to-touchdown, 101 samples, 0.15-2s cycles. No hardware torque certification.',
        completed=True,all_samples_finite=all_finite,cases=summaries)
    (root/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    write_csv(root/'joint_summary.csv',joint_rows)
    write_csv(root/'phase_comparison.csv',leg_rows)
    write_csv(root/'phase_curves.csv',phase_rows)
    write_csv(root/'actuator_parameters.csv',parameter_rows)
    # Human-readable complete joint time series for neutral replica of each case (200Hz).
    for case in meta['cases']:
        i=case['env_ids'][0]; a=field('q',[i])[:,0]; b=field('target',[i])[:,0]
        columns=['time_s','before_first_fall']+[f'{j}_{s}' for s in ['q_rad','target_rad','qd_rad_s','torque_est_nm'] for j in names]
        cutoff=fall[i] if np.isfinite(fall[i]) else np.inf
        values=np.column_stack([ts,ts<cutoff,a,b,field('qd',[i])[:,0],field('torque_est',[i])[:,0]])
        with (root/(case['name']+'_joints.csv')).open('w',newline='') as f:
            writer=csv.writer(f); writer.writerow(columns); writer.writerows(values)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    main(parser.parse_args().root)
