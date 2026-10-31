import json
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt
OUT=Path(__file__).resolve().parent;review=json.loads((OUT/'matched_shoulder_review.json').read_text())
rows=[]
for row in review['rows']:
    for label in ['nominal','pushed']:
        with np.load(Path(row[label])/'gait_diagnostics_50hz.npz') as z:
            q=z['q'];targets=z['target'];raw=z['raw_action'];commands=z['commands'];time=z['time'];done=z['done'];names=z['joint_names'].tolist()
        for command in sorted(set(commands.tolist())):
            ids=np.flatnonzero(commands==command);end=len(time)
            terminal=np.flatnonzero(done[:,ids].any(1))
            if len(terminal):end=int(terminal[0])
            selected=(time[:end]>=5)&(time[:end]<=17)
            shoulder=[names.index(n) for n in ['left_shoulder_pitch','right_shoulder_pitch']]
            head=[names.index(n) for n in ['head_yaw','head_pitch']]
            actual=q[:end][selected][:,ids][:,:,shoulder]*180/np.pi
            wanted=targets[:end][selected][:,ids][:,:,shoulder]*180/np.pi
            headq=q[:end][selected][:,ids][:,:,head]*180/np.pi
            sos=butter(4,.5,fs=1/float(time[1]-time[0]),output='sos')
            slow=sosfiltfilt(sos,q[:end][:,ids][:,:,shoulder]*180/np.pi,axis=0)[selected]
            wanted_slow=sosfiltfilt(sos,targets[:end][:,ids][:,:,shoulder]*180/np.pi,axis=0)[selected]
            rows.append(dict(branch=row['branch'],updates=row['updates'],test=label,command=float(command),representative_robots=len(ids),reset_ticks=int(done[:end,ids].sum()),
                mean_pitch_deg=actual.mean((0,1)).tolist(),mean_target_pitch_deg=wanted.mean((0,1)).tolist(),
                absolute_mean_bias_deg=float(np.abs(actual.mean((0,1))).mean()),
                absolute_mean_target_bias_deg=float(np.abs(wanted.mean((0,1))).mean()),
                noncancelling_slow_RMS_deg=np.sqrt(np.mean(slow**2,axis=(0,1))).tolist(),
                noncancelling_slow_target_RMS_deg=np.sqrt(np.mean(wanted_slow**2,axis=(0,1))).tolist(),
                slow_p95_p05_range_deg=np.median(np.percentile(slow,95,axis=0)-np.percentile(slow,5,axis=0),axis=0).tolist(),
                faster_residual_RMS_deg=np.sqrt(np.mean((actual-slow)**2,axis=(0,1))).tolist(),
                head_RMS_deg=float(np.sqrt((headq**2).mean())),
                raw_shoulder_clip_fraction=float((abs(raw[:end][selected][:,ids][:,:,shoulder])>3).mean())))
result=dict(rows=rows,scope='50Hz representative first episodes,5-17s;per-command averages and target means,not a calibrated human motion or full-population fast-jitter statistic.',
    mean_bias_limitation='A time/population mean can cancel slow reversals or opposite individual postures. Noncancelling per-shoulder slow RMS and slow range are also retained;0.5Hz is descriptive,not an imposed gait boundary.',
    primary_feature_reference='https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043',
    official_robot_reference='https://github.com/unitreerobotics/unitree_rl_mjlab',mapped_speed_matched_Unitree_encoder_reference_available=False,
    natural_motion_certified=False,world_optical_gaze_certified=False)
(OUT/'upper_matched_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([r for r in rows if r['command'] in [1.,2.5,3.,3.5]],indent=2))
