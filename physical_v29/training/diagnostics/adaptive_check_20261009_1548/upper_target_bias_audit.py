"""Separate actual shoulder bias from physical target bias; no training edits."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--gate-review',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(exist_ok=True)
pair=json.loads(a.gate_review.read_text())['latest'];records=[];plots=[]
for label in ['nominal','pushed']:
    f=Path(pair[label])/'gait_diagnostics_50hz.npz'
    with np.load(f) as z:d={k:z[k].copy() for k in z.files}
    names=d['joint_names'].tolist();dt=float(d['step_dt']);time=d['time'];sos=butter(4,.5,fs=1/dt,output='sos')
    for speed in np.unique(d['commands']):
        robots=np.flatnonzero(np.isclose(d['commands'],speed));end=len(time)
        terminal=np.flatnonzero(d['done'][:,robots].any(1))
        if len(terminal):end=int(terminal[0])
        if end*dt<8:continue
        actual=np.rad2deg(d['q'][:end,robots]);target=np.rad2deg(d['target'][:end,robots]);
        slow=sosfiltfilt(sos,actual,axis=0);wanted=sosfiltfilt(sos,target,axis=0)
        window=(time[:end]>=5)&(time[:end]<=time[end-1]-3);row={}
        for name in ['head_yaw','head_pitch','left_shoulder_pitch','right_shoulder_pitch','left_shoulder_roll','right_shoulder_roll','left_elbow_pitch','right_elbow_pitch']:
            j=names.index(name)
            row[name]=dict(actual_slow_mean_deg=float(slow[window,:,j].mean()),target_slow_mean_deg=float(wanted[window,:,j].mean()),
                actual_minus_target_slow_RMS_deg=float(np.sqrt(np.mean((slow[window,:,j]-wanted[window,:,j])**2))),
                actual_faster_residual_RMS_deg=float(np.sqrt(np.mean((actual[window,:,j]-slow[window,:,j])**2))),
                target_faster_residual_RMS_deg=float(np.sqrt(np.mean((target[window,:,j]-wanted[window,:,j])**2))),
                raw_action_outside_clamp_fraction=float((abs(d['raw_action'][:end,robots][:,:,j][window])>3).mean()))
        records.append(dict(test=label,speed=float(speed),representative_robots=len(robots),first_episode_seconds=(end-1)*dt,joints=row))
        if label=='nominal' and speed in [1.,2.5,3.,3.5]:plots.append((speed,time[:end],slow[:,0],wanted[:,0],names))
fig,axes=plt.subplots(2,len(plots),figsize=(16,7),squeeze=False,sharex=True)
for c,(speed,time,actual,target,names) in enumerate(plots):
    for r,side in enumerate(['left','right']):
        j=names.index(side+'_shoulder_pitch');axes[r,c].plot(time,actual[:,j],label='actual slow component')
        axes[r,c].plot(time,target[:,j],ls='--',label='physical target slow component')
        axes[r,c].set_title(f'{speed:g}m/s {side} shoulder');axes[r,c].grid(alpha=.2);axes[r,c].legend(fontsize=7)
        axes[r,c].set_ylabel('Native degrees');axes[r,c].set_xlabel('Simulation time(s)')
fig.suptitle('Shoulder held bias: measured joint versus commanded physical target\n50Hz representatives,0.5Hz descriptive filter;3s edges excluded from statistics')
fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(a.output_dir/'upper_target_bias.png',dpi=150);plt.close(fig)
result=dict(checkpoint=pair['checkpoint'],sha256=pair['sha256'],records=records,
    scope='Actual physical targets, not raw policy means or PPO derivatives. Native signs/axes; no reference human/Unitree curves. Descriptive0.5Hz filter; each first episode separate,3s edges excluded. Representative samples do not equal population statistics.',
    interpretation='Persistent target bias and joint bias may identify a commanded posture component; agreement alone does not establish why PPO chooses it or that eliminating it retains gait.',
    original_upper_refinement_remains='Eight upper final-output rows only;comfortable bands and pointwise opposite-pitch symmetry,not a trajectory or cycle-mean objective')
(a.output_dir/'upper_target_bias_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(records,indent=2))
