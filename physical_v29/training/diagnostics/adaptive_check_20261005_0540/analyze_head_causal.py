"""CPU analysis of physical causal probe, including exact CAD motor speeds."""
import json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
prior=OUT.parent/'adaptive_check_20261005_0240/analyze_physical.py'
namespace={'__file__':str(prior),'__name__':'cpu_motor_audit'}
exec(compile(prior.read_text().split('\nresults={}')[0],str(prior),'exec'),namespace)
folder=OUT/'head_causal_response';z=np.load(folder/'telemetry.npz');names=list(z['joint_names'])
per=int(z['robots_per_group']);t=z['time'];mask=t>=5.
head_ids=[names.index('head_yaw'),names.index('head_pitch')]
records=[];fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
labels=['Current learned actor','Original flight actor','Neutral head / actual inputs','Neutral head / reference head inputs']
for i,g in enumerate(z['groups']):
    ids=slice(i*per,(i+1)*per);q=z['q'][:,ids];root=z['root'][:,ids]
    w,x,y,v=[root[:,:,j] for j in [3,4,5,6]];yaw=np.arctan2(2*(w*v+x*y),1-2*(y*y+v*v))
    head=q[:,:,head_ids]
    target=z['target'][mask,ids][:,:,head_ids]
    flight=(z['clearance'][:,ids]>.001).all(-1)&(z['force'][:,ids]<1.).all(-1)
    record={'group':str(g),'head_rms_deg_after5s':float(np.rad2deg(np.sqrt((head[mask]**2).sum(-1).mean()))),
        'head_target_max_abs_after5s_rad':float(np.abs(target).max()),
        'head_raw_action_max_abs_after5s':float(np.abs(z['raw_action'][mask,ids][:,:,head_ids]).max()),
        'flight_fraction_after5s':float(flight[mask].mean()),
        'signed_forward_speed_after5s':float(root[mask,:,7].mean()),
        'abs_lateral_speed_after5s':float(np.abs(root[mask,:,8]).mean()),
        'final_abs_yaw_deg':float(np.abs(np.rad2deg(yaw[-1])).mean()),
        'survival':float(z['alive'][-1,ids].mean())}
    if i>=2:assert record['head_raw_action_max_abs_after5s']==0 and record['head_target_max_abs_after5s_rad']==0
    records.append(record)
    axes[0,0].plot(t,np.rad2deg(np.sqrt((head**2).sum(-1).mean(1))),label=labels[i])
    axes[0,1].plot(t,np.rad2deg(yaw).mean(1),label=labels[i])
    axes[1,0].plot(t,np.abs(root[:,:,8]).mean(1),label=labels[i])
    smoothed=np.convolve(flight.mean(1),np.ones(25)/25,mode='same')*100
    axes[1,1].plot(t,smoothed,label=labels[i])
for ax,title,unit in zip(axes.flat,['Head orientation','Body heading','Lateral travel','Actual flight'],['RMS degrees','Degrees','m/s','Population % (0.5s smoothing)']):
    ax.set_title(title);ax.set_ylabel(unit);ax.grid(alpha=.25);ax.axvline(3,color='black',ls='--',lw=.7)
    ax.set_xlabel('Simulated seconds')
axes[0,0].legend(fontsize=8,loc='upper right')
fig.suptitle('Diagnostic only: identical native-zero head commands, different head feedback histories',fontsize=12)
fig.tight_layout();fig.savefig(OUT/'head_causal_comparison.png',dpi=150);plt.close(fig)
result={'completed':True,'post_transient_behavior':records,'motor_audit':namespace['motor_stats'](folder),
    'diagnostic_only':True,'production_observations_changed':False,
    'interpretation':'With native-zero head action in both groups, original head-history feedback largely restores direction and flight. This reproduces learned feedback dependence; it does not prove mechanical effects absent or authorize deployment of substituted observations.'}
(OUT/'head_causal_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(records,indent=2))
