import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
data=np.load(OUT/'phase_causal_response/telemetry.npz');names=list(data['joint_names']);headids=[names.index('head_yaw'),names.index('head_pitch')]
time=data['time'];window=time>=5.
fig,axes=plt.subplots(3,4,figsize=(15,8),sharex=True,constrained_layout=True)
for col,speed in enumerate([.35,.65,1.,1.5]):
    for variant,label,color in [(0,'Current R35','#2877af'),(1,'Original teacher','#ba6060'),(4,'Neutral head, phase context','#168064')]:
        ids=slice((col*5+variant)*32,(col*5+variant+1)*32)
        root=data['root'][:,ids];heads=data['q'][:,ids][:,:,headids]
        force=data['force'][:,ids];clearance=data['clearance'][:,ids]
        flight=((force<1).all(-1)&(clearance>.001).all(-1)).mean(1)
        axes[0,col].plot(time,root[:,:,7].mean(1),label=label,color=color,lw=1)
        axes[1,col].plot(time,np.rad2deg(np.sqrt((heads**2).sum(-1).mean(1))),color=color,lw=1)
        smooth=np.convolve(flight,np.ones(50)/50,'same')*100
        axes[2,col].plot(time,smooth,color=color,lw=1)
    axes[0,col].axhline(speed,color='#555555',ls='--',lw=.8)
    axes[0,col].set_title(f'Command {speed:g} m/s')
    axes[2,col].set_xlabel('Simulated seconds')
    for row in range(3):
        axes[row,col].axvspan(0,5,color='gray',alpha=.08);axes[row,col].grid(alpha=.2)
axes[0,0].set_ylabel('Signed forward m/s');axes[1,0].set_ylabel('Head RMS degrees');axes[2,0].set_ylabel('True flight %, 1s average')
axes[0,0].legend(fontsize=8)
fig.suptitle('Diagnostic normal-limit physics: forward head with teacher-only phase context\nNo input rewrites or diagnostic controls are deployed')
fig.savefig(OUT/'phase_causal_comparison.png',dpi=150);plt.close(fig)
print(str(OUT/'phase_causal_comparison.png'))
