"""Standalone CPU charts of saved trials; no simulated/edited image pixels."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent
e=json.loads((OUT/'evidence.json').read_text())
a=json.loads((OUT/'running_audit.json').read_text())
p=Path(a['trace_audit']['folder'])
d=np.load(p/'gait_diagnostics_50hz.npz',allow_pickle=False)
robot=int(np.flatnonzero(d['commands']==1.5)[0])
mask=(d['time']>=8)&(d['time']<=9.25)
t=d['time'][mask]
phase=np.remainder(t*1.65,1.)
flight=(phase>=.43)&(phase<.5)|(phase>=.93)
fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True,layout='constrained')
for k,color in [(0,'#1f77b4'),(1,'#d45d00')]:
    axes[0].plot(t,d['foot_force'][mask,robot,k],color=color,label=['Left foot','Right foot'][k])
    axes[1].plot(t,d['clearance'][mask,robot,k]*1000,color=color)
names=list(d['joint_names'])
for name,color in [('left_knee_pitch','#1f77b4'),('right_knee_pitch','#d45d00')]:
    idx=names.index(name)
    axes[2].plot(t,np.rad2deg(d['q'][mask,robot,idx]),color=color,label=name.replace('_',' '))
for ax in axes:
    ax.grid(alpha=.2)
    for tt,ff in zip(t,flight):
        if ff:ax.axvspan(tt-.01,tt+.01,color='#8ab28a',alpha=.13,linewidth=0)
axes[0].axhline(34.46334144435157*9.81*.05,color='#555',ls='--',lw=.8,label='Loaded-contact threshold')
axes[0].legend(loc='upper right',fontsize=8,ncol=3)
axes[2].legend(loc='upper right',fontsize=8)
axes[0].set_ylabel('Vertical foot force (N)')
axes[1].set_ylabel('Sole clearance (mm)')
axes[2].set_ylabel('Knee position (degrees)')
axes[2].set_xlabel('Independent rollout time (s)')
fig.suptitle('1.5 m/s command: alternating steps, no flight\nGreen windows = intended flight; one foot remains loaded')
fig.savefig(OUT/'running_contact_timing.png',dpi=150)
plt.close(fig)

nom=[g for g in e['gates'] if not g['result']['perturb']]
fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
for speed,color in [(.15,'#8e4d9c'),(.35,'#1f77b4'),(.65,'#df8500'),(1.,'#157f4c'),(1.5,'#c43343')]:
    rows=[(i,c['statistics']['reset_frame_forward_speed_m_s']) for i,g in enumerate(nom,1) for c in g['result']['cases'] if c['statistics']['command_m_s']==speed]
    ax.plot([r[0] for r in rows],[r[1] for r in rows],marker='o',ms=3,color=color,label=f'{speed:g} m/s command')
    ax.axhline(speed,color=color,alpha=.25,ls='--',lw=.8)
ax.axvline(next(i for i,g in enumerate(nom,1) if 'run_150' in g['folder'])-.5,color='#555',ls=':',lw=1)
ax.set(xlabel='R31 nominal gate sequence',ylabel='Signed forward speed (m/s)',
       title='Walking advances; running still has retention and flight failures')
ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=8)
fig.savefig(OUT/'speed_tracking_trend.png',dpi=150)
plt.close(fig)
print('Saved running_contact_timing.png and speed_tracking_trend.png from recorded physics.')
