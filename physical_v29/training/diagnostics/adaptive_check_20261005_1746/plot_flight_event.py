"""Saved-physics event plot; no simulator import or training mutation."""
import json
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;d=np.load(OUT/'trajectory_validation/round_00.npz')
time=d['time'];n=int(d['robots_per_candidate']);ids=slice(2*n,3*n)
flight=(d['clearance'][:,ids]>.001).all(-1)&(d['force'][:,ids]<1).all(-1)&d['alive'][:,ids]&(time[:,None]>=5)
where=np.argwhere(flight);assert len(where)
tick,local=where[0];robot=2*n+local;sel=abs(time-time[tick])<=.6;t=time[sel]-time[tick]
phase=d['reward_phase'][sel,robot];expected=np.remainder(phase,.5)>=.43
fig,axes=plt.subplots(3,1,figsize=(8,7),sharex=True)
axes[0].plot(t,d['force'][sel,robot,0],label='Left');axes[0].plot(t,d['force'][sel,robot,1],label='Right');axes[0].set_ylabel('Normal foot force (N)');axes[0].legend()
axes[1].plot(t,d['clearance'][sel,robot,0]*1000);axes[1].plot(t,d['clearance'][sel,robot,1]*1000);axes[1].axhline(1,color='gray',ls='--');axes[1].set_ylabel('Minimum sole clearance (mm)')
axes[2].plot(t,np.remainder(d['phase'][sel,robot],.5),label='Pre-action phase');axes[2].plot(t,np.remainder(phase,.5),label='Reward phase');axes[2].axhline(.43,color='gray',ls='--');axes[2].set_ylabel('Half-cycle phase');axes[2].set_xlabel('Time relative to first flight sample (s)');axes[2].legend(loc='upper left')
for ax in axes:
 ax.axvline(0,color='red',ls=':',label='First real flight sample')
 for x,yes in zip(t,expected):
  if yes:ax.axvspan(x-.01,x+.01,color='gold',alpha=.25)
 ax.grid(alpha=.2)
fig.suptitle('Independent best-flight validation: shaded intended double-swing windows')
fig.tight_layout();fig.savefig(OUT/'flight_timing_event.png',dpi=160);plt.close(fig)
half=np.remainder(d['reward_phase'][:,ids],.5)[flight]
record=dict(first_flight_time_s=float(time[tick]),robot=int(robot),samples=int(flight.sum()),
 flight_reward_halfcycle_min=float(half.min()),median=float(np.median(half)),max=float(half.max()),
 desired_double_swing_range=[.43,.5],normal_limits=True,scope='50Hz saved physical samples, not substep duration certification')
(OUT/'flight_event.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
