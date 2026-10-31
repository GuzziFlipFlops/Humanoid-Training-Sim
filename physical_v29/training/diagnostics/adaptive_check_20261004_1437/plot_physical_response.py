"""Saved 50 Hz physics telemetry; no renderer or new simulator."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
data=np.load(OUT/'physical_response/telemetry.npz')
fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,layout='constrained')
for column,name in enumerate(['baseline','retract_030']):
    group=list(data['groups']).index(name);start=group*int(data['robots_per_group'])
    # Select the maximum-clearance trial in each group, declare this selection.
    best=int(np.argmax(np.min(data['clearance'][:,start:start+12],axis=2).max(axis=0)))+start
    for leg in range(2):
        axes[0,column].plot(data['time'],data['force'][:,best,leg],label=['Left','Right'][leg],lw=1)
    axes[1,column].plot(data['time'],np.min(data['clearance'][:,best],axis=1)*1000,label='Minimum across both soles',lw=1,color='purple')
    axes[0,column].set_title(name+' — selected maximum-clearance trial')
    axes[0,column].set_ylabel('Foot normal force (N)')
    axes[1,column].set_ylabel('Minimum clearance across both soles (mm)')
    axes[1,column].set_ylim(-.1,1.6)
    axes[1,column].axhline(1,color='black',ls='--',lw=.7,label='1 mm flight threshold')
    flight=(data['clearance'][:,best]>.001).all(1)&(np.abs(data['force'][:,best])<1).all(1)
    for ax in axes[:,column]:
        ax.axvline(3,color='gray',ls=':',lw=1)
        for t in data['time'][flight]:ax.axvspan(t-.01,t+.01,color='green',alpha=.3)
        ax.set_xlim(0,10);ax.grid(alpha=.2)
    axes[1,column].set_xlabel('Time (s); intervention begins at 3 s')
    axes[0,column].legend();axes[1,column].legend(fontsize=8)
fig.suptitle('Normal-limit take-off intervention: brief small flight, not running qualification\nGreen: both soles >1 mm and both normal forces <1 N; population summary in result.json')
fig.savefig(OUT/'physical_takeoff_timing.png',dpi=160)
plt.close(fig)
print(OUT/'physical_takeoff_timing.png')
