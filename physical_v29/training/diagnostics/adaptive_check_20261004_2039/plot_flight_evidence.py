"""Measured forces and true sole clearance, not synthetic motion or edited images."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
d=np.load(OUT/'physical_response/telemetry.npz');t=d['time'];window=(t>=3.)&(t<=5.)
phase=np.remainder(np.arange(1,len(t)+1,dtype=np.float32)*np.float32(.02)/np.float32(1/1.65),np.float32(1))
sw=(np.remainder(phase[:,None]+np.array([0,.5],dtype=np.float32),1)>=.43)
fig,axes=plt.subplots(3,2,figsize=(12,7),sharex=True,constrained_layout=True)
for col,robot,label in [(0,0,'Final R33 actor: grounded'),(1,48,'Preserved R33 checkpoint021: some genuine flight')]:
    force=np.maximum(d['force'][:,robot],0);cl=d['clearance'][:,robot]*1000
    old=(np.exp(-force/(338.0853795690889*.25))*sw).sum(1)/np.maximum(sw.sum(1),1)
    new=np.exp(-(force*sw).sum(1)/(338.0853795690889*.25))*sw.any(1)
    axes[0,col].set_title(label,fontsize=11)
    axes[0,col].plot(t[window],force[window,0],label='Left foot');axes[0,col].plot(t[window],force[window,1],label='Right foot')
    axes[1,col].plot(t[window],cl[window,0],label='Left sole');axes[1,col].plot(t[window],cl[window,1],label='Right sole')
    axes[1,col].axhline(1,color='black',ls=':',lw=1,label='1 mm flight threshold')
    axes[2,col].plot(t[window],old[window],label='Old mean of foot scores')
    axes[2,col].plot(t[window],new[window],label='Corrected simultaneous-load score')
    axes[2,col].set_ylim(0,1.05);axes[2,col].set_xlabel('Simulation time (s)')
    for row in range(3):
        ax=axes[row,col];ax.grid(alpha=.2)
        ax.fill_between(t[window],0,1,where=sw[window].all(1),transform=ax.get_xaxis_transform(),color='gold',alpha=.17)
        ax.legend(fontsize=7,loc='upper right')
axes[0,0].set_ylabel('Normal foot force (N)');axes[1,0].set_ylabel('Minimum sole clearance (mm)');axes[2,0].set_ylabel('Unloading quality (0-1)')
fig.suptitle('Measured flight-window failure: one unloaded foot can hide the other loaded foot\nYellow: both feet are intended to be airborne; 12-robot groups are diagnostic, not qualification',fontsize=12)
fig.savefig(OUT/'flight_force_evidence.png',dpi=150);plt.close(fig)
print(OUT/'flight_force_evidence.png')
