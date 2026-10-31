"""Standalone scientific illustration of declared kernels, not training success."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
x=np.linspace(-.4,.4,1001)
fig,axs=plt.subplots(1,2,figsize=(11,4))
for ax,speed in zip(axs,[.35,3.5]):
    tol=max(.1,.12*speed);width=tol if speed<=1. else .5
    ax.plot(x,4*np.exp(-x*x/.25),label='Current tracking rate, width0.5m/s')
    ax.plot(x,4*np.exp(-x*x/width**2),ls='--',label='Diagnostic tolerance-scaled rate')
    ax.axvspan(-tol,tol,color='grey',alpha=.12,label='Existing speed tolerance')
    ax.set(title=f'{speed:g}m/s command',xlabel='Instantaneous forward velocity error(m/s)',ylabel='Tracking reward rate',ylim=(0,4.1))
    ax.grid(alpha=.3)
axs[0].legend(fontsize=8)
fig.suptitle('Only slow-command Gaussian precision changes; gait oscillations remain a risk\nWeights, yaw frame and all faster/stopped rewards stay unchanged')
fig.tight_layout();fig.savefig(OUT/'tracking_width_candidate.png',dpi=150);fig.savefig(OUT/'tracking_width_candidate.svg')
