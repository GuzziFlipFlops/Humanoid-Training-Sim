"""Standalone current/prior physical response plot, not gate qualification."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
r=json.loads((OUT/'motor_retention_summary.json').read_text());rows=r['records']
fig,axes=plt.subplots(1,3,figsize=(14,4.2))
colors={'current':'#1767a7','qualified_prior':'#bd6324'}
for axis,key,title,scale in zip(axes,['signed_speed','lateral_speed','flight'],['Signed fixed-frame speed (m/s)','Absolute fixed-frame lateral speed (m/s)','Unloaded geometric flight (%)'],[1,1,100]):
 for actor in colors:
  chosen=[x for x in rows if x['actor']==actor]
  commands=sorted(set(x['command'] for x in chosen))
  average=[np.mean([x[key]*scale for x in chosen if x['command']==c]) for c in commands]
  axis.plot(commands,average,'o-',color=colors[actor],label=actor.replace('_',' '))
  for x in chosen:axis.plot(x['command'],x[key]*scale,'.',color=colors[actor],alpha=.6)
 axis.set_xlabel('Command (m/s)');axis.set_ylabel(title);axis.grid(alpha=.25)
axes[0].plot([0,3.5],[0,3.5],'--',color='grey',alpha=.4,label='command line')
axes[2].axhline(1.5,color='grey',linestyle=':',label='existing running minimum')
axes[0].legend(loc='upper left');axes[2].legend(loc='upper left')
fig.suptitle('Exact current checkpoint773 versus qualified prior — truthful normal-limit physics\nTwo seeds,32 independent initial states per cohort,20s; plotted averages after5s. No full-stage certification.')
fig.tight_layout();fig.savefig(OUT/'motor_retention_response.png',dpi=160);fig.savefig(OUT/'motor_retention_response.svg');plt.close(fig)
print('Saved current/prior physical response chart.')
