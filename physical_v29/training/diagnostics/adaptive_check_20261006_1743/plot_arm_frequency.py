import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent;r=json.loads((out/'arm_response_analysis.json').read_text());z=np.load(out/'arm_response/telemetry.npz');names=z['joint_names'].tolist();per=int(z['robots_per_group'])
rows=[a for a in r['rows'] if a['diagnostic_sine_frequency_Hz'] is not None]
fig,axes=plt.subplots(4,2,figsize=(14,12))
for i,a in enumerate(rows):
 robot=(6+i)*per;mask=(z['time']>=3)&(z['time']<=8)
 for color,side in [('blue','left'),('red','right')]:
  j=names.index(side+'_shoulder_pitch')
  axes[i,0].plot(z['time'][mask],np.rad2deg(z['target'][mask,robot,j]),color=color,alpha=.35,ls='--')
  axes[i,0].plot(z['time'][mask],np.rad2deg(z['q'][mask,robot,j]),color=color,label=side)
 axes[i,0].set_title(f"Stopped, {a['diagnostic_sine_frequency_Hz']}Hz diagnostic offset")
 axes[i,0].legend();axes[i,0].grid(alpha=.2);axes[i,0].set_ylabel('Shoulder pitch degrees')
 axes[i,1].plot(z['time'][mask],z['root'][mask,robot,8]);axes[i,1].set_title('Actual base lateral velocity');axes[i,1].grid(alpha=.2);axes[i,1].set_ylabel('m/s')
fig.suptitle('Closed-loop free-base shoulder response, native motor limits\nSolid actual joints, dashed physical targets. Diagnostic target offsets only; no changes deployed.')
fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(out/'arm_frequency_response.png',dpi=160);plt.close(fig)
print('Saved full actual/target timelines and physical coupling plots.')
