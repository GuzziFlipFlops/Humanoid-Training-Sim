import json,numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;folder=OUT/'steering_response';r=json.loads((folder/'result.json').read_text());d=np.load(folder/'telemetry.npz');dt=.02
q=d['root'][:,:,3:7];w,x,y,z=np.moveaxis(q,-1,0)
yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)),axis=0)
heading=d['reference_heading'];dx=d['root'][:,:,7]*np.cos(heading)[None,:]+d['root'][:,:,8]*np.sin(heading)[None,:]
dy=-d['root'][:,:,7]*np.sin(heading)[None,:]+d['root'][:,:,8]*np.cos(heading)[None,:]
window=d['time']>=5.;rows=[]
fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
for i,g in enumerate(r['groups']):
 ids=slice(32*i,32*(i+1));column=0 if g['command']==1. else 1
 for a,arr in zip(axes[:,column],[dx,dy,np.rad2deg(yaw-heading[None,:])]):a.plot(d['time'],arr[:,ids].mean(1),label=g['name'].split('_',1)[1])
 rows.append({k:g[k] for k in ['name','mean_lateral_body_velocity','mean_yaw_rate','mean_absolute_world_lateral_velocity','mean_forward_world_speed','survival','head_RMS_deg','unloaded_geometric_flight_fraction','mean_reward_rate']})
 rows[-1].update(final_heading_mean_deg=float(np.rad2deg(yaw[-1,ids]-heading[ids]).mean()),mean_signed_fixed_lateral_speed=float(dy[window,ids].mean()))
for j,speed in enumerate([1.,3.5]):
 axes[0,j].set_title(f'{speed}m/s truthful command interventions');axes[2,j].set_xlabel('Simulation seconds')
for a,label in zip(axes[:,0],['Fixed forward speed (m/s)','Fixed lateral speed (m/s)','Reset heading error (deg)']):a.set_ylabel(label)
for a in axes.flat:a.grid(alpha=.2);a.legend(fontsize=7)
fig.tight_layout();fig.savefig(OUT/'steering_response.png',dpi=140);plt.close(fig)
mot=json.loads((folder/'substep_motor_audit.json').read_text());clips=[dict(name=x['name'],driving_clip_fraction=max(v['speed_clipped_fraction'] for v in x['motors']),maximum_contiguous_seconds=max(v['maximum_contiguous_speed_clip_seconds'] for v in x['motors'])) for x in mot['commands']]
result=dict(completed=True,checkpoint=r['checkpoint'],rows=rows,substep_summary=clips,zero_resets=all(g['resets']==0 for g in r['groups']),command_mismatch=r['maximum_command_difference'],term_sum_error=r['maximum_reward_term_sum_error'],interpretation='Lateral command interventions barely change motion; constant yaw commands alter heading. This reveals untrained lateral-command response, not a bad yaw sign, impossible steering, or proven benefit of command exposure. Normal straight cohorts have no native 5ms driving speed clips. No high-speed ceiling is established.')
(OUT/'steering_analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
