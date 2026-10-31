import json,numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;folder=OUT/'heading_response'
r=json.loads((folder/'result.json').read_text());d=np.load(folder/'telemetry.npz');per=int(d['robots_per_group'])
q=d['root'][:,:,3:7];w,x,y,z=np.moveaxis(q,-1,0)
yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)),axis=0)
heading=d['reference_heading'];dx=d['root'][:,:,7]*np.cos(heading)[None,:]+d['root'][:,:,8]*np.sin(heading)[None,:]
dy=-d['root'][:,:,7]*np.sin(heading)[None,:]+d['root'][:,:,8]*np.cos(heading)[None,:]
window=d['time']>=5.;rows=[];fig,axes=plt.subplots(4,2,figsize=(13,11),sharex=True)
for i,g in enumerate(r['groups']):
 ids=slice(per*i,per*(i+1));column=0 if g['command']==1. else 1
 for a,arr in zip(axes[:,column],[dx,dy,np.rad2deg(yaw-heading[None,:]),d['action_command'][:,:,2]]):
  a.plot(d['time'],arr[:,ids].mean(1),label=g['name'].split('_',1)[1],lw=1)
 row={k:g[k] for k in ['name','command','heading_feedback_gain','course_feedback_gain','requested_yaw_rate','mean_lateral_body_velocity','mean_yaw_rate','mean_absolute_world_lateral_velocity','mean_forward_world_speed','survival','resets','head_RMS_deg','unloaded_geometric_flight_fraction','mean_reward_rate','reward_terms']}
 row.update(final_heading_mean_deg=float(np.rad2deg(yaw[-1,ids]-heading[ids]).mean()),
  mean_signed_fixed_lateral_speed=float(dy[window,ids].mean()),mean_abs_fixed_lateral_speed=float(abs(dy[window,ids]).mean()),
  mean_body_lateral_signed_speed=g['mean_lateral_body_velocity'],
  applied_yaw_command_mean=float(d['action_command'][window,ids,2].mean()),
  yaw_command_at_cap_fraction=float((abs(d['action_command'][window,ids,2])>=.19999).mean()))
 rows.append(row)
for g in rows:
 control=next(x for x in rows if x['command']==g['command'] and x['heading_feedback_gain']==.5 and x['course_feedback_gain']==0 and x['requested_yaw_rate'] is None)
 g['return_rate_difference_from_control']=g['mean_reward_rate']-control['mean_reward_rate']
 g['lateral_reduction_fraction']=1-g['mean_abs_fixed_lateral_speed']/control['mean_abs_fixed_lateral_speed']
 g['term_differences_from_control']={k:g['reward_terms'][k]-control['reward_terms'][k] for k in g['reward_terms']}
for j,speed in enumerate([1.,3.5]):
 axes[0,j].set_title(f'{speed}m/s truthful heading-command interventions');axes[3,j].set_xlabel('Simulation seconds')
for a,label in zip(axes[:,0],['Fixed forward speed (m/s)','Fixed lateral speed (m/s)','Reset heading error (degrees)','Desired yaw rate (rad/s)']):a.set_ylabel(label)
for a in axes.flat:a.grid(alpha=.2);a.legend(fontsize=6,ncol=2)
fig.tight_layout();fig.savefig(OUT/'heading_response.png',dpi=150);plt.close(fig)
motor=json.loads((folder/'substep_motor_audit.json').read_text());clips=[{'name':g['name'],'maximum_driving_clip_fraction':max(m['speed_clipped_fraction'] for m in g['motors']),'longest_driving_clip_seconds':max(m['maximum_contiguous_speed_clip_seconds'] for m in g['motors'])} for g in motor['commands']]
result={'completed':True,'checkpoint':r['checkpoint'],'checkpoint_sha256':r['checkpoint_sha256'],'rows':rows,'native_5ms_motor_summary':clips,
 'all_survive_no_resets':all(g['survival']==1 and g['resets']==0 for g in rows),
 'observed_action_command_mismatch':r['maximum_command_difference'],'reward_term_sum_error':r['maximum_reward_term_sum_error'],
 'candidate_deployed':False,'qualification':False,'scope':'384robot20s physical command-response diagnosis.5s warmup,32 clones/cohort,instantaneous velocity averages.Independent gate finite-difference metrics and complete prior qualification remain separate.'}
(OUT/'heading_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'rows':[{k:g[k] for k in ['name','mean_forward_world_speed','mean_abs_fixed_lateral_speed','mean_body_lateral_signed_speed','final_heading_mean_deg','mean_reward_rate','return_rate_difference_from_control','unloaded_geometric_flight_fraction','survival','head_RMS_deg']} for g in rows],'native_clips':clips,'command_mismatch':result['observed_action_command_mismatch'],'reward_sum_error':result['reward_term_sum_error']},indent=2))
