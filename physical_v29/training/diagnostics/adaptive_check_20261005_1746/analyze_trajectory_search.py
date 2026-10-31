"""CPU-only objective/timing consolidation; diagnostics never qualify a policy."""
import json,hashlib
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
r=read(OUT/'trajectory_search/result.json'); v=read(OUT/'trajectory_validation/result.json')
assert r['completed'] and v['completed'] and r['checkpoint_sha256']==v['checkpoint_sha256']
rows=[]; round_summary=[]
for group in r['rounds']:
 b=group['rows'][0]
 for a in group['rows']:rows.append(dict(a,generation=group['generation'],return_delta=a['mean_complete_return']-b['mean_complete_return']))
 round_summary.append(dict(generation=group['generation'],baseline_return=b['mean_complete_return'],
  best_complete_return=max(group['rows'],key=lambda x:x['mean_complete_return']),
  best_flight=max(group['rows'],key=lambda x:x['geometric_flight_fraction']),
  all_survive=all(x['survival']==1 for x in group['rows']),command_difference=group['max_command_difference']))
data=np.load(OUT/'trajectory_validation/round_00.npz'); time=data['time']; window=time>=5.; per=int(data['robots_per_candidate'])
names=list(data['joint_names']);leg=[names.index(s+'_'+j) for s in ['left','right'] for j in ['hip_pitch','knee_pitch','ankle_pitch']]
base=v['rounds'][0]['rows'][0]; validation=[]
for i,row in enumerate(v['rounds'][0]['rows']):
 ids=slice(i*per,(i+1)*per);good=data['alive'][window,ids]
 flight=(data['clearance'][window,ids]>.001).all(-1)&(data['force'][window,ids]<1).all(-1)&good
 phase=data['reward_phase'][window,ids]; half=np.remainder(phase,.5);expected=(half>=.43)
 delta={k:val-base['reward_terms'][k] for k,val in row['reward_terms'].items()}
 duration=np.zeros(per,dtype=int);longest=duration.copy()
 for tick in flight:duration=np.where(tick,duration+1,0);longest=np.maximum(longest,duration)
 terms=sum(data[k] for k in data.files if k.startswith('term_'))
 row=dict(row,label=['unchanged_control','best_complete_return','best_geometric_flight'][i],
  return_delta=row['mean_complete_return']-base['mean_complete_return'],reward_term_delta=delta,
  full_term_sum_max_error=float(abs(terms[:,ids]*.02-data['reward'][:,ids]).max()),
  true_flight_in_desired_double_swing_fraction=float((flight&expected).sum()/flight.sum()) if flight.any() else None,
  desired_double_swing_loaded_fraction=float(((data['force'][window,ids].sum(-1)>1)&expected&good).sum()/(expected&good).sum()),
  maximum_contiguous_true_flight_seconds=float(longest.max()*.02),
  mean_actual_min_sole_clearance_during_true_flight_m=float(data['clearance'][window,ids].min(-1)[flight].mean()) if flight.any() else None,
  pre_action_reward_phase_offset_cycles=float(np.mean(np.remainder(data['reward_phase'][:,ids]-data['phase'][:,ids],1))),
  modified_leg_target_delta_from_control_RMS_rad=float(np.sqrt(np.mean((data['target'][window,ids][:,:,leg]-data['target'][window,:per][:,:,leg])**2))))
 validation.append(row)
assert all(x['full_term_sum_max_error']<2e-6 for x in validation)
assert all(x['command_difference']==0 for x in round_summary)
passing=[x for x in validation[1:] if x['survival']==1 and x['geometric_flight_fraction']>=.015 and x['return_delta']>=0 and x['absolute_lateral_m_s']<=.2 and x['heading_RMS_deg']<=10 and x['head_RMS_deg']<=10]
analysis=dict(completed=True,checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],
 evaluated_trajectories=len(rows),robots_per_search_trajectory=8,search_duration_s=12,rounds=round_summary,
 independent_validation_seed=83,independent_robots_per_trajectory=32,validation_duration_s=20,validation=validation,
 all_search_robots_survived=all(a['survival']==1 for a in rows),
 maximum_search_flight_fraction=max(a['geometric_flight_fraction'] for a in rows),
 simultaneous_search_flight_and_return_candidates=[dict(generation=a['generation'],candidate=a['candidate'],flight=a['geometric_flight_fraction'],return_delta=a['return_delta']) for a in rows if a['geometric_flight_fraction']>=.015 and a['return_delta']>=0],
 deploy_supported=bool(passing),production_changes=False,
 limitation='Bounded ten-parameter local search is not proof that physical running is impossible, all running has lower reward, or a hardware ceiling. Validation tests diagnostic offsets, not the unmodified policy or whole-chain independent qualification. Different robots within a population are not paired-state counterfactuals.')
(OUT/'trajectory_analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(8,4.5));scatter=ax.scatter([a['geometric_flight_fraction']*100 for a in rows],[a['return_delta'] for a in rows],c=[a['generation'] for a in rows],cmap='viridis',s=24)
ax.axvline(1.5,color='red',ls='--',label='Unchanged flight minimum (diagnostic)');ax.axhline(0,color='gray',lw=1)
ax.set(xlabel='Actual unloaded geometric flight (% after 5s)',ylabel='Complete 12s return difference from round control',title='Normal-limit leg timing search: flight versus full return')
fig.colorbar(scatter,ax=ax,label='Search generation');ax.legend(loc='lower left');fig.tight_layout();fig.savefig(OUT/'trajectory_return_flight.png',dpi=160);plt.close(fig)
print(json.dumps(dict(trajectories=len(rows),maximum_flight=analysis['maximum_search_flight_fraction'],deploy_supported=bool(passing),
 validation=[{k:a[k] for k in ['label','return_delta','geometric_flight_fraction','signed_forward_m_s','head_RMS_deg','maximum_contiguous_true_flight_seconds','true_flight_in_desired_double_swing_fraction']} for a in validation]),indent=2))
