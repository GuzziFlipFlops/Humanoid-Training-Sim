"""Consolidate physical timing and fixed-state reward evidence on CPU."""
import json,hashlib
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
r=read(OUT/'timing_probe/result.json');z=np.load(OUT/'timing_probe/round_00.npz')
assert r['completed'] and r['actor_loaded_exact'] and r['normal_limits'] and r['truthful_observations']
assert r['checkpoint_sha256']==read(OUT/'diagnostic_pause.json')['checkpoint_sha256']
assert r['rounds'][0]['max_command_difference']==0
per=int(z['robots_per_candidate']);window=z['time']>=5.;rows=[]
terms=sum(z[k] for k in z.files if k.startswith('term_'))
assert float(abs(terms*.02-z['reward']).max())<2e-6
base=r['rounds'][0]['rows'][0]
for i,row in enumerate(r['rounds'][0]['rows']):
    ids=slice(i*per,(i+1)*per)
    force=z['force'][window,ids];clear=z['clearance'][window,ids];phase=z['reward_phase'][window,ids]
    flight=(clear>.001).all(-1)&(force<1.).all(-1)&z['alive'][window,ids]
    half=np.remainder(phase,.5);intended=half>=.43
    length=np.zeros(per,dtype=int);longest=length.copy()
    for tick in flight:
        length=np.where(tick,length+1,0);longest=np.maximum(longest,length)
    rows.append(dict(row,return_delta=row['mean_complete_return']-base['mean_complete_return'],
        mean_min_sole_clearance_in_flight_m=float(clear.min(-1)[flight].mean()) if flight.any() else None,
        flight_in_desired_double_swing_fraction=float((flight&intended).sum()/flight.sum()) if flight.any() else None,
        flight_reward_halfcycle_quantiles=np.quantile(half[flight],[0,.5,1]).tolist() if flight.any() else None,
        longest_50Hz_flight_sample_run_s=float(longest.max()*.02),
        reward_term_delta={k:v-base['reward_terms'][k] for k,v in row['reward_terms'].items()}))
smooth=read(OUT/'current_smoothed_counterfactual.json')
passing=[a for a in rows[1:] if a['geometric_flight_fraction']>=.015 and a['return_delta']>=0 and a['survival']==1]
analysis=dict(completed=True,checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],
    validation=rows,all_survived=all(a['survival']==1 for a in rows),robots=192,robots_per_cohort=32,seconds=20,seed=91,
    flight_measurement_window_s=[5,20],maximum_flight_fraction=max(a['geometric_flight_fraction'] for a in rows),
    command_difference=0,full_reward_reconstruction_error=float(abs(terms*.02-z['reward']).max()),
    smoothing_reconstruction_error=smooth['reconstruction_max_abs'],deploy_supported=bool(passing),production_changes=False,
    smoothing_decision='Reject this summed-force box-smoothing candidate before learning: tested flight trajectory remains lower complete-return and edge load redistribution changes credit. No 64/128 learning test is claimed.',
    physical_timing_decision='Reject these fixed 20/40/60ms advances: all remain grounded and lose full return. Keep the current learning actor; evidence does not justify a clock correction or arbitrary weight change.',
    limitation='Fixed-cohort local offsets do not prove running impossible or a motor ceiling. Robots are not paired-state counterfactuals. Smoothed offline rewards change no dynamics. 50Hz flight duration is not substep-certified contact pressure. No diagnostic controller is deployed.')
(OUT/'trajectory_analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(10,4))
labels=['Control','Original','20ms early','40ms early','60ms early','Best return']
ax[0].bar(np.arange(6),[100*a['geometric_flight_fraction'] for a in rows],color='#3979a8')
ax[0].axhline(1.5,color='#c54747',ls='--',label='Independent flight requirement')
ax[0].set(ylabel='True geometric flight after 5s (%)',title='Earlier offsets did not induce take-off')
ax[1].bar(np.arange(6),[a['return_delta'] for a in rows],color=['#777777' if a['return_delta']>=0 else '#c77a47' for a in rows])
ax[1].axhline(0,color='gray',lw=1);ax[1].set(ylabel='Complete 20-second return difference',title='Full objective remains better for grounded motion')
for a in ax:a.set_xticks(np.arange(6),labels,rotation=25,ha='right')
ax[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'timing_tradeoff.png',dpi=160);plt.close(fig)
print(json.dumps(dict(rows=[{k:a[k] for k in ['label','return_delta','geometric_flight_fraction','flight_reward_halfcycle_quantiles']} for a in rows],deploy_supported=bool(passing)),indent=2))
