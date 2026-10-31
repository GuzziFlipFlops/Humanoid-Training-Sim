"""Separate steady command error from normal recorded stride variation."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;F=OUT/'motor_retention_response'
r=json.loads((F/'result.json').read_text())
with np.load(F/'telemetry.npz') as z:
    root=z['root'];commands=z['action_command'];alive=z['alive'];time=z['time'];reward=z['reward_rate'];per=int(z['robots_per_group'])
window=time>=5
q=root[:,:,3:7];yaw=np.arctan2(2*(q[:,:,0]*q[:,:,3]+q[:,:,1]*q[:,:,2]),1-2*(q[:,:,2]**2+q[:,:,3]**2))
v=np.stack([np.cos(yaw)*root[:,:,7]+np.sin(yaw)*root[:,:,8],-np.sin(yaw)*root[:,:,7]+np.cos(yaw)*root[:,:,8]],-1)
rows=[]
for idx,g in enumerate(r['groups']):
    ids=slice(idx*per,(idx+1)*per);mask=alive[window,ids]
    error=(v[window,ids]-commands[window,ids,:2])[mask];mean=error.mean(0);variance=error.var(0)
    mse=float((error**2).sum(-1).mean());assert abs(mse-float((mean**2+variance).sum()))<1e-5
    width=max(.10,.12*g['command']) if .1<g['command']<=1 else .5
    rows.append(dict(name=g['name'],actor=g['actor'],command=g['command'],mean_error_xy_m_s=mean.tolist(),
        root_mean_square_error_m_s=float(np.sqrt(mse)),error_bias_squared_m2_s2=float((mean**2).sum()),
        within_sample_velocity_variance_m2_s2=float(variance.sum()),
        temporal_and_population_variation_fraction=float(variance.sum()/mse) if mse else 0,
        original_tracking_mean=float(np.exp(-(error**2).sum(-1)/.25).mean()),
        candidate_tracking_mean=float(np.exp(-(error**2).sum(-1)/width**2).mean()),
        mean_error_only_candidate_score=float(np.exp(-(mean**2).sum()/width**2)),
        original_complete_reward_rate=float(reward[window,ids][mask].mean()),
        sample_scope='All32 living first-episode robots,5-20s. Variation includes temporal stride and between-robot variation; it is not pure sensor noise.'))
result=dict(completed=True,checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],rows=rows,
    moment_decomposition_checked=True,no_new_reward_deployed=True,
    scope='CPU counterfactual on recorded truthful states. Mean-error-only score is diagnostic and is not an implemented temporal reward or a convergence guarantee. A narrower instantaneous width can penalize normal stride variation and change the ranking of qualified motion.')
(OUT/'tracking_variation_analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
fig,axs=plt.subplots(1,2,figsize=(11,4.6))
for actor in ['current','qualified_prior']:
    selected=[x for x in rows if x['actor']==actor and x['name'].startswith('seed4491') and x['command']<=1]
    commands=[x['command'] for x in selected]
    axs[0].plot(commands,[x['root_mean_square_error_m_s'] for x in selected],'o-',label=actor+' instantaneous RMS')
    axs[0].plot(commands,[float(np.sqrt(x['error_bias_squared_m2_s2'])) for x in selected],'o--',label=actor+' mean error')
    axs[1].plot(commands,[x['candidate_tracking_mean'] for x in selected],'o-',label=actor+' narrow score')
    axs[1].plot(commands,[x['original_tracking_mean'] for x in selected],'o--',label=actor+' original score')
for ax in axs:
    ax.set_xlabel('Command (m/s)');ax.grid(alpha=.25);ax.legend(fontsize=8)
axs[0].set_ylabel('Yaw-frame error (m/s)');axs[1].set_ylabel('Mean exponential tracking score')
fig.suptitle('Recorded slow-command tracking: bias and stride/population variation')
fig.tight_layout();fig.savefig(OUT/'tracking_variation.png',dpi=160);fig.savefig(OUT/'tracking_variation.svg');plt.close(fig)
print(json.dumps([x for x in rows if x['actor']=='qualified_prior' and x['command']<=1],indent=2))
