"""Saved-state phase/target/contact comparison; no policy or physics changes."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
F=ROOT/'motor_retention_response'
result=json.loads((F/'result.json').read_text())
with np.load(F/'telemetry.npz') as z:
    time=z['time'];phase=z['reward_phase'];prephase=z['phase'];q=z['q'];target=z['target']
    force=z['force'];clearance=z['clearance'];alive=z['alive'];clips=z['target_clipped'];slew=z['target_slew']
    names=z['joint_names'].tolist();labels=z['groups'].tolist();per=int(z['robots_per_group'])
with np.load(F/'representative_joint_200hz.npz') as z:
    rq=z['joint_position'];rt=z['target'];dt=float(z['dt'])
window=time>=5.;bins=np.linspace(0,1,41);centers=(bins[1:]+bins[:-1])/2
rows=[];fig,axs=plt.subplots(3,2,figsize=(13,10),sharex=True)
for seed,column in [(4491,0),(4511,1)]:
    for actor,color in [('current','tab:red'),('qualified_prior','tab:blue')]:
        group=next(g for g in result['groups'] if g['name']==f'seed{seed}_{actor}_1.5')
        index=labels.index(group['name']);ids=slice(index*per,(index+1)*per)
        p=np.remainder(phase[window,ids],1.)
        good=alive[window,ids]
        assert good.all(), 'This comparison requires surviving no-reset cohorts'
        flight=(clearance[window,ids]>.001).all(-1)&(force[window,ids]<1.).all(-1)
        min_clear=clearance[window,ids].min(-1)
        foot_force=force[window,ids]
        leg_names=['left_hip_pitch','right_hip_pitch','left_knee_pitch','right_knee_pitch','left_ankle_pitch','right_ankle_pitch']
        qi=[names.index(n) for n in leg_names]
        aq=np.rad2deg(q[window,ids][:,:,qi]);at=np.rad2deg(target[window,ids][:,:,qi])
        phase_rows=[]
        for bi in range(40):
            mask=(p>=bins[bi])&(p<bins[bi+1])
            if not mask.any(): continue
            phase_rows.append(dict(phase_center=float(centers[bi]),samples=int(mask.sum()),
                genuine_flight_fraction=float(flight[mask].mean()),
                minimum_sole_clearance_mean_m=float(min_clear[mask].mean()),
                left_force_mean_N=float(foot_force[:,:,0][mask].mean()),right_force_mean_N=float(foot_force[:,:,1][mask].mean()),
                measured_leg_mean_deg={n:float(aq[:,:,j][mask].mean()) for j,n in enumerate(leg_names)},
                physical_target_mean_deg={n:float(at[:,:,j][mask].mean()) for j,n in enumerate(leg_names)}))
        common=np.array([r['phase_center'] for r in phase_rows])
        axs[0,column].plot(common,[r['left_force_mean_N']+r['right_force_mean_N'] for r in phase_rows],color=color,label=actor)
        axs[1,column].plot(common,[100*r['genuine_flight_fraction'] for r in phase_rows],color=color,label=actor)
        axs[2,column].plot(common,[1000*r['minimum_sole_clearance_mean_m'] for r in phase_rows],color=color,label=actor)
        dq=np.remainder(phase[window,ids]-prephase[window,ids]+.5,1.)-.5
        physical_tracking={n:float(np.sqrt(np.mean((aq[:,:,j]-at[:,:,j])**2))) for j,n in enumerate(leg_names)}
        # Descriptive target/joint transfer at the hip's dominant cycle. One
        # 200 Hz representative per cohort, not population causal system ID.
        sample=rq[1000:,index];target_sample=rt[1000:,index]
        freq=np.fft.rfftfreq(len(sample),dt)
        spectrum=np.abs(np.fft.rfft(sample-sample.mean(0),axis=0))**2
        hip=names.index('left_hip_pitch');band=(freq>=.5)&(freq<=4.3)
        peak=np.flatnonzero(band)[np.argmax(spectrum[band,hip])]
        transfer={}
        for n in leg_names:
            j=names.index(n);s=np.fft.rfft(sample[:,j]-sample[:,j].mean());t=np.fft.rfft(target_sample[:,j]-target_sample[:,j].mean())
            transfer[n]=dict(joint_relative_to_target_phase_deg=float(np.rad2deg(np.angle(s[peak]*np.conj(t[peak])))),
                dominant_harmonic_gain=float(abs(s[peak])/max(abs(t[peak]),1e-12)),
                joint_p95_p05_deg=float(np.rad2deg(np.quantile(sample[:,j],.95)-np.quantile(sample[:,j],.05))))
        rows.append(dict(seed=seed,actor=actor,command=1.5,robots=per,genuine_flight_fraction=float(flight.mean()),
            full_reward_rate=group['mean_reward_rate'],physical_leg_target_joint_RMS_deg=physical_tracking,
            target_joint_limit_fraction=float(clips[window,ids][:,:,qi].mean()),
            target_slew_limit_fraction=float(slew[window,ids][:,:,qi].mean()),
            pre_action_to_reward_phase_delta_median=float(np.median(dq)),
            phase_conditioned=phase_rows,representative_200Hz_hip_cycle_Hz=float(freq[peak]),
            representative_target_joint_transfer=transfer))
    for ax in axs[:,column]:ax.grid(alpha=.25);ax.legend(fontsize=8)
    axs[0,column].set_title(f'1.5 m/s command; initial-state seed {seed}')
    axs[2,column].set_xlabel('Saved reward phase (cycles)')
for ax,y in zip(axs[:,0],['Summed sole force (N)','True flight within phase bin (%)','Mean minimum sole clearance (mm)']):ax.set_ylabel(y)
fig.suptitle('Current policy versus qualified earlier policy: take-off timing\n32 robots/controller/seed, normal limits; own truthful closed-loop states')
fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(ROOT/'takeoff_phase_comparison.png',dpi=160);fig.savefig(ROOT/'takeoff_phase_comparison.svg');plt.close(fig)
output=dict(completed=True,checkpoint_sha256=result['checkpoint_sha256'],qualified_prior_sha256=result['actor_sha256']['qualified_prior'],
    telemetry_sha256=hashlib.sha256((F/'telemetry.npz').read_bytes()).hexdigest(),rows=rows,
    scope='Phase-conditioned first-episode population force/clearance and actual physical leg targets at 50 Hz after 5 seconds. The one representative/cohort 200 Hz transfer is descriptive. Controllers share paired initial states but diverge under their own truthful feedback. This comparison is not same-state teacher-label error, a mechanical isolation, a reward-clock defect, or justification for a fixed latency shift.')
(ROOT/'takeoff_phase_comparison.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps([dict(seed=r['seed'],actor=r['actor'],flight=r['genuine_flight_fraction'],tracking=r['physical_leg_target_joint_RMS_deg'],slew=r['target_slew_limit_fraction']) for r in rows],indent=2))
