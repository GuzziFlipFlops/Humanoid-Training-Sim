"""Higher-speed gate and first-episode motor review from saved evidence only."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'high_speed_extrapolation';F=OUT/'physical_response'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
physical=read(F/'result.json');pause=read(OUT/'diagnostic_pause.json')
assert physical['checkpoint_sha256']==pause['checkpoint_sha256']
assert physical['actor_loaded_exact'] and physical['maximum_command_difference']==0
assert physical['maximum_reward_term_sum_error']<1e-3
assert len(physical['groups'])==8 and physical['initial_state_seeds']==[4591,4611]
assert physical['normal_motor_limits'] and not physical['observation_rewrites']
gate_rows=[]
for label in ['nominal','pushed','diverse_nominal','diverse_pushed']:
    folder=OUT/label
    assert read(folder/'result.json')['completed']
    assert read(folder/'run_config.json')['arguments']['checkpoint']==pause['checkpoint']
    ev=read(folder/'evaluation.json');assert len(ev['cases'])==13
    with np.load(folder/'telemetry.npz') as z:
        gate_alive=z['metrics'][:,:,0]>0;labels=z['case_labels']
    cases=[]
    for idx,c in enumerate(ev['cases']):
        s=c['statistics'];assert s['trials']==32
        good=gate_alive[:,labels==idx];paired_count=int((good[1:]&good[:-1]).sum())
        cases.append(dict(command=s['command_m_s'],passed=c['passed'],failures=c['failures'],statistics=s,
            paired_first_episode_live_postwarmup_samples=paired_count,
            measured_signed_forward_m_s=s['reset_frame_forward_speed_m_s'] if paired_count else None,
            measured_absolute_lateral_m_s=s['reset_frame_lateral_speed_m_s'] if paired_count else None,
            numeric_gate_failure_sentinels_are_not_achieved_speed=True))
    row=dict(label=label,passed=ev['passed'],cases=cases)
    if label.startswith('diverse'):
        diversity=read(folder/'initial_diversity.json')
        assert all(n==32 for n in diversity['unique_initial_states_by_command'].values())
        row['initial_state_diversity']=diversity
    gate_rows.append(row)

with np.load(F/'telemetry.npz') as z:
    time=z['time'];alive=z['alive'];done=z['failures'];root=z['pre_reset_root'];reason=z['termination_causes']
    reason_names=z['termination_reason_names'].tolist();groups=z['groups'].tolist();per=int(z['robots_per_group'])
    reference=z['reference_heading'];initial_q=z['initial_q'];initial_qd=z['initial_qd']
    observed=z['observed_command'];command=z['action_command']
assert np.array_equal(observed,command)
assert alive.shape==(1000,256) and per==32 and len(time)==1000
assert np.array_equal(alive,np.cumprod(~done,axis=0).astype(bool)), 'First-episode alive mask must never recover after a reset'
assert all(len({r.tobytes() for r in np.concatenate([initial_q[i*per:(i+1)*per],initial_qd[i*per:(i+1)*per]],1)})==32 for i in range(8))
with np.load(F/'substep_motor_200hz.npz') as z:
    ratio=z['cap_ratio'];velocity=z['native_velocity'];request=z['requested_torque'];applied=z['applied_torque'];saved=z['speed_clip'];motor_names=z['motor_names'].tolist()
assert ratio.shape==velocity.shape==request.shape==applied.shape==saved.shape and len(ratio)==4000
assert np.isfinite(ratio).all() and np.isfinite(velocity).all() and np.isfinite(request).all() and np.isfinite(applied).all()
valid=np.repeat(alive,4,axis=0)
independent=(ratio>=1)&(request*velocity>0)&(applied*velocity<=0)
assert np.array_equal(independent[valid],saved[valid]), 'Driving clip attribution differs on the first living episode'

limits=read(OUT/'control_source/physical_v29/assets/actuator_limits.json')
geometry=read(OUT/'control_source/physical_v29/assets/differential_geometry.json')
motor_rows=[];first_failure_rows=[]
for idx,g in enumerate(physical['groups']):
    ids=slice(idx*per,(idx+1)*per);mask=valid[1000:,ids];clips=saved[1000:,ids]&mask[:,:,None]
    all_mask=valid[:,ids];all_clips=saved[:,ids]&all_mask[:,:,None]
    ra=ratio[1000:,ids];req=request[1000:,ids];app=applied[1000:,ids];vel=velocity[1000:,ids]
    streak=np.zeros((per,len(motor_names)),dtype=np.int32);longest=streak.copy()
    for tick in clips:
        streak=np.where(tick,streak+1,0);longest=np.maximum(longest,streak)
    motors=[]
    for j,name in enumerate(motor_names):
        peak=limits[geometry[name.rsplit('_motor_',1)[0]]['joint_names'][0]]['motor_peak_Nm'] if '_motor_' in name else limits[name]['effort_Nm']
        over=ra[:,:,j]>=1;driving=req[:,:,j]*vel[:,:,j]>0
        def mean(x):return float(x[mask].mean()) if mask.any() else None
        motors.append(dict(name=name,first_episode_alive_p99_cap_ratio=(float(np.quantile(ra[:,:,j][mask],.99)) if mask.any() else None),
            alive_maximum_cap_ratio=(float(ra[:,:,j][mask].max()) if mask.any() else None),
            overspeed_fraction=mean(over),driving_clip_fraction=mean(clips[:,:,j]),
            overspeed_braking_or_zero_fraction=mean(over&~driving),
            maximum_contiguous_driving_clip_seconds=float(longest[:,j].max()*.005),
            declared_peak_torque_Nm=peak,requested_above_declared_peak_fraction=mean(abs(req[:,:,j])>peak+1e-6),
            applied_at_declared_peak_fraction=mean(abs(app[:,:,j])>=peak-1e-4)))
    full_streak=np.zeros((per,len(motor_names)),dtype=np.int32);full_longest=full_streak.copy()
    for tick in all_clips:
        full_streak=np.where(tick,full_streak+1,0);full_longest=np.maximum(full_longest,full_streak)
    all_motor=[dict(name=n,first_episode_alive_driving_clip_fraction=(float(all_clips[:,:,j].sum()/all_mask.sum()) if all_mask.any() else None),
        first_episode_alive_driving_clip_count=int(all_clips[:,:,j].sum()),
        longest_first_episode_alive_driving_clip_seconds=float(full_longest[:,j].max()*.005)) for j,n in enumerate(motor_names)]
    motor_rows.append(dict(name=g['name'],command=g['command'],first_episode_alive_postwarmup_substep_samples=int(mask.sum()),motors=motors,
        entire_first_living_episode_native_substep_samples=int(all_mask.sum()),
        startup_inclusive_first_episode_motors=all_motor,
        startup_samples_are_not_steady_state_ceiling_evidence=True))
    times=[];cause_counts={n:0 for n in reason_names}
    for robot in range(idx*per,(idx+1)*per):
        events=np.flatnonzero(done[:,robot])
        if len(events):
            first=int(events[0]);times.append(float(time[first]))
            for j,n in enumerate(reason_names):cause_counts[n]+=int(reason[first,robot,j])
    heading=reference[ids]
    signed=root[:,ids,7]*np.cos(heading)[None,:]+root[:,ids,8]*np.sin(heading)[None,:]
    live_mask=alive[:,ids]
    first_failure_rows.append(dict(name=g['name'],command=g['command'],first_episode_failed_robots=len(times),
        first_failure_times_seconds=times,first_failure_reason_counts=cause_counts,
        first_episode_live_control_samples=int(live_mask.sum()),
        transient_maximum_signed_forward_before_terminal_m_s=(float(signed[live_mask].max()) if live_mask.any() else None),
        first_episode_live_mean_signed_forward_m_s=(float(signed[live_mask].mean()) if live_mask.any() else None),
        transient_peak_is_not_sustained_locomotion_or_qualification=True))

primary_qualified=all(row['passed'] and next(c for c in row['cases'] if c['command']==5)['measured_signed_forward_m_s'] is not None and next(c for c in row['cases'] if c['command']==5)['measured_signed_forward_m_s']>=4.95 for row in gate_rows)
four_cases=[next(c for c in row['cases'] if c['command']==4) for row in gate_rows]
fallback_speed_reached=all(c['passed'] and c['measured_signed_forward_m_s'] is not None and c['measured_signed_forward_m_s']>=3.96 for c in four_cases)
result=dict(completed=True,passed=True,checkpoint=pause['checkpoint'],checkpoint_sha256=pause['checkpoint_sha256'],
    gates=gate_rows,physical_groups=physical['groups'],motor_records=motor_rows,first_episode_failures=first_failure_rows,
    sample_rate_Hz=200,independent_driving_clip_mask_matches_while_first_episode_alive=True,
    terminal_tick_and_postfall_reset_motor_samples_excluded=True,
    all_gate_classifiers_and_native_caps_unchanged=True,untrained_commands_above_3_5=True,
    all_matching_existing_and_independently_seeded_gates_pass=primary_qualified,
    endpoint_proven=primary_qualified,measured_4m_tolerance_reached=fallback_speed_reached,causal_velocity_ceiling_proven=False,
    scope='Tests extend through5m/s on the current saved actor without skipping curriculum qualification. Surviving speed, failures and first-episode native driving clips are reported separately. A fall, command or brief cap spike proves neither success nor a persistent actuator ceiling.')
(OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')

fig,axs=plt.subplots(1,3,figsize=(14,4.5))
for row in gate_rows:
    high=[c for c in row['cases'] if c['command']>=3.5];speeds=[c['command'] for c in high]
    values=[[np.nan if c['measured_signed_forward_m_s'] is None else c['measured_signed_forward_m_s'] for c in high],
        [np.nan if c['measured_absolute_lateral_m_s'] is None else c['measured_absolute_lateral_m_s'] for c in high],
        [c['statistics']['survival'] for c in high]]
    for ax,y in zip(axs,values):ax.plot(speeds,y,'o-',label=row['label'])
for ax,title in zip(axs,['Measured signed forward speed','Absolute fixed-frame lateral speed','Complete-trial survival']):
    ax.set_title(title);ax.set_xlabel('Command (m/s)');ax.grid(alpha=.25)
axs[0].plot([3.5,5],[3.5,5],'k--',lw=1,label='command reference')
axs[0].set_ylabel('m/s');axs[1].set_ylabel('m/s');axs[1].axhline(.14,color='k',linestyle='--',lw=1)
axs[2].set_ylabel('fraction');axs[2].set_ylim(-.02,1.03)
axs[0].legend(fontsize=8)
fig.suptitle('Current actor under higher commands — diagnostic extrapolation, all original gates')
fig.tight_layout();fig.savefig(OUT/'higher_speed_results.png',dpi=160);fig.savefig(OUT/'higher_speed_results.svg');plt.close(fig)
print(json.dumps(dict(checkpoint=pause['checkpoint'],higher_gates=[dict(label=r['label'],cases=[dict(command=c['command'],passed=c['passed'],failures=c['failures'],signed=c['statistics']['reset_frame_forward_speed_m_s'],survival=c['statistics']['survival']) for c in r['cases'] if c['command']>=3.5]) for r in gate_rows]),indent=2))
