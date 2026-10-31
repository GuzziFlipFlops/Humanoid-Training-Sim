"""Independent recorded-state review; physical screens never replace gates."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;SECOND=OUT/'secondary_analysis';F=SECOND/'teacher_compatibility_response'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=read(F/'result.json');pause=read(SECOND/'diagnostic_pause.json');cert=read(OUT/'neutral_teacher_certificate.json')
assert r['completed'] and r['actor_loaded_exact'] and r['checkpoint_sha256']==pause['checkpoint_sha256']
assert r['actor_sha256']['neutral_prior']==cert['checkpoint_sha256'] and sha(cert['checkpoint'])==cert['checkpoint_sha256']
assert r['maximum_command_difference']==0 and r['maximum_reward_term_sum_error']<1e-3
assert r['initial_state_seeds']==[4631,4651] and len(r['groups'])==20 and r['composite_tick_assertions']==1000
assert not r['diagnostic_actors_deployed'] and r['observation_rewrites'] is False and r['normal_motor_limits']
with np.load(F/'telemetry.npz') as z:
    alive=z['alive'];done=z['failures'];initial_q=z['initial_q'];initial_qd=z['initial_qd'];time=z['time']
    force=z['force'];clearance=z['clearance'];phase=z['reward_phase'];root=z['root'];heading=z['reference_heading']
    target_clipped=z['target_clipped'];slew=z['target_slew'];healthy=z['healthy'];names=z['joint_names'].tolist()
assert alive.shape==(1000,640) and np.array_equal(alive,np.cumprod(~done,axis=0).astype(bool))
hashes=[]
for i in range(20):
    state=np.concatenate([initial_q[i*32:(i+1)*32],initial_qd[i*32:(i+1)*32]],1)
    assert len({row.tobytes() for row in state})==32
    hashes.append(hashlib.sha256(state.tobytes()).hexdigest())
assert len(set(hashes[:10]))==1 and len(set(hashes[10:]))==1 and hashes[0]!=hashes[10]
window=time>=5.;records=[]
for i,g in enumerate(r['groups']):
    ids=slice(i*32,(i+1)*32);valid=alive[:,ids]&window[:,None]
    flight=(clearance[:,ids]>.001).all(-1)&(force[:,ids]<1.).all(-1)&alive[:,ids]
    empirical=float(flight[window].mean());assert abs(empirical-g['unloaded_geometric_flight_fraction'])<1e-8
    pos=root[:,:,0:2];delta=pos[-1,ids]-pos[249,ids]
    signed_net=delta[:,0]*np.cos(heading[ids])+delta[:,1]*np.sin(heading[ids])
    step=np.diff(pos[249:,ids],axis=0);path=np.linalg.norm(step,axis=2).sum(0)
    efficiency=signed_net/np.maximum(path,1e-12)
    bins=[]
    for lo in np.arange(0.,1.,.125):
        selected=valid&(phase[:,ids]>=lo)&(phase[:,ids]<lo+.125)
        bins.append(dict(phase_low=float(lo),phase_high=float(lo+.125),samples=int(selected.sum()),
            flight_fraction=float(flight[selected].mean()) if selected.any() else None,
            summed_sole_force_N=float(force[:,ids].sum(-1)[selected].mean()) if selected.any() else None,
            minimum_sole_clearance_m=float(clearance[:,ids].min(-1)[selected].mean()) if selected.any() else None))
    records.append(dict(name=g['name'],command=g['command'],controller=g['actor'],survival=g['survival'],resets=g['resets'],
        signed_forward_m_s=g['mean_forward_world_speed'],absolute_world_lateral_m_s=g['mean_absolute_world_lateral_velocity'],
        genuine_flight_fraction=empirical,head_RMS_deg=g['head_RMS_deg'],complete_mean_reward_rate=g['mean_reward_rate'],
        physical_reward_terms=g['reward_terms'],postwarmup_signed_net_m_mean=float(signed_net.mean()),
        postwarmup_forward_path_efficiency_mean=float(efficiency.mean()),phase_force_clearance=bins,
        target_joint_clip_fraction=float(target_clipped[:,ids][valid].mean()),target_slew_fraction=float(slew[:,ids][valid].mean()),
        healthy_mean=float(healthy[:,ids][valid].mean())))
del root,force,clearance,target_clipped,slew,healthy
with np.load(F/'substep_motor_200hz.npz') as z:
    cap=z['cap_ratio'];request=z['requested_torque'];applied=z['applied_torque'];velocity=z['native_velocity'];saved=z['speed_clip'];motor_names=z['motor_names'].tolist()
valid=np.repeat(alive,4,axis=0)
assert len(valid)==4000
independent=(cap>=1)&(request*velocity>0)&(applied*velocity<=0)
assert np.array_equal(independent[valid],saved[valid])
motor_records=[]
for i,g in enumerate(r['groups']):
    ids=slice(i*32,(i+1)*32);mask=valid[1000:,ids];clips=saved[1000:,ids]&mask[:,:,None]
    streak=np.zeros((32,len(motor_names)),dtype=int);longest=streak.copy()
    for tick in clips:streak=np.where(tick,streak+1,0);longest=np.maximum(longest,streak)
    motor_records.append(dict(name=g['name'],first_episode_valid_substeps=int(mask.sum()),
        driving_clip_count=int(clips.sum()),maximum_contiguous_driving_clip_s=float(longest.max()*.005),
        motors={name:dict(count=int(clips[:,:,j].sum()),fraction=float(clips[:,:,j].sum()/max(1,mask.sum())),
            maximum_contiguous_driving_clip_s=float(longest[:,j].max()*.005)) for j,name in enumerate(motor_names)}))
screen={}
for controller in ['neutral_body_current_upper','neutral_body_current_head']:
    own=[g for g in records if g['controller']==controller]
    passed=all(g['survival']==1 and g['resets']==0 and g['absolute_world_lateral_m_s']<.14
        and (g['command']!=1.5 or g['genuine_flight_fraction']>.015)
        and (g['command']!=.15 or abs(g['signed_forward_m_s']-.15)<.1) for g in own)
    screen[controller]=dict(bounded_physical_screen_passed=passed,
        complete_return_at_least_current_at_each_command=all(g['complete_mean_reward_rate']>=next(x['complete_mean_reward_rate'] for x in records
            if x['controller']=='current' and x['command']==g['command'] and x['name'].split('_')[0]==g['name'].split('_')[0]) for g in own))
diverse={test:dict(passed=read(SECOND/test/'evaluation.json')['passed'],
    failures=[dict(command=c['statistics']['command_m_s'],failures=c['failures']) for c in read(SECOND/test/'evaluation.json')['cases'] if not c['passed']],
    initial_state_receipt=str(SECOND/test/'initial_diversity.json')) for test in ['diverse_nominal','diverse_push']}
result=dict(completed=True,current_checkpoint=pause['checkpoint'],current_sha256=pause['checkpoint_sha256'],neutral_teacher_certificate=cert,
    records=records,diverse_teacher_gates=diverse,physical_screens=screen,independent_state_hashes=hashes,
    native_5ms_first_episode_motor_attribution=motor_records,independent_driving_clip_reconstruction_passed=True,
    no_production_composite=True,candidate_learning_tested=False,qualification_established=False,
    scope='Fresh neutral267 qualification and two-seed640robot20s truthful closed-loop component test. Each actor reads actual physical feedback; feedback and mechanics are not isolated. Full reward remains original. The bounded compatibility screen is only a prerequisite for a training experiment, not locomotion qualification. Motor statistics exclude terminal/reset ticks and do not establish an actuator ceiling.')
(OUT/'neutral_teacher_response_analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
labels=['current','qualified_prior','neutral_prior','neutral_body_current_upper','neutral_body_current_head']
fig,axes=plt.subplots(1,3,figsize=(16,4.6))
for seed,marker in [(4631,'o'),(4651,'x')]:
    for ax,command,key,scale in [(axes[0],.15,'signed_forward_m_s',1),(axes[1],1.5,'genuine_flight_fraction',100),(axes[2],1.5,'complete_mean_reward_rate',1)]:
        vals=[next(g for g in records if g['name']==f'seed{seed}_{label}_{command}') for label in labels]
        ax.plot(range(5),[g[key]*scale for g in vals],marker+'-',label='seed'+str(seed))
for ax,title in zip(axes,['0.15 command: signed speed m/s','1.5 command: genuine flight %','1.5 command: complete reward rate']):
    ax.set_title(title);ax.set_xticks(range(5),['current','qualified349','neutral267','267 body\ncurrent upper','267 body/arms\ncurrent head']);ax.grid(alpha=.25);ax.legend(fontsize=8)
axes[0].axhline(.15,color='k',ls='--',lw=1);axes[1].axhline(1.5,color='k',ls='--',lw=1)
fig.suptitle('Freshly qualified neutral-head checkpoint and component compatibility — diagnostic only')
fig.tight_layout();fig.savefig(OUT/'neutral_teacher_response.png',dpi=160);fig.savefig(OUT/'neutral_teacher_response.svg');plt.close(fig)
print(json.dumps(dict(completed=True,physical_screens=screen,diverse_teacher_gates=diverse,
    records=[{k:g[k] for k in ['name','signed_forward_m_s','genuine_flight_fraction','absolute_world_lateral_m_s','complete_mean_reward_rate','survival']} for g in records]),indent=2))
