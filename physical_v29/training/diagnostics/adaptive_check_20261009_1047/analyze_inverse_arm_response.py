"""Independent saved-data review of current-body/current-head arm diagnosis."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'high_speed_extrapolation';F=OUT/'inverse_arm_response'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
r=read(F/'result.json');pause=read(OUT/'diagnostic_pause.json');manifest=read(F/'controller_manifest.json')
assert r['completed'] and r['checkpoint_sha256']==pause['checkpoint_sha256']
assert r['actor_loaded_exact'] and r['maximum_command_difference']==0 and r['maximum_reward_term_sum_error']<1e-3
assert r['initial_state_seeds']==[4591,4611] and len(r['groups'])==12
assert all(g['command']<=3. and g['robots']==32 for g in r['groups'])
assert manifest['current_head_and_body_preserved'] and manifest['production_composite_deployed'] is False
assert manifest['helper_sha256']==read(OUT/'inverse_arm_CPU_tests.json')['helper_sha256']
with np.load(F/'telemetry.npz') as z:
    alive=z['alive'];done=z['failures'];initial_q=z['initial_q'];initial_qd=z['initial_qd']
assert np.array_equal(alive,np.cumprod(~done,axis=0).astype(bool))
state_hashes=[]
for i in range(12):
    state=np.concatenate([initial_q[i*32:(i+1)*32],initial_qd[i*32:(i+1)*32]],1)
    assert len({x.tobytes() for x in state})==32
    state_hashes.append(hashlib.sha256(state.tobytes()).hexdigest())
for seed in [4591,4611]:
    selected=[i for i,g in enumerate(r['groups']) if g['name'].startswith('seed'+str(seed)+'_')]
    assert len({state_hashes[i] for i in selected})==1
with np.load(F/'substep_motor_200hz.npz') as z:
    cap=z['cap_ratio'];request=z['requested_torque'];applied=z['applied_torque'];velocity=z['native_velocity'];saved=z['speed_clip'];names=z['motor_names'].tolist()
valid=np.repeat(alive,4,axis=0)
assert len(valid)==4000
independent=(cap>=1)&(request*velocity>0)&(applied*velocity<=0)
assert np.array_equal(independent[valid],saved[valid])
records=[]
for i,g in enumerate(r['groups']):
    mask=valid[1000:,i*32:(i+1)*32]
    clips=saved[1000:,i*32:(i+1)*32]&mask[:,:,None]
    records.append(dict(name=g['name'],command=g['command'],actor=g['actor'],survival=g['survival'],resets=g['resets'],
        signed_forward_m_s=g['mean_forward_world_speed'],absolute_lateral_m_s=g['mean_absolute_world_lateral_velocity'],
        genuine_flight_fraction=g['unloaded_geometric_flight_fraction'],head_RMS_deg=g['head_RMS_deg'],
        mean_reward_rate=g['mean_reward_rate'],first_episode_postwarmup_native_motor_samples=int(mask.sum()),
        maximum_motor_driving_clip_fraction=(float((clips.sum((0,1))/mask.sum()).max()) if mask.any() else None),
        driving_clip_count_by_motor={n:int(clips[:,:,j].sum()) for j,n in enumerate(names)}))
physical_promising=all(g['survival']==1 and (g['command']!=1.5 or g['genuine_flight_fraction']>=.015)
    and (g['command']!=.15 or abs(g['signed_forward_m_s']-.15)<=.1) for g in records if g['actor']=='current_body_teacher_arms')
result=dict(completed=True,checkpoint_sha256=pause['checkpoint_sha256'],qualified_teacher_sha256=r['actor_sha256']['qualified_prior'],
    records=records,controller_manifest=manifest,initial_state_sha256_by_cohort=state_hashes,
    completed_physical_ticks_with_exact_action_row_preservation=1000,
    recorded_full_reward_sum_maximum_error=r['maximum_reward_term_sum_error'],
    independent_native_driving_mask_matches_on_first_episode=True,initial_states_per_command_controller_seed=32,
    bounded_physical_candidate_promising=physical_promising,qualification_established=False,production_composite_deployed=False,
    candidate_learning_tested=False,
    scope='Two-seed 384-robot20second truthful closed-loop component diagnosis, commands0.15/1.5 within the exact teacher certificate. Current head and every non-arm action row are preserved per tick before state response; both networks read actual physical observations/history. Feedback and mechanical effects remain coupled. No production hybrid, observation substitution, new learned actor or above-certificate teacher label. Native motor summaries exclude terminal ticks and later reset episodes.')
(ROOT/'inverse_arm_response_analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
fig,axes=plt.subplots(1,3,figsize=(13,4.5));labels=['current','qualified_prior','current_body_teacher_arms']
for seed,marker in [(4591,'o'),(4611,'x')]:
    for axis,command,key,factor in [(axes[0],.15,'signed_forward_m_s',1),(axes[1],1.5,'genuine_flight_fraction',100),(axes[2],1.5,'head_RMS_deg',1)]:
        data=[next(g for g in records if g['name']==f'seed{seed}_{label}_{command}') for label in labels]
        axis.plot(range(3),[np.nan if g[key] is None else factor*g[key] for g in data],marker+'-',label='seed'+str(seed))
for axis,title in zip(axes,['0.15 command: signed forward m/s','1.5 command: genuine flight %','1.5 command: actual neck RMS deg']):
    axis.set_title(title);axis.set_xticks(range(3),['current','qualified\nprior','current body/head\nprior arms']);axis.grid(alpha=.25);axis.legend(fontsize=8)
axes[0].axhline(.15,color='k',linestyle='--',lw=1);axes[1].axhline(1.5,color='k',linestyle='--',lw=1)
fig.suptitle('Inverse arms-only component test — diagnostic, no production composite')
fig.tight_layout();fig.savefig(ROOT/'inverse_arm_response.png',dpi=160);fig.savefig(ROOT/'inverse_arm_response.svg');plt.close(fig)
print(json.dumps(dict(completed=True,bounded_physical_candidate_promising=physical_promising,records=records),indent=2))
