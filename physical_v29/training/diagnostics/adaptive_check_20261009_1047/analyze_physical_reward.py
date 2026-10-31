"""Full-term counterfactual on current and independently certified prior states."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
OUT=Path(__file__).resolve().parent
TRAIN=OUT/'control_source/physical_v29/training'
sys.path[:0]=[str(OUT),str(TRAIN)];torch.set_num_threads(4)
from unitree_reward_port import quat_apply_inverse,yaw_quat
from slow_tracking_kernel import score,tracking_width
r=json.loads((OUT/'motor_retention_response/result.json').read_text())
z=np.load(OUT/'motor_retention_response/telemetry.npz')
shape=z['alive'].shape
q=torch.from_numpy(z['root'][:,:,3:7].reshape(-1,4))
v=torch.from_numpy(z['root'][:,:,7:10].reshape(-1,3))
c=torch.from_numpy(z['action_command'].reshape(-1,3))
yaw_v=quat_apply_inverse(yaw_quat(q),v)
error=(yaw_v[:,:2]-c[:,:2]).square().sum(1)
old_raw=torch.exp(-error/.5**2).numpy().reshape(shape)
new_raw=score(c,yaw_v).numpy().reshape(shape)
old_tracking=z['term_track_lin_vel_xy'];reconstructed=4*old_raw*z['healthy']
parity=float(abs(old_tracking-reconstructed).max())
assert parity<2e-5,parity
eligible=((c[:,:2].norm(dim=1)>.1)&(c[:,:2].norm(dim=1)<=1.)).numpy().reshape(shape)
# Preserve observed unchanged terms exactly; GPU-vs-CPU quaternion arithmetic
# can differ in last bits despite numerical reproduction passing.
new_tracking=np.where(eligible,4*new_raw*z['healthy'],old_tracking)
old_full=z['reward_rate'];new_full=old_full+(new_tracking-old_tracking)
assert np.isfinite(new_full).all()
terms={k:z[k] for k in z.files if k.startswith('term_')}
term_error=float(abs(sum(terms.values())-old_full).max());assert term_error<1e-3
window=z['time']>=5.;rows=[];per=int(z['robots_per_group'])
for i,group in enumerate(r['groups']):
    ids=slice(i*per,(i+1)*per);valid=z['alive'][window,ids]
    old=old_full[window,ids][valid];new=new_full[window,ids][valid]
    cmd=group['command'];after=new_tracking[window,ids][valid]
    if cmd>1.:assert np.array_equal(old_tracking[:,ids],new_tracking[:,ids]),cmd
    row=dict(name=group['name'],command=cmd,actor=group['actor'],survival=group['survival'],
        old_complete_rate_mean=float(old.mean()),new_complete_rate_mean=float(new.mean()),
        complete_rate_difference=float((new-old).mean()),old_tracking_rate=float(old_tracking[window,ids][valid].mean()),
        new_tracking_rate=float(after.mean()),mean_signed_forward=group['mean_forward_world_speed'],
        genuine_flight=group['unloaded_geometric_flight_fraction'],head_RMS_deg=group['head_RMS_deg'],
        old_term_means=group['reward_terms'],scope='Same truthful physical trajectories; every other complete objective term held exactly as observed. No causal policy-effect claim.')
    rows.append(row)
comparisons=[]
for seed in [4491,4511]:
    for speed in [.15,.35,.65,1.,1.5,3.]:
        current=next(x for x in rows if x['name']==f'seed{seed}_current_{speed}')
        prior=next(x for x in rows if x['name']==f'seed{seed}_qualified_prior_{speed}')
        comparisons.append(dict(seed=seed,command=speed,
            prior_minus_current_old_full_rate=prior['old_complete_rate_mean']-current['old_complete_rate_mean'],
            prior_minus_current_new_full_rate=prior['new_complete_rate_mean']-current['new_complete_rate_mean'],
            prior_minus_current_old_tracking=prior['old_tracking_rate']-current['old_tracking_rate'],
            prior_minus_current_new_tracking=prior['new_tracking_rate']-current['new_tracking_rate']))
safe=all(x['survival']==1. for x in rows) and float(new_full[z['alive']].min())>=-1e-4
result=dict(completed=True,checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],
    prior_checkpoint=r['actor_checkpoints']['qualified_prior'],prior_sha256=r['actor_sha256']['qualified_prior'],
    robots=sum(x['robots'] for x in r['groups']),duration_seconds=20,initial_seeds=[4491,4511],
    matched_initial_states_per_command_and_actor=32,maximum_tracking_reconstruction_error=parity,
    maximum_reward_sum_error=term_error,minimum_alive_candidate_full_rate=float(new_full[z['alive']].min()),
    safe_to_test_learning=safe,all_faster_reward_terms_bitwise_unchanged=True,
    complete_term_counterfactual=True,controlled_physical_trajectory_unchanged=True,
    policy_improvement_not_proven=True,groups=rows,paired_actor_comparisons=comparisons,
    inference='The broad kernel is a task-precision mismatch hypothesis, not an implementation fault. Narrower width can also punish normal within-cycle speed variation. Real matched learning and all gates determine acceptance.',
    sources=[dict(url='https://isaac-sim.github.io/IsaacLab/develop/source/api/lab/isaaclab.envs.mdp.html',role='Official exponential tracking formula'),
        dict(url='https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/tasks/velocity/velocity_env_cfg.py',role='Official task uses0.5m/s width; not evidence this local adaptation will converge')])
(OUT/'physical_reward_counterfactual.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['groups','paired_actor_comparisons']},indent=2))
