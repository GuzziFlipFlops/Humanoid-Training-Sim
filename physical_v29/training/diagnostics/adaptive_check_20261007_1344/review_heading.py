"""Exact saved matched actor/evaluation comparison, CPU only."""
import json,hashlib,argparse
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_heading'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
def cases(folder):
 r=read(folder/'evaluation.json');z=np.load(folder/'telemetry.npz');names=z['joint_names'].tolist();head=[names.index('head_yaw'),names.index('head_pitch')]
 result=[]
 for i,c in enumerate(r['cases']):
  s=c['statistics'];mask=z['case_labels']==i
  result.append(dict(command=s['command_m_s'],passed=c['passed'],failures=c['failures'],statistics=s,
   head_RMS_deg=float(np.rad2deg(np.sqrt(np.mean(np.sum(z['q'][:,mask][:,:,head]**2,-1)))))))
 return dict(passed=r['passed'],cases=result,seed=r['seed'],perturb=r['perturb'],folder=str(folder))
rows=[]
for branch in ['control','candidate']:
 for update in [64,128]:
  train=ROOT/f'{branch}_{update:03d}_train';nom=ROOT/f'{branch}_{update:03d}_nominal';push=ROOT/f'{branch}_{update:03d}_push'
  if not all((p/'result.json').exists() for p in [train,nom,push]):
   assert args.partial,'Incomplete matched comparison';continue
  load=read(train/'full_load_proof.json');assert all(load[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
  progress=[json.loads(x) for x in (train/'progress.jsonl').read_text().splitlines()]
  assert len(progress)==64 and all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in progress)
  row=dict(branch=branch,updates=update,checkpoint=str(train/'candidate.pt'),sha256=hashlib.sha256((train/'candidate.pt').read_bytes()).hexdigest(),
   nominal=cases(nom),pushed=cases(push),exact_full_load=load,KL_command_mismatch_zero=True,
   conditioned_samples=sum(r['locomotion_retention']['teacher_phase_conditioned_samples'] for r in progress),
   raw_teacher_MSE_mean=sum(r['locomotion_retention']['standing_prior_mse'] for r in progress)/len(progress),
   min_free_gpu_MiB=min(r['gpu']['free_mb'] for r in progress),
   all_upper_and_priors20_batches=all(r['upper_refinement']['batches']==20 and r['retention']['batches']==20 and r['locomotion_retention']['batches']==20 for r in progress))
  for suffix in ['push_seed1901','diverse_nominal']:
   extra=ROOT/f'{branch}_{update:03d}_{suffix}'
   if (extra/'result.json').exists():row[suffix]=cases(extra)
   if (extra/'initial_diversity.json').exists():row['initial_diversity']=read(extra/'initial_diversity.json')
  rows.append(row)
teacher={}
for name in ['nominal','push']:
 f=ROOT/f'truthful_teacher_{name}'
 if (f/'result.json').exists():teacher[name]=cases(f)
result=dict(completed=not args.partial,comparisons=rows,truthful_teacher=teacher,
 native_physics_actions_PPO_gates_upper_weights_unchanged=True,diagnostic_actors_deployed=False,
 original_signature_does_not_certify_training_override=True,nominal_clone_seeds_not_initial_diversity=True)
(OUT/('matched_heading_partial.json' if args.partial else 'matched_heading_review.json')).write_text(json.dumps(result,indent=2))
print(json.dumps([dict(branch=r['branch'],updates=r['updates'],current={k:{v:r[k]['cases'][-1]['statistics'][v] for v in ['reset_frame_forward_speed_m_s','reset_frame_lateral_speed_m_s','geometric_flight_fraction']} for k in ['nominal','pushed']},
 failures={k:[dict(command=c['command'],failures=c['failures']) for c in r[k]['cases'] if not c['passed']] for k in ['nominal','pushed']}) for r in rows],indent=2))
