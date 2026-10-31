import json,hashlib
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def cases(folder):
 e=read(folder/'evaluation.json');d=np.load(folder/'telemetry.npz');names=list(d['joint_names']);heads=[names.index('head_yaw'),names.index('head_pitch')];rows=[]
 for ci,case in enumerate(e['cases']):
  st=case['statistics'];mask=d['case_labels']==ci;q=d['q'][:,mask][:,:,heads]
  arms={}
  for joint in ['left_shoulder_pitch','right_shoulder_pitch','left_shoulder_roll','right_shoulder_roll','left_elbow_pitch','right_elbow_pitch']:
   j=names.index(joint);position=d['q'][:,mask,j];vel=d['qd'][:,mask,j]
   arms[joint]=dict(mean_deg=float(position.mean()*180/np.pi),p95_abs_deg=float(np.quantile(np.abs(position),.95)*180/np.pi),rms_velocity_rad_s=float(np.sqrt(np.mean(vel*vel))))
  rows.append(dict(command=st['command_m_s'],passed=case['passed'],failures=case['failures'],
   speed=st['reset_frame_forward_speed_m_s'],flight=st['geometric_flight_fraction'],survival=st['survival'],
   head_combined_RMS_deg=float(np.sqrt(np.mean(np.sum(q*q,-1)))*180/np.pi),
   head_within_5deg_fraction=float((np.abs(q).max(-1)<=np.deg2rad(5)).mean()),arms=arms))
 return rows
arms={};decisions={}
for boundary in [64,128]:
 if not all((OUT/f'{kind}_{boundary:03d}_{mode}/evaluation.json').exists() for kind in ['control','candidate'] for mode in ['nominal','push']):continue
 arm={}
 for kind in ['control','candidate']:
  folder=OUT/f'{kind}_{boundary:03d}_train'
  if kind=='control' and boundary==128:folder=OUT/'control_128_train_retry01'
  cfg=read(folder/'run_config.json');proof=read(folder/'full_load_proof.json')
  assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
  progress=[json.loads(s) for s in (folder/'progress.jsonl').read_text().splitlines()]
  assert len(progress)==64 and all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in progress)
  cp=folder/'candidate.pt';nom=OUT/f'{kind}_{boundary:03d}_nominal';push=OUT/f'{kind}_{boundary:03d}_push'
  for gate in [nom,push]:assert Path(read(gate/'run_config.json')['arguments']['checkpoint']).resolve()==cp.resolve()
  arm[kind]=dict(checkpoint=str(cp),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
   nominal=cases(nom),pushed=cases(push),full_load_exact=True,updates=64,
   zero_KL_command_mismatch=True,min_GPU_MiB=min(r['gpu']['free_mb'] for r in progress),
   min_host_MiB=min(r['gpu']['ram_available_mb'] for r in progress))
 priors=all(r['passed'] for mode in ['nominal','pushed'] for r in arm['candidate'][mode] if r['command']<=1.)
 flight=all(arm['candidate'][mode][-1]['flight']>arm['control'][mode][-1]['flight'] for mode in ['nominal','pushed'])
 run=all(arm['candidate'][mode][-1]['passed'] for mode in ['nominal','pushed'])
 decisions[str(boundary)]=dict(all_prior_pass=priors,flight_improves_both=flight,full_running_cases_pass=run,
  deploy_supported=priors and flight and run,criterion='Require matching stopped/prior retention plus improved genuinely airborne running, not a numerical teacher-loss change alone.')
 arms[str(boundary)]=arm
record=dict(completed_boundaries=list(arms),arms=arms,decisions=decisions,diagnostic_actors_deployed=False,
 concludes_repair=False,scope='Matched learning seeds, initial full state and unchanged physical/reward contract at 4096 environments; 192 robots/20s/32 trials per command for each nominal and pushed pair.')
(OUT/'matched_comparison.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(boundaries=list(arms),decisions=decisions,brief={b:{k:{mode:[dict(command=r['command'],passed=r['passed'],speed=r['speed'],flight=r['flight'],head=r['head_combined_RMS_deg']) for r in arm[k][mode]] for mode in ['nominal','pushed']} for k in arm} for b,arm in arms.items()}),indent=2))
