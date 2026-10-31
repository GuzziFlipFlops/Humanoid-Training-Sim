"""CPU matched-weight response on recorded truthful initial training states.

This does not isolate PPO, optimizer/clipping or physical feedback. The recorded
MLP input transform is deliberately fixed, and is not a production substitution.
"""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
import torch
from torch import nn
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_row_route'
torch.set_num_threads(4)
def model():return nn.Sequential(nn.Linear(400,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,23))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
capture=OUT/'current_gradient_00.pt';d=torch.load(capture,map_location='cpu',weights_only=False)
raw=d['raw_observations'];speed=raw[:,-80+6];ids=[];groups={}
for c in [0.,.15,.35,.65,1.,1.5,2.,2.5,3.,3.5]:
 mask=(speed-c).abs()<1e-6 if c!=3.5 else (speed>=3.25)&(speed<=3.5+1e-6)
 candidates=mask.nonzero().flatten();chosen=candidates[::max(1,(len(candidates)+127)//128)][:128]
 groups[str(c)]=(len(ids),len(ids)+len(chosen));ids.extend(chosen.tolist())
idx=torch.tensor(ids,dtype=torch.long);inputs=d['mlp_input'][idx]
base=model();anchor=Path(read(OUT/'diagnostic_pause.json')['checkpoint'])
def load(p):
 m=model();state=torch.load(p,map_location='cpu',weights_only=False)['actor_state_dict']
 m.load_state_dict({k.removeprefix('mlp.'):v for k,v in state.items() if k.startswith('mlp.')});m.eval();return m
base=load(anchor);head=d['upper_indices'][:2];arms=d['upper_indices'][2:];body=[i for i in range(23) if i not in d['upper_indices']]
with torch.no_grad():initial=base(inputs);initial_features=base[:-1](inputs)
records=[]
for seed in [251,263]:
 for branch in ['control','shared','rows']:
  for updates in [64,128]:
   cp=ROOT/f'seed{seed}_{branch}_{updates:03d}_train/candidate.pt'
   if not cp.is_file():continue
   m=load(cp)
   with torch.no_grad():y=m(inputs);features=m[:-1](inputs);delta=y-initial;endpoint=y.clamp(-3,3)-initial.clamp(-3,3)
   commands=[]
   for name,(start,end) in groups.items():
    if start==end:continue
    sl=slice(start,end)
    commands.append(dict(command=name,samples=end-start,frontier_is_3_25_to_3_5_band=name=='3.5',
     raw_output_change_RMS=float(delta[sl].square().mean().sqrt()),clamped_action_endpoint_change_RMS=float(endpoint[sl].square().mean().sqrt()),
     head_raw_change_RMS=float(delta[sl][:,head].square().mean().sqrt()),arm_raw_change_RMS=float(delta[sl][:,arms].square().mean().sqrt()),
     body_raw_change_RMS=float(delta[sl][:,body].square().mean().sqrt()),last_feature_change_RMS=float((features[sl]-initial_features[sl]).square().mean().sqrt())))
   records.append(dict(seed=seed,branch=branch,updates=updates,checkpoint=str(cp),sha256=sha(cp),commands=commands))
result=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),all_12_checkpoints_complete=len(records)==12,
 anchor=str(anchor),anchor_sha256=sha(anchor),recorded_states=str(capture),recorded_states_sha256=sha(capture),
 actual_native_head_rows=head,records=records,production_change=False,
 scope='At most128 deterministically sampled states per command from the first genuine captured PPO minibatch. Truthful raw states identify command, but MLP inputs use their fixed recorded transform. Weight response only; no mechanical/optimizer/teacher-only attribution, deterministic gate coverage or gait guarantee. Clamp differences are action endpoints, not physical joint/slew/CAD targets.')
(OUT/'fixed_state_learning_response.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(complete=result['all_12_checkpoints_complete'],records=len(records),selected_states=len(ids))))
