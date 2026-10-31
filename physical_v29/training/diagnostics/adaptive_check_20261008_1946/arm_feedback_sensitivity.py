"""CPU-only response to saved arm-history variation; never production input."""
import hashlib,json,torch
from pathlib import Path
from torch import nn
OUT=Path(__file__).resolve().parent;torch.set_num_threads(4)
pause=json.loads((OUT/'diagnostic_pause.json').read_text())
f=OUT/'current_gradient_00.pt';d=torch.load(f,map_location='cpu',weights_only=False)
model=nn.Sequential(nn.Linear(400,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,23));model.load_state_dict(d['state']);model.eval()
raw=d['raw_observations'];assert torch.equal(raw,d['mlp_input']),'Cannot apply raw counterfactual without captured normalization'
with torch.no_grad():
 original=model(raw);assert float((original-d['mean']).abs().max())<6e-5
arm=d['upper_indices'][2:];lower=[i for i in range(23) if i not in d['upper_indices']]
frames=raw.reshape(-1,5,80);speed=frames[:,:,6:8].norm(dim=-1)
phase=torch.remainder(torch.atan2(frames[:,-1,78],frames[:,-1,79])/(2*torch.pi),1.);bins=(phase*8).long()
rows=[]
for command in [.15,.35,.65,1.,1.5,2.,2.5,3.,3.5]:
 ids=((speed-command).abs()<1e-6).all(1).nonzero().flatten()
 if len(ids)<2:continue
 source=[];donor=[]
 for b in range(8):
  chosen=ids[bins[ids]==b]
  if len(chosen)>1:source.extend(chosen.tolist());donor.extend(chosen.roll(1).tolist())
 if not source:continue
 source=torch.tensor(source);donor=torch.tensor(donor);base=frames[source].clone();replacement=frames[donor]
 old=original[source];stats=[]
 for name,columns in [('arm_q',[9+i for i in arm]),('arm_qd',[32+i for i in arm]),('previous_arm_actions',[55+i for i in arm]),
   ('arm_q_qd_previous_actions',[offset+i for offset in [9,32,55] for i in arm])]:
  changed=base.clone();changed[:,:,columns]=replacement[:,:,columns]
  # Head, roots, commands, phases and all lower/body feedback remain exact.
  unchanged=[j for j in range(80) if j not in columns];assert torch.equal(changed[:,:,unchanged],base[:,:,unchanged])
  with torch.no_grad():y=model(changed.reshape(-1,400))
  delta=(y.clamp(-3,3)-old.clamp(-3,3))[:,lower]
  stats.append(dict(counterfactual=name,rows=len(source),raw_lower_output_RMS=float((y-old)[:,lower].square().mean().sqrt()),
   effective_clipped_lower_output_RMS=float(delta.square().mean().sqrt()),effective_clipped_lower_output_max=float(delta.abs().max()),
   input_feedback_RMS=float((changed-base).square().mean().sqrt()),
   note='Donor histories are saved truthful states at the same exact command and one of eight phase bins. The recombined whole-body state need not be dynamically realizable. No physical coupling conclusion.'))
 rows.append(dict(command=command,stable_history_rows=len(ids),paired_counterfactual_rows=len(source),statistics=stats))
result=dict(completed=True,capture_file=str(f),capture_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),checkpoint=pause['checkpoint'],
 checkpoint_sha256=pause['checkpoint_sha256'],scope='First actual PPO minibatch, captured model and truthful states. CPU counterfactual only; finite response shows input dependence, not causality, safe arm trajectories or mechanical effects. No production observations altered.',
 selected_arm_output_indices=arm,direct_lower_derivative_or_policy_change=False,physical_arm_base_coupling_identified=False,records=rows)
(OUT/'arm_feedback_sensitivity.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps([dict(command=r['command'],rows=r['paired_counterfactual_rows'],effective_RMS={s['counterfactual']:s['effective_clipped_lower_output_RMS'] for s in r['statistics']}) for r in rows]))
