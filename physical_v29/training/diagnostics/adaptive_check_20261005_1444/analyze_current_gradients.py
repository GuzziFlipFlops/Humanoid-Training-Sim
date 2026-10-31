import torch,json,sys
from pathlib import Path
from torch import nn
OUT=Path(__file__).resolve().parent;torch.set_num_threads(4)
rows=[]
for file in sorted(OUT.glob('current_gradient_*.pt')):
 d=torch.load(file,map_location='cpu',weights_only=False)
 m=nn.Sequential(nn.Linear(400,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,23))
 m.load_state_dict(d['state']);y=m(d['mlp_input']);assert torch.allclose(y,d['mean'],atol=2e-5,rtol=2e-5)
 ids=d['ids'];target=d['teacher_raw_targets'].clamp(-3,3);difference=y[ids].detach()-target;difference[:,d['upper_indices']]=0
 derivative=.5*difference/difference.numel()
 same_executed=(y[ids].detach().clamp(-3,3)==target)&(y[ids].detach().abs()>3)
 plateau=derivative*same_executed
 remaining=derivative-plateau
 parameters=list(m.parameters());speed=d['raw_observations'][:,-80+6:-80+8].norm(dim=1)
 def vjp(gradient):return torch.cat([g.flatten() for g in torch.autograd.grad(y,parameters,grad_outputs=gradient,retain_graph=True)])
 prior=torch.zeros_like(y);prior[ids]=derivative;flat=torch.zeros_like(y);flat[ids]=plateau
 gp,gf=vjp(prior),vjp(flat)
 run_der=d['incoming_gradient']*(speed>=1.5-1e-6)[:,None];gr=vjp(run_der)
 prior_ppo=d['incoming_gradient']*((speed>.1)&(speed<=1.+1e-6))[:,None];gpp=vjp(prior_ppo)
 cos=lambda a,b:float(a@b/(a.norm()*b.norm()).clamp_min(1e-12))
 raw=d['raw_observations'][ids].reshape(-1,5,80);teacher=d['teacher_inputs'].reshape(-1,5,80)
 changed=(raw!=teacher).any(-1).any(-1)
 upper=[i for i in range(23) if i not in d['upper_indices']]
 commands=[]
 for command in [.15,.35,.65,1.]:
  mask=(raw[:,-1,6]-command).abs()<1e-6
  if not mask.any():continue
  rd=difference[mask][:,upper];ed=(y[ids][mask].detach().clamp(-3,3)-target[mask])[:,upper]
  commands.append(dict(command=command,selected=int(mask.sum()),conditioned_fraction=float(changed[mask].float().mean()),
   raw_lower_MSE=float(rd.square().mean()),effective_action_MSE=float(ed.square().mean()),
   same_endpoint_raw_error_fraction=float((plateau[mask].square().sum()/derivative[mask].square().sum().clamp_min(1e-12))),
   latest_head_RMS_deg=float(raw[mask][:,-1,[20,25]].square().mean().sqrt()*180/torch.pi)))
 rows.append(dict(batch=file.name,reproduction_max_error=float((y-d['mean']).abs().max()),
  batch_samples=len(y),selected=len(ids),per_command=commands,
  prior_parameter_norm=float(gp.norm()),plateau_parameter_norm=float(gf.norm()),
  plateau_prior_cosine=cos(gf,gp),running_PPO_norm=float(gr.norm()),prior_PPO_norm=float(gpp.norm()),
  running_teacher_cosine=cos(gr,gp),prior_PPO_teacher_cosine=cos(gpp,gp),
  plateau_teacher_derivative_energy_fraction=float(plateau.square().sum()/derivative.square().sum().clamp_min(1e-12)),
  running_raw_action_clip_fraction=float((y[speed>=1.5-1e-6].abs()>3).float().mean())))
result=dict(completed=True,batches=rows,scope='Four genuine PPO minibatches at 4096 environments. Passive labels and pre-auxiliary VJPs before clipping/Adam; finite-state evidence, not a causally established running failure.')
(OUT/'current_gradient_analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
