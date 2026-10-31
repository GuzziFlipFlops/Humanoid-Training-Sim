import json,sys
from pathlib import Path
import torch
OUT=Path(__file__).resolve().parent;RUN=Path(json.loads((OUT/'diagnostic_pause.json').read_text())['run']);sys.path.insert(0,str(RUN/'source/physical_v29/training'))
from upper_body_refinement import posture_loss
from torch import nn
torch.set_num_threads(4)
rows=[]
for file in sorted(OUT.glob('gradient_batch_*.pt')):
 d=torch.load(file,weights_only=False,map_location='cpu');state=d['mlp_state'];weights=[(int(k.split('.')[0]),v) for k,v in state.items() if k.endswith('.weight')];weights.sort()
 layers=[]
 for number,(idx,w) in enumerate(weights):
  layers.append(nn.Linear(w.shape[1],w.shape[0]))
  if number<len(weights)-1:layers.append(nn.ELU())
 m=nn.Sequential(*layers);m.load_state_dict(state);x=d['mlp_input'];y=m(x);assert torch.allclose(y,d['mean'],atol=1e-5,rtol=1e-5)
 params=list(m.parameters());cmd=d['latest_commands'];speed=cmd[:,:2].norm(dim=1);run=speed>=1.5-1e-6;prior=(speed>.1)&(speed<=1.+1e-6);ids=d['ids']
 def vec(derivative):
  return torch.cat([v.flatten() for v in torch.autograd.grad(y,params,grad_outputs=derivative,retain_graph=True)])
 ppo_run=d['incoming_gradient']*run[:,None];ppo_prior=d['incoming_gradient']*prior[:,None];teacher=torch.zeros_like(y);teacher[ids]=d['prior_derivative']
 g_run,g_prior,g_teacher=map(vec,[ppo_run,ppo_prior,teacher])
 # Exactly the auxiliary module's strided sample and last-layer-only derivative.
 stride=max(1,(len(y)+255)//256);sample=torch.arange(0,len(y),stride)[:256];u=y[sample][:,d['upper_indices']].detach().requires_grad_(True);loss,_=posture_loss(u);der=torch.autograd.grad(loss,u)[0];hidden=m[:-1](x).detach()[sample]
 extra_w=torch.zeros_like(params[-2]);extra_b=torch.zeros_like(params[-1]);extra_w[d['upper_indices']]=der.T@hidden;extra_b[d['upper_indices']]=der.sum(0)
 g_upper=torch.cat([torch.zeros_like(v).flatten() for v in params[:-2]]+[extra_w.flatten(),extra_b.flatten()])
 def cosine(a,b):return float((a@b)/(a.norm()*b.norm()).clamp_min(1e-12))
 row=dict(batch=file.name,batch_rows=len(y),running_rows=int(run.sum()),earlier_walking_rows=int(prior.sum()),prior_selected_rows=len(ids),running_ppo_norm=float(g_run.norm()),prior_ppo_norm=float(g_prior.norm()),teacher_norm=float(g_teacher.norm()),upper_aux_norm=float(g_upper.norm()),running_prior_ppo_cosine=cosine(g_run,g_prior),running_teacher_cosine=cosine(g_run,g_teacher),prior_ppo_teacher_cosine=cosine(g_prior,g_teacher),running_upper_aux_cosine=cosine(g_run,g_upper),raw_running_mean_clip_fraction=float((y[run].abs()>3).float().mean()) if run.any() else None,source_mean_reproduction_max_error=float((y-d['mean']).abs().max()))
 rows.append(row)
result=dict(completed=True,batches=rows,scope='Six passive actual PPO minibatches of four-update64-env diagnostic; per-command parameter VJPs before global clipping/Adam. No deployed gradient or actor change; finite sample does not prove all future interference.')
(OUT/'gradient_analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
