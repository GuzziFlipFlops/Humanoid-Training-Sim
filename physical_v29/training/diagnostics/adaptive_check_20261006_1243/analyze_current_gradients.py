import torch,json,sys
from pathlib import Path
from torch import nn
OUT=Path(__file__).resolve().parent;torch.set_num_threads(4)
sys.path.insert(0,str(OUT/'control_source/physical_v29/training'))
from upper_body_refinement import posture_loss
pause=json.loads((OUT/'diagnostic_pause.json').read_text())
cert=json.loads((Path(pause['checkpoint']).parent/'qualified_prior_certificate.json').read_text())
teacher_state=torch.load(cert['checkpoint'],map_location='cpu',weights_only=False)['actor_state_dict']
teacher_m=nn.Sequential(nn.Linear(400,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,23))
teacher_m.load_state_dict({k.removeprefix('mlp.'):v for k,v in teacher_state.items() if k.startswith('mlp.')})
teacher_m.requires_grad_(False)
smoke_arguments=json.loads((OUT/'gradient_smoke/run_config.json').read_text())['arguments']
stopped_state=torch.load(smoke_arguments['standing_prior'],map_location='cpu',weights_only=False)['actor_state_dict']
stopped_m=nn.Sequential(nn.Linear(400,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,23))
stopped_m.load_state_dict({k.removeprefix('mlp.'):v for k,v in stopped_state.items() if k.startswith('mlp.')});stopped_m.requires_grad_(False)
rows=[]
for file in sorted(OUT.glob('current_gradient_*.pt')):
 d=torch.load(file,map_location='cpu',weights_only=False)
 m=nn.Sequential(nn.Linear(400,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,23))
 m.load_state_dict(d['state']);y=m(d['mlp_input']);assert float((y-d['mean']).abs().max())<6e-5
 ids=d['ids'];target=d['teacher_raw_targets'].clamp(-3,3);difference=y[ids].detach()-target;difference[:,d['upper_indices']]=0
 derivative=.5*difference/difference.numel()
 same_executed=(y[ids].detach().clamp(-3,3)==target)&(y[ids].detach().abs()>3)
 plateau=derivative*same_executed
 remaining=derivative-plateau
 parameters=list(m.parameters());speed=d['raw_observations'][:,-80+6:-80+8].norm(dim=1)
 def vjp(gradient):return torch.cat([g.flatten() for g in torch.autograd.grad(y,parameters,grad_outputs=gradient,retain_graph=True)])
 prior=torch.zeros_like(y);prior[ids]=derivative;flat=torch.zeros_like(y);flat[ids]=plateau
 gp,gf=vjp(prior),vjp(flat)
 run_der=d['incoming_gradient']*(speed>=3.5-1e-6)[:,None];gr=vjp(run_der)
 prior_ppo=d['incoming_gradient']*((speed>.1)&(speed<=3.+1e-6))[:,None];gpp=vjp(prior_ppo)
 cos=lambda a,b:float(a@b/(a.norm()*b.norm()).clamp_min(1e-12))
 raw=d['raw_observations'][ids].reshape(-1,5,80);teacher=d['teacher_inputs'].reshape(-1,5,80)
 changed=(raw!=teacher).any(-1).any(-1)
 with torch.no_grad():
  truthful_target=teacher_m(d['raw_observations'][ids]).clamp(-3,3)
  conditioned_reproduction=teacher_m(d['teacher_inputs']).clamp(-3,3)
 assert float((conditioned_reproduction-target).abs().max())<6e-5
 correction=(truthful_target-target);correction[:,d['upper_indices']]=0
 truth_derivative=.5*(y[ids].detach()-truthful_target)/difference.numel()
 truth_derivative[:,d['upper_indices']]=0
 gt_der=torch.zeros_like(y);gt_der[ids]=truth_derivative;gt=vjp(gt_der)
 # Exactly reproduce upper-only detached-final-layer auxiliary, not a trunk VJP.
 stride=max(1,(len(y)+255)//256);urows=torch.arange(0,len(y),stride)[:256]
 u=y.detach()[urows][:,d['upper_indices']].requires_grad_(True)
 uloss,parts=posture_loss(u);ud=torch.autograd.grad(uloss,u)[0]
 hidden=m[:6](d['mlp_input']).detach()[urows]
 uw=torch.zeros_like(m[-1].weight);ub=torch.zeros_like(m[-1].bias)
 uw[d['upper_indices']]=ud.T@hidden;ub[d['upper_indices']]=ud.sum(0)
 gu=torch.cat([torch.zeros_like(p).flatten() for p in parameters[:-2]]+[uw.flatten(),ub.flatten()])
 # The stopped-prior hook was registered before this passive moving capture.
 # Reconstruct/subtract it for the whole-batch PPO comparison. Running/prior
 # moving masks already exclude its disjoint stopped rows.
 stopped_ids=(d['raw_observations'][:,-80+6:-80+9].norm(dim=1)<=.1).nonzero().flatten()
 ss=max(1,(len(stopped_ids)+255)//256);stopped_ids=stopped_ids[::ss][:256]
 stopped_der=torch.zeros_like(y)
 if len(stopped_ids):
  with torch.no_grad():
   stopped_difference=y[stopped_ids].detach()-stopped_m(d['raw_observations'][stopped_ids]).clamp(-3,3)
   stopped_difference[:,d['upper_indices']]=0
   stopped_der[stopped_ids]=.5*stopped_difference/stopped_difference.numel()
 gppo=vjp(d['incoming_gradient']-stopped_der)
 upper=[i for i in range(23) if i not in d['upper_indices']]
 commands=[]
 for command in [.15,.35,.65,1.,1.5,2.,2.5,3.]:
  mask=(raw[:,-1,6]-command).abs()<1e-6
  if not mask.any():continue
  rd=difference[mask][:,upper];ed=(y[ids][mask].detach().clamp(-3,3)-target[mask])[:,upper]
  commands.append(dict(command=command,selected=int(mask.sum()),conditioned_fraction=float(changed[mask].float().mean()),
   raw_lower_MSE=float(rd.square().mean()),effective_action_MSE=float(ed.square().mean()),
   same_endpoint_raw_error_fraction=float((plateau[mask].square().sum()/derivative[mask].square().sum().clamp_min(1e-12))),
   teacher_conditioning_lower_effective_RMS=float(correction[mask][:,upper].square().mean().sqrt()),
   truthful_teacher_effective_action_MSE=float((y[ids][mask].detach().clamp(-3,3)-truthful_target[mask])[:,upper].square().mean()),
   latest_head_RMS_deg=float(raw[mask][:,-1,[20,25]].square().mean().sqrt()*180/torch.pi)))
 rows.append(dict(batch=file.name,reproduction_max_error=float((y-d['mean']).abs().max()),
  batch_samples=len(y),selected=len(ids),per_command=commands,
  prior_parameter_norm=float(gp.norm()),plateau_parameter_norm=float(gf.norm()),
  plateau_prior_cosine=cos(gf,gp),running_PPO_norm=float(gr.norm()),prior_PPO_norm=float(gpp.norm()),
  running_teacher_cosine=cos(gr,gp),prior_PPO_teacher_cosine=cos(gpp,gp),
  truthful_teacher_parameter_norm=float(gt.norm()),conditioned_truthful_teacher_cosine=cos(gp,gt),
  upper_auxiliary_parameter_norm=float(gu.norm()),all_PPO_parameter_norm=float(gppo.norm()),
  upper_auxiliary_PPO_cosine=cos(gu,gppo),upper_losses={k:float(v) for k,v in parts.items()},
  stopped_hook_subtracted_for_whole_PPO_comparison=True,
  plateau_teacher_derivative_energy_fraction=float(plateau.square().sum()/derivative.square().sum().clamp_min(1e-12)),
  running_raw_action_clip_fraction=float((y[speed>=3.5-1e-6].abs()>3).float().mean())))
result=dict(completed=True,teacher_checkpoint=cert['checkpoint'],teacher_sha256=cert['checkpoint_sha256'],batches=rows,scope='Four genuine PPO minibatches at 4096 environments. Passive labels and pre-auxiliary VJPs before clipping/Adam; exact upper-only final-row auxiliary. Truthful teacher is a CPU counterfactual, absent from production. Finite-state evidence is not a causal repair or proof of globally opposed gradients.')
(OUT/'current_gradient_analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
