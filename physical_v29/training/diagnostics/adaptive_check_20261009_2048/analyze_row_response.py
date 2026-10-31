"""Actual minibatch, isolated target-route counterfactual on unchanged CPU clones."""
import copy,json
from pathlib import Path
import torch
from torch import nn
from command_local_teacher_target import eligible,target_derivative
OUT=Path(__file__).resolve().parent
torch.set_num_threads(4)
read=lambda p:json.loads(Path(p).read_text())
def model():return nn.Sequential(nn.Linear(400,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,23))
def flat(gs):return torch.cat([g.flatten() for g in gs])
cert=read(OUT/'neutral_teacher_certificate.json')
teacher=model();saved=torch.load(cert['checkpoint'],map_location='cpu',weights_only=False)
teacher.load_state_dict({k.removeprefix('mlp.'):v for k,v in saved['actor_state_dict'].items() if k.startswith('mlp.')});teacher.requires_grad_(False)
head_columns=read(OUT/'control_source/physical_v29/training/teacher_head_context.json')['columns'][:2]
records=[]
for p in sorted(OUT.glob('current_gradient_*.pt')):
 d=torch.load(p,map_location='cpu',weights_only=False);m=model();m.load_state_dict(d['state'])
 ids=d['ids'];raw=d['raw_observations'];chosen=eligible(raw[ids],head_columns);upper=d['upper_indices'];head=upper[:2]
 with torch.no_grad():neutral=teacher(raw[ids]).clamp(-3,3)
 _,_,derivative,_=target_derivative(d['mean'][ids],d['teacher_raw_targets'].clamp(-3,3),neutral,chosen,upper,head)
 y=m(d['mlp_input']);original=y.detach().clone();features=m[:-1](d['mlp_input']).detach();full=torch.zeros_like(y);full[ids]=derivative*chosen[:,None]
 assert not full[:,head].any()
 parameters=list(m.parameters());shared=torch.autograd.grad(y,parameters,grad_outputs=full,retain_graph=True)
 row=[torch.zeros_like(g) for g in shared];row[-2]=full.T@features;row[-1]=full.sum(0)
 assert all(not g.any() for g in row[:-2]) and not row[-2][head].any() and not row[-1][head].any()
 assert torch.allclose(shared[-2],row[-2],atol=2e-6,rtol=1e-5)
 incoming=d['incoming_PPO_gradient']
 gradients={}
 if incoming is not None:
  pg=torch.autograd.grad(y,parameters,grad_outputs=incoming,retain_graph=True)
  gradients['PPO_or_captured_incoming_scope']=dict(key='incoming_PPO_gradient',norm=float(flat(pg).norm()),
    shared_cosine=float(torch.nn.functional.cosine_similarity(flat(pg)[None],flat(shared)[None])),row_cosine=float(torch.nn.functional.cosine_similarity(flat(pg)[None],flat(row)[None])))
 responses={}
 for label,gs in [('shared',shared),('rows',row)]:
  probe=copy.deepcopy(m)
  with torch.no_grad():
   for parameter,g in zip(probe.parameters(),gs):parameter.add_(g,alpha=-1e-4)
   after=probe(d['mlp_input']);delta=after-original;clamped=after.clamp(-3,3)-original.clamp(-3,3)
  bins={};speed=raw[:,-80+6]
  masks={str(c):(speed-c).abs()<1e-6 for c in [0.,.15,.35,.65,1.,1.5,2.,2.5,3.]}
  masks['frontier_3.25_3.5']=(speed>=3.25)&(speed<=3.5+1e-6)
  body=[i for i in range(23) if i not in upper]
  for name,mask in masks.items():
   if not mask.any():continue
   bins[name]=dict(samples=int(mask.sum()),all_raw_change_RMS=float(delta[mask].square().mean().sqrt()),
      head_raw_change_RMS=float(delta[mask][:,head].square().mean().sqrt()),body_raw_change_RMS=float(delta[mask][:,body].square().mean().sqrt()),
      clamped_endpoint_change_RMS=float(clamped[mask].square().mean().sqrt()))
  responses[label]=dict(head_raw_max_change=float(delta[:,head].abs().max()),head_raw_RMS_change=float(delta[:,head].square().mean().sqrt()),
      trunk_gradient_norm=float(flat(gs[:-2]).norm()),final_gradient_norm=float(flat(gs[-2:]).norm()),per_command=bins)
 assert responses['rows']['head_raw_max_change']==0
 records.append(dict(batch=p.name,eligible_states=int(chosen.sum()),same_final_derivative=True,head_rows=head,responses=responses,gradient_comparison=gradients))
assert len(records)==4 and sum(r['eligible_states'] for r in records)>0
result=dict(completed=True,passed_numerical_and_scope_checks=True,records=records,checkpoint=read(OUT/'diagnostic_pause.json')['checkpoint'],
 scope='One isolated 1e-4 SGD target step on actual recorded minibatches. Clamped endpoints are action-unit endpoints, not joint/slew/CAD applied targets. No Adam/global clipping/PPO/physical isolation or guaranteed off-command retention.',
 production_change=False,learning_comparison_required=True)
(OUT/'current_row_response.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
