"""CPU local target-gradient counterfactual; no Adam or closed-loop causal claim."""
import copy,json
from pathlib import Path
import torch
from torch import nn
from command_local_teacher_target import eligible,target_derivative
OUT=Path(__file__).resolve().parent;SECOND=OUT/'secondary_analysis'
torch.set_num_threads(4)
read=lambda p:json.loads(Path(p).read_text())
def model():return nn.Sequential(nn.Linear(400,512),nn.ELU(),nn.Linear(512,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,23))
cert=read(OUT/'neutral_teacher_certificate.json');saved=torch.load(cert['checkpoint'],map_location='cpu',weights_only=False)
teacher=model();teacher.load_state_dict({k.removeprefix('mlp.'):v for k,v in saved['actor_state_dict'].items() if k.startswith('mlp.')});teacher.requires_grad_(False)
head_columns=read(SECOND/'control_source/physical_v29/training/teacher_head_context.json')['columns'][:2]
records=[]
for p in sorted(SECOND.glob('current_gradient_*.pt')):
    d=torch.load(p,map_location='cpu',weights_only=False);m=model();m.load_state_dict(d['state'])
    ids=d['ids'];raw=d['raw_observations'][ids];selected=eligible(raw,head_columns);upper=d['upper_indices'];head=upper[:2]
    with torch.no_grad():neutral=teacher(raw).clamp(-3,3)
    _,_,derivative,_=target_derivative(d['mean'][ids],d['teacher_raw_targets'].clamp(-3,3),neutral,selected,upper,head)
    y=m(d['mlp_input']);original=y.detach().clone();features=m[:-1](d['mlp_input']).detach()
    full=torch.zeros_like(y);full[ids]=derivative*selected[:,None]
    assert not full[:,head].any()
    parameters=list(m.parameters());shared=torch.autograd.grad(y,parameters,grad_outputs=full)
    last_only=[torch.zeros_like(g) for g in shared]
    last_only[-2]=full.T@features;last_only[-1]=full.sum(0)
    assert all(not g.any() for g in last_only[:-2])
    assert torch.allclose(shared[-2],last_only[-2],atol=2e-6,rtol=1e-5)
    assert not shared[-2][head].any() and not shared[-1][head].any()
    responses={}
    for label,gradients in [('shared_trunk',shared),('last21_rows_only',last_only)]:
        probe=copy.deepcopy(m)
        with torch.no_grad():
            for parameter,g in zip(probe.parameters(),gradients):parameter.add_(g,alpha=-1e-4)
            changed=probe(d['mlp_input'])-original
        responses[label]=dict(head_raw_mean_change_RMS=float(changed[:,head].square().mean().sqrt()),
            head_raw_mean_max_absolute_change=float(changed[:,head].abs().max()),
            all_raw_mean_change_RMS=float(changed.square().mean().sqrt()),
            non_head_final_rows_gradient_norm=float(torch.cat([gradients[-2].flatten(),gradients[-1].flatten()]).norm()),
            trunk_gradient_norm=float(torch.cat([g.flatten() for g in gradients[:-2]]).norm()))
    assert responses['last21_rows_only']['head_raw_mean_max_absolute_change']<=1e-7
    records.append(dict(batch=p.name,selected_truthful_states=int(selected.sum()),head_rows=head,
        same_final_layer_teacher_derivative=True,zero_direct_head_gradient=True,responses=responses))
assert len(records)==4
result=dict(completed=True,records=records,probe_step='One1e-4 SGD target-only diagnostic on cloned captured weights,not actualAdam/PPO training',
    original_checkpoint=read(SECOND/'diagnostic_pause.json')['checkpoint'],
    production_change=False,new_learning_variant_tested=False,
    scope='Direct head-row exclusion does not mathematically insulate the head output from a shared-trunk target gradient. The row-only counterfactual isolates that route on fixed recorded states. Physical state changes,PPO,global clipping,Adam and long-term interactions are not isolated; this does not prove the measured128 head regression is caused solely by this route or certify a safe repair.')
(OUT/'shared_target_head_effect.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
