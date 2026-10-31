"""Gradient-route, reuse and fixed-state invariants, not a gait certificate."""
import copy,json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from final_row_route import FinalRowRoute,VERSION
torch.set_num_threads(4)
OUT=Path(__file__).resolve().parent
records=[]
HEAD=[11,16]
def check(condition,name):
    assert condition,name
    records.append(name)
def make():return nn.Sequential(nn.Linear(400,32),nn.ELU(),nn.Linear(32,16),nn.ELU(),nn.Linear(16,23))
def fixture(chosen_kind,n_graphs=1,backward_reuses=1):
    torch.manual_seed(311+n_graphs)
    actual=make()
    expected=copy.deepcopy(actual)
    route=FinalRowRoute(SimpleNamespace(mlp=actual),HEAD)
    actual_loss=0
    expected_loss=0
    state=[]
    for k in range(n_graphs):
        x=torch.randn(19,400)
        ids=torch.tensor([1,3,7,9,12,16])
        chosen=torch.ones(6,dtype=torch.bool) if chosen_kind=='all' else torch.zeros(6,dtype=torch.bool)
        if chosen_kind=='mixed':chosen[::2]=True
        d=torch.randn(6,23)*.002
        d[:,HEAD]=0
        a=actual(x)
        e=expected(x)
        route.attach(a,ids,d,chosen)
        # PPO-like nonlinear derivative remains present, including head outputs.
        actual_loss+=a.square().mean()+.003*a.sin().sum()
        ordinary=d.clone();ordinary[chosen]=0
        restricted=d.clone();restricted[~chosen]=0
        features=expected[:-1](x).detach()
        last=nn.functional.linear(features,expected[-1].weight,expected[-1].bias)
        expected_loss+=e.square().mean()+.003*e.sin().sum()+(e[ids]*ordinary).sum()+(last[ids]*restricted).sum()
        state.append((x,ids,d,chosen))
    for reuse in range(backward_reuses):
        actual.zero_grad();expected.zero_grad()
        actual_loss.backward(retain_graph=reuse+1<backward_reuses)
        expected_loss.backward(retain_graph=reuse+1<backward_reuses)
        check(all(torch.allclose(a.grad,e.grad,atol=3e-7,rtol=2e-5) for a,e in zip(actual.parameters(),expected.parameters())),
              f'{chosen_kind}_{n_graphs}_graphs_backward{reuse}_matches_explicit_detached_objective')
        check(not route.pending,f'{chosen_kind}_{n_graphs}_graphs_backward{reuse}_no_pending_gradient')
    if backward_reuses==1:
        audit=route.read_audit()
        check(audit['batches']==n_graphs,f'{chosen_kind}_{n_graphs}_graphs_all_weight_and_bias_contributions_consumed')
    if chosen_kind=='none':
        # No candidate states means full equivalence to the original mean hook.
        other=copy.deepcopy(actual)
        other.zero_grad()
        x,ids,d,_=state[0]
        y=other(x)
        (y.square().mean()+.003*y.sin().sum()+(y[ids]*d).sum()).backward()
        check(all(torch.allclose(a.grad,b.grad,atol=3e-7,rtol=2e-5) for a,b in zip(actual.parameters(),other.parameters())),
              'no_eligible_state_matches_original_teacher_path')
    for h in route.handles:h.remove()
for kind in ['none','mixed','all']:fixture(kind)
fixture('mixed',n_graphs=2)
# Repeated backwards must not retain stale pending values; audit counts describe
# ordinary one-backward PPO and are not invoked for this deliberately reused graph.
fixture('mixed',backward_reuses=2)
torch.manual_seed(419)
actor=make();route=FinalRowRoute(SimpleNamespace(mlp=actor),HEAD)
x=torch.randn(17,400);y=actor(x);before=y.detach().clone()
ids=torch.arange(len(x));d=torch.randn(17,23)*.01;d[:,HEAD]=0
route.attach(y,ids,d,torch.ones(17,dtype=torch.bool))
(y*0).sum().backward()
check(all(not p.grad.any() for p in list(actor.parameters())[:-2]),'selected_target_has_zero_hidden_feature_derivative')
check(not actor[-1].weight.grad[HEAD].any() and not actor[-1].bias.grad[HEAD].any(),'selected_target_has_zero_direct_head_derivative')
with torch.no_grad():
    for p in actor.parameters():p.add_(p.grad,alpha=-1e-4)
    after=actor(x)
check(torch.equal(after[:,HEAD],before[:,HEAD]),'isolated_SGD_target_step_preserves_fixed_state_head_means_bitwise')
check(not torch.equal(after,before),'isolated_SGD_step_changes_selected_nonhead_means')
route.read_audit()
for h in route.handles:h.remove()
result=dict(passed=True,version=VERSION,checks=records,count=len(records),production_change=False,
    scope='Exact local gradient-route and lifecycle fixtures. No physical gait, optimizer/global-clipping retention, head trajectory or natural-motion qualification.')
(OUT/'final_row_CPU_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
