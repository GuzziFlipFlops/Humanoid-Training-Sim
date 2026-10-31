"""CPU weak-reference test without relying on cyclic garbage collection."""
import gc,json,math,sys,weakref
from pathlib import Path
from types import SimpleNamespace
import torch
from rsl_rl.modules.mlp import MLP
from sequence_shoulder_refinement import SequenceShoulderRefinement
expected_retained='--expect-retained' in sys.argv
OUT=Path(__file__).resolve().parent
was_enabled=gc.isenabled();gc.disable()
try:
    actor=SimpleNamespace(mlp=MLP(400,23,[128]),obs_normalizer=torch.nn.Identity())
    names=['left_shoulder_pitch','right_shoulder_pitch']+[str(i) for i in range(21)]
    helper=SequenceShoulderRefinement(actor,names,apply=False)
    for tick in range(33):
        angle=(tick+.25)*2*math.pi/16
        obs=torch.zeros(2,5,80);obs[:,:,6]=.35;obs[:,:,78]=math.sin(angle);obs[:,:,79]=math.cos(angle)
        helper.cache.record(obs.flatten(1))
    held=[];hook_variables=[]
    for i in range(12):
        out=actor.mlp(obs.flatten(1));reference=weakref.ref(out)
        hook_variables.append([list(fn.__code__.co_freevars) for fn in out._backward_hooks.values()])
        out.square().mean().backward();actor.mlp.zero_grad(set_to_none=True)
        del out
        held.append(reference() is not None)
    assert all(held) if expected_retained else not any(held),held
    result=dict(passed=True,expected_retained=expected_retained,output_retained_without_GC_by_update=held,
        callback_free_variables=hook_variables,gc_disabled_during_fixture=True,actual_RSL_MLP=True,
        diagnostic_not_production=True)
    (OUT/('graph_lifetime_before.json' if expected_retained else 'graph_lifetime_after.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,expected_retained=expected_retained,retained=sum(held))))
    helper.close()
finally:
    if was_enabled:gc.enable()
    gc.collect()
