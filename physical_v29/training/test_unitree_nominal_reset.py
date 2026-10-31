"""Regress the reset/reference alias that silently moved the nominal pose.

Executes the actual _reference method from frozen and corrected source, then
the inherited reset's indexed assignment. This needs no second Isaac worker.
"""
import ast,json,warnings
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
HERE=Path(__file__).parent
MOTION=np.load(HERE/'assets/motion_reference.npz')

def reference(path):
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LocomotionEnv')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_reference')
    code=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
    scope=dict(torch=torch,MOTION=MOTION)
    exec(compile(code,str(path),'exec'),scope)
    return scope['_reference']

def probe(path,count=512):
    f=reference(path)
    nominal=torch.as_tensor(MOTION['stand']).clone()
    e=SimpleNamespace(stand=nominal.clone(),num_envs=16,device='cpu',reference_height=torch.zeros(16),
        foot_clips=torch.zeros(7,120,2,3))
    rng=torch.Generator().manual_seed(98231)
    maximum=0.;private_rows=True
    changed_unreset_rows=False
    with warnings.catch_warnings(record=True) as warning:
        warnings.simplefilter('always')
        for tick in range(count):
            f(e)
            private_rows &= e.reference.stride(0)>0 and e.reference.data_ptr()!=e.stand.data_ptr()
            # Actual reset distribution and assignment; a single randomized
            # row suffices to demonstrate the write-through defect.
            ids=torch.tensor([tick%16])
            untouched=torch.tensor([i for i in range(16) if i not in ids.tolist()])
            before=e.reference[untouched].clone()
            q=e.stand.expand(len(ids),-1).clone()+torch.randn((len(ids),23),generator=rng)*.012
            e.reference[ids]=q
            changed_unreset_rows |= not torch.equal(e.reference[untouched],before)
            maximum=max(maximum,float((e.stand-nominal).abs().max()))
            assert torch.equal(e.reference[ids],q)
    return dict(maximum_nominal_drift_rad=maximum,private_rows=private_rows,changed_unreset_rows=changed_unreset_rows,
        warnings=sorted(set(str(w.message) for w in warning)),resets=count)

old=probe(HERE/'runs/locomotion_20261002_r21_unitree_lifetime/source/physical_v29/training/environment_unitree.py')
new=probe(HERE/'environment_unitree.py')
assert old['maximum_nominal_drift_rad']>.01 and not old['private_rows'],old
assert new['maximum_nominal_drift_rad']==0. and new['private_rows'],new
assert old['changed_unreset_rows'] and not new['changed_unreset_rows']
assert not new['warnings'],new
protected=['environment.py','evaluation.py','evaluation_steps.py','walking_rewards.py','reference_geometry.py','mass_estimate.json',
    'assets/robot_training.urdf','assets/motion_reference.npz','unitree_reward_port.py']
source=HERE/'runs/locomotion_20261002_r21_unitree_lifetime/source/physical_v29'
for name in protected:assert (source/'training'/name).read_bytes()==(HERE/name).read_bytes(),name
for name in ['hardware_actuators.py','cad_differential.py','assets/actuator_limits.json','assets/differential_geometry.json','assets/export_metadata.json']:
    assert (source/name).read_bytes()==(HERE.parent/name).read_bytes(),name
result=dict(passed=True,frozen_r21=old,repaired=new,
    protected_sources_unchanged=protected+['physical_v29/'+n for n in ['hardware_actuators.py','cad_differential.py','assets/actuator_limits.json','assets/differential_geometry.json','assets/export_metadata.json']],
    scope='Actual source method and inherited reset assignment; proves alias repair, not learned balance or a CUDA benchmark')
out=HERE/'diagnostics/unitree_termination_20261002/nominal_reset_regression.json'
out.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
