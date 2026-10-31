"""Physical actuator implementation parity; import only after AppLauncher."""
import importlib.util
import itertools
import json
from pathlib import Path
import torch
from isaaclab.utils.types import ArticulationActions


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _clone(value):
    if isinstance(value, torch.Tensor):return value.clone()
    if isinstance(value, dict):return {k:_clone(v) for k,v in value.items()}
    return tuple(_clone(v) for v in value)


def _equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a,b), ('actuator mismatch',float((a.float()-b.float()).abs().max()))
    elif isinstance(a, dict):
        assert set(a)==set(b)
        for key in a:_equal(a[key],b[key])
    else:
        for x,y in zip(a,b):_equal(x,y)


def check(source, raw, output):
    here = Path(__file__).resolve().parent
    reference = here / 'baseline/source/physical_v29'
    old_hardware = _module('reference_hardware',reference/'hardware_actuators.py')
    old_cad = _module('reference_cad',reference/'cad_differential.py')
    import hardware_actuators
    from cad_differential import CadDifferential
    from owned_cuda_graph import OwnedCudaCache
    spec = json.loads((source/'assets/differential_geometry.json').read_text())
    cases = []

    def construct(cls, cad, count, dtype):
        obj = cls.__new__(cls)
        obj._joint_names = list(raw.robot.joint_names)
        obj.stiffness=raw.actuator.stiffness[:1].to(dtype).repeat(count,1)
        obj.damping=raw.actuator.damping[:1].to(dtype).repeat(count,1)
        obj._peak=raw.actuator._peak.to(dtype).clone()
        obj._speed=raw.actuator._speed.to(dtype).clone()
        obj._linear=raw.actuator._linear.clone()
        obj.records=raw.actuator.records
        obj.last_motor_states={};obj.invalid_transmission=None
        obj.pairs=[(key,[obj.joint_names.index(n) for n in s['joint_names']],cad(s)) for key,s in spec.items()]
        obj._hardware_graphs=OwnedCudaCache(2)
        obj._pair_indices={key:torch.tensor(ids,device=obj.stiffness.device) for key,ids,_ in obj.pairs}
        if hasattr(cls,'project_valid_targets'):
            obj._projection_graphs=OwnedCudaCache(2);obj._validity_graphs=OwnedCudaCache(2)
        return obj

    for dtype in [torch.float32,torch.float64]:
        a = construct(old_hardware.HardwareActuator,old_cad.CadDifferential,72,dtype)
        b = construct(hardware_actuators.HardwareActuator,CadDifferential,72,dtype)
        q=raw.stand.to(dtype).repeat(72,1)
        variation=torch.linspace(-1.8,1.8,72,device=q.device,dtype=dtype)
        for _,ids,_ in b.pairs:
            q[:,ids[0]]=variation;q[:,ids[1]]=variation.flip(0)*.5
        qd=torch.linspace(-160,160,q.numel(),device=q.device,dtype=dtype).reshape_as(q)
        target=q+torch.linspace(-3,3,q.numel(),device=q.device,dtype=dtype).reshape_as(q)
        ff=qd*2
        with torch.inference_mode():
            for mode in itertools.product([False,True],repeat=3):
                action_args=dict(zip(('joint_positions','joint_velocities','joint_efforts'),
                                     [v if flag else None for v,flag in zip((target,-qd,ff),mode)]))
                ea, va = ArticulationActions(**action_args),ArticulationActions(**action_args)
                a.compute(ea,q,qd);b.compute(va,q,qd)
                _equal((a.computed_effort,a.applied_effort,a.invalid_transmission,a.last_motor_states),
                       (b.computed_effort,b.applied_effort,b.invalid_transmission,b.last_motor_states))
                prior=_clone((b.computed_effort,b.applied_effort,b.invalid_transmission,b.last_motor_states))
                prior_refs=(b.computed_effort,b.applied_effort,b.invalid_transmission,b.last_motor_states)
                a.stiffness.mul_(1.01);b.stiffness.mul_(1.01)
                a.damping.mul_(.99);b.damping.mul_(.99)
                qa=q.roll(1,0).clone();velocity=qd.flip(0).clone()
                a.compute(ArticulationActions(**action_args),qa,velocity)
                b.compute(ArticulationActions(**action_args),qa,velocity)
                _equal((a.computed_effort,a.applied_effort,a.invalid_transmission,a.last_motor_states),
                       (b.computed_effort,b.applied_effort,b.invalid_transmission,b.last_motor_states))
                _equal(prior,prior_refs)
                assert len(b._hardware_graphs)<=2
                assert va.joint_positions is None and va.joint_velocities is None
                _equal(va.joint_efforts,prior[1])
                cases.append(dict(dtype=str(dtype),mode=list(mode),bitwise=True,ownership=True,changed_gains=True))
            if hasattr(b,'project_valid_targets'):
                previous=q*0
                expected=q.clone()
                for _,ids,model in a.pairs:
                    _,_,valid,_=model.state(expected[:,ids])
                    expected[:,ids]=torch.where(valid[:,None],expected[:,ids],previous[:,ids])
                actual=b.project_valid_targets(q,previous)
                _equal(expected,actual)
                saved=actual.clone()
                b.project_valid_targets(q.roll(2,0),previous+1)
                _equal(saved,actual)
                invalid=torch.zeros(72,dtype=torch.bool,device=q.device)
                for _,ids,model in a.pairs:invalid|=~model.state(q[:,ids])[2]
                actual_invalid=b.invalid_at_pose(q)
                _equal(invalid,actual_invalid)
                saved_invalid=actual_invalid.clone()
                b.invalid_at_pose(q*0)
                _equal(saved_invalid,actual_invalid)
    result=dict(passed=True,cases=cases,projection_and_current_pose_validity=hasattr(b,'project_valid_targets'),
                scope='Detached exact arithmetic, all optional PD/control modes, changed inputs/gains and output ownership; no physics change')
    output.write_text(json.dumps(result,indent=2))
    return result
