"""Isaac Lab explicit actuation using September 30 assumptions and CAD ratios.

Import after AppLauncher has started Isaac. Use make_actuators() in every task;
URDF effort boxes alone cannot enforce a two-motor differential envelope.
"""
import json
from pathlib import Path
import torch
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass
from cad_differential import CadDifferential
from owned_cuda_graph import OwnedCudaCache
from isaaclab.utils.types import ArticulationActions

HERE=Path(__file__).resolve().parent


class HardwareActuator(IdealPDActuator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.records=json.loads((HERE/'assets/actuator_limits.json').read_text())
        geometry=json.loads((HERE/'assets/differential_geometry.json').read_text())
        self.pairs=[]
        for key,spec in geometry.items():
            indexes=[self.joint_names.index(n) for n in spec['joint_names']]
            self.pairs.append((key,indexes,CadDifferential(spec)))
        self.last_motor_states={}
        self.invalid_transmission=None
        self._hardware_graphs=OwnedCudaCache(2)
        self._peak=torch.tensor([self.records[n]['effort_Nm'] for n in self.joint_names],device=self.stiffness.device)
        self._speed=torch.tensor([self.records[n]['velocity_rad_s'] for n in self.joint_names],device=self.stiffness.device)
        self._linear=torch.tensor([self.records[n].get('speed_model','').startswith('linear') for n in self.joint_names],device=self.stiffness.device)

    def _hardware_kernel(self, joint_pos, joint_vel, stiffness, damping, *targets, mode):
        values = iter(targets)
        action = ArticulationActions(**{name:next(values) if present else None
            for name,present in zip(('joint_positions','joint_velocities','joint_efforts'),mode)})
        old_stiffness,old_damping = self.stiffness,self.damping
        old_modes = [model._graph_passthrough for _,_,model in self.pairs]
        try:
            self.stiffness,self.damping = stiffness,damping
            for _,_,model in self.pairs:model._graph_passthrough=True
            self._compute_eager(action,joint_pos,joint_vel)
            return self.computed_effort,self.applied_effort,self.invalid_transmission,self.last_motor_states.copy()
        finally:
            self.stiffness,self.damping = old_stiffness,old_damping
            for (_,_,model),old in zip(self.pairs,old_modes):model._graph_passthrough=old

    def compute(self,control_action,joint_pos,joint_vel):
        targets = (control_action.joint_positions,control_action.joint_velocities,control_action.joint_efforts)
        mode = tuple(x is not None for x in targets)
        inputs = (joint_pos,joint_vel,self.stiffness,self.damping)+tuple(x for x in targets if x is not None)
        outputs = self._hardware_graphs.call(lambda *values:self._hardware_kernel(*values,mode=mode),*inputs,mode=mode)
        self.computed_effort,self.applied_effort,self.invalid_transmission,self.last_motor_states = outputs
        control_action.joint_efforts=self.applied_effort
        control_action.joint_positions=None
        control_action.joint_velocities=None
        return control_action

    def _compute_eager(self,control_action,joint_pos,joint_vel):
        ep=torch.zeros_like(joint_pos) if control_action.joint_positions is None else control_action.joint_positions-joint_pos
        ev=-joint_vel if control_action.joint_velocities is None else control_action.joint_velocities-joint_vel
        ff=torch.zeros_like(joint_pos) if control_action.joint_efforts is None else control_action.joint_efforts
        self.computed_effort=self.stiffness*ep+self.damping*ev+ff
        applied=torch.maximum(torch.minimum(self.computed_effort,self._peak),-self._peak)
        upper=torch.maximum(torch.minimum(self._peak*(1-joint_vel/self._speed),self._peak),-self._peak)
        lower=torch.maximum(torch.minimum(self._peak*(-1-joint_vel/self._speed),self._peak),-self._peak)
        linear_applied=torch.maximum(torch.minimum(applied,upper),lower)
        capped=torch.where((joint_vel.abs()>self._speed+1e-6)&(applied*joint_vel>0),0.,applied)
        applied=torch.where(self._linear,linear_applied,capped)
        invalid=[]
        for key,ids,model in self.pairs:
            record=self.records[self.joint_names[ids[0]]]
            effort,state=model.limit(joint_pos[:,ids],joint_vel[:,ids],self.computed_effort[:,ids],record['motor_peak_Nm'],record['motor_speed_cap_RPM']*2*torch.pi/60,efficiency=record['transmission_efficiency'])
            applied[:,ids]=effort
            self.last_motor_states[key]=state
            invalid.append(~state['valid'])
        self.invalid_transmission=torch.stack(invalid,1).any(1)
        self.applied_effort=applied
        control_action.joint_efforts=applied
        control_action.joint_positions=None
        control_action.joint_velocities=None
        return control_action


@configclass
class HardwareActuatorCfg(IdealPDActuatorCfg):
    class_type:type=HardwareActuator


def make_actuators(stiffness=35.,damping=4.,armature=0.):
    records=json.loads((HERE/'assets/actuator_limits.json').read_text())
    return {'cad_hardware':HardwareActuatorCfg(
        joint_names_expr=['.*'],
        effort_limit={n:r.get('URDF_bounding_effort_Nm',r['effort_Nm']) for n,r in records.items()},
        effort_limit_sim={n:r.get('URDF_bounding_effort_Nm',r['effort_Nm']) for n,r in records.items()},
        velocity_limit={n:r.get('URDF_bounding_velocity_rad_s',r['velocity_rad_s']) for n,r in records.items()},
        velocity_limit_sim={n:r.get('URDF_bounding_velocity_rad_s',r['velocity_rad_s']) for n,r in records.items()},
        stiffness=stiffness,damping=damping,armature=armature)}
