"""Create a new immutable experiment source tree; never edit the active frozen run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

here = Path(__file__).resolve().parent


def freeze(destination):
    files = {str(p.relative_to(destination)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in destination.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.suffix != '.pyc'}
    (destination.parents[1] / 'manifest.json').write_text(json.dumps(files, indent=2))


def first(destination):
    path = destination / 'cad_differential.py'
    original = path.read_text()
    prefix = original[:original.index('    def state(self, q):')]
    prefix = prefix.replace('        self._tensors = {}', '        self._tensors = {}\n        self._prepared = {}')
    suffix = original[original.index('    def limit(self, q, qd, desired,'):]
    state = '''    def _geometry(self, q):
        # Bounded by actual device/dtype combinations, never by transient poses.
        key = (str(q.device), q.dtype)
        if key not in self._prepared:
            def t(v):
                tensor_key = (str(q.device), q.dtype, tuple(v))
                if tensor_key not in self._tensors:
                    self._tensors[tensor_key] = torch.as_tensor(v, device=q.device, dtype=q.dtype)
                return self._tensors[tensor_key]
            p1,p2,a1,a2 = [t(self.spec[k]) for k in ('pitch_origin_mm','roll_origin_mm','pitch_axis','roll_axis')]
            motors = []
            for motor in self.spec['motors']:
                m,a,v,b = [t(motor[k]) for k in ('center_mm','axis','crank_vector_mm','bottom_eye_mm')]
                parallel = a*(a*v).sum()
                vp = v-parallel
                cross = torch.linalg.cross(a,v)
                motors.append((motor,m,a,v,b,parallel,vp,cross,vp.square().sum()))
            self._prepared[key] = (p1,p2,a1,a2,p2-p1,motors)
        return self._prepared[key]

    def state(self, q):
        """Exact original equations, sharing immutable geometry and trigonometry."""
        p1,p2,a1,a2,delta,motors = self._geometry(q)
        p,r = q[:,0],q[:,1]
        cp,sp = torch.cos(p)[...,None],torch.sin(p)[...,None]
        cr,sr = torch.cos(r)[...,None],torch.sin(r)[...,None]
        def rotation(v,axis,c,s):
            v,axis = torch.broadcast_tensors(v,axis)
            return v*c + torch.linalg.cross(axis,v,dim=-1)*s + axis*(axis*v).sum(-1,keepdim=True)*(1-c)
        def transform(point):
            return p1 + rotation(delta+rotation(point-p2,a2,cr,sr),a1,cp,sp)
        if self.spec['motor_on_moving_link']:
            cnp,snp = torch.cos(-p)[...,None],torch.sin(-p)[...,None]
            cnr,snr = torch.cos(-r)[...,None],torch.sin(-r)[...,None]
            def inverse(point):
                return p2 + rotation(p1+rotation(point-p1,a1,cnp,snp)-p2,a2,cnr,snr)
        def vector_transform(v):
            return rotation(rotation(v,a2,cr,sr),a1,cp,sp)
        a2world = rotation(a2,a1,cp,sp)
        p2world = p1+rotation(delta,a1,cp,sp)
        angles,rows,valids,errors = [],[],[],[]
        for motor,m,a,v,b,parallel,vp,cross,vp2 in motors:
            moving_motor = self.spec['motor_on_moving_link']
            bottom = b.expand(q.shape[0],3) if moving_motor else transform(b)
            ref = inverse(b) if moving_motor else bottom
            d = m+parallel-ref
            ca,sa = (d*vp).sum(-1),(d*cross).sum(-1)
            amp = torch.sqrt(ca.square()+sa.square())
            value = (motor['rod_length_mm']**2-d.square().sum(-1)-vp2)/(2*amp.clamp_min(1e-9))
            phi = torch.atan2(sa,ca)
            ac = torch.acos(value.clamp(-1,1))
            candidates = torch.stack((phi+ac,phi-ac),-1)
            candidates = torch.atan2(torch.sin(candidates),torch.cos(candidates))
            theta = candidates.gather(1,candidates.abs().argmin(1,keepdim=True)).squeeze(1)
            ct,st = torch.cos(theta)[...,None],torch.sin(theta)[...,None]
            tip_local = m+rotation(v,a,ct,st)
            tip = transform(tip_local) if moving_motor else tip_local
            tangent = rotation(cross,a,ct,st)
            if moving_motor:
                tangent = vector_transform(tangent)
                derivatives = torch.stack((torch.linalg.cross(a1.expand_as(tip),tip-p1),torch.linalg.cross(a2world,tip-p2world)),dim=-1)
                numerator_sign = -1
            else:
                derivatives = torch.stack((torch.linalg.cross(a1.expand_as(bottom),bottom-p1),torch.linalg.cross(a2world,bottom-p2world)),dim=-1)
                numerator_sign = 1
            rod = tip-bottom
            denom = (rod*tangent).sum(-1)
            safe = torch.where(denom.abs()>1e-7,denom,torch.ones_like(denom))
            row = numerator_sign*(rod[...,None]*derivatives).sum(1)/safe[:,None]
            angles.append(theta); rows.append(row)
            lo,hi = motor['motor_limits_rad']
            valids.append((value.abs()<=1+1e-8)&(amp>1e-7)&(denom.abs()>1e-7)&(theta>=lo-1e-8)&(theta<=hi+1e-8))
            errors.append((torch.linalg.vector_norm(rod,dim=-1)-motor['rod_length_mm']).abs())
        theta,J = torch.stack(angles,1),torch.stack(rows,1)
        det = J[:,0,0]*J[:,1,1]-J[:,0,1]*J[:,1,0]
        valid = torch.stack(valids,1).all(1)&(det.abs()>1e-5)
        return theta,J,valid,torch.stack(errors,1)

'''
    path.write_text(prefix + state + suffix)


def second(destination):
    shutil.copyfile(here / 'owned_cuda_graph.py', destination / 'owned_cuda_graph.py')
    path = destination / 'cad_differential.py'
    text = path.read_text()
    text = text.replace('import torch\n', 'import torch\nfrom owned_cuda_graph import OwnedCudaCache\n', 1)
    text = text.replace('        self._prepared = {}',
                        '        self._prepared = {}\n        self._state_graphs = OwnedCudaCache(2)\n        self._graph_passthrough = False', 1)
    text = text.replace('    def state(self, q):', '''    def state(self, q):
        if self._graph_passthrough:
            return self._state_eager(q)
        return self._state_graphs.call(self._state_eager, q)

    def _state_eager(self, q):''', 1)
    path.write_text(text)


def third(destination):
    path = destination / 'hardware_actuators.py'
    text = path.read_text()
    text = text.replace('from cad_differential import CadDifferential',
                        'from cad_differential import CadDifferential\nfrom owned_cuda_graph import OwnedCudaCache\nfrom isaaclab.utils.types import ArticulationActions', 1)
    text = text.replace('        self.invalid_transmission=None',
                        '        self.invalid_transmission=None\n        self._hardware_graphs=OwnedCudaCache(2)', 1)
    text = text.replace('    def compute(self,control_action,joint_pos,joint_vel):', '''    def _hardware_kernel(self, joint_pos, joint_vel, stiffness, damping, *targets, mode):
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

    def _compute_eager(self,control_action,joint_pos,joint_vel):''', 1)
    path.write_text(text)


def fourth(destination):
    path=destination/'hardware_actuators.py'
    text=path.read_text()
    text=text.replace('        self._hardware_graphs=OwnedCudaCache(2)',
                      '        self._hardware_graphs=OwnedCudaCache(2)\n        self._pair_indices={key:torch.tensor(ids,device=self.stiffness.device) for key,ids,_ in self.pairs}',1)
    text=text.replace("            record=self.records[self.joint_names[ids[0]]]\n            effort,state=",
                      "            record=self.records[self.joint_names[ids[0]]]\n            ids=self._pair_indices[key] if joint_pos.is_cuda else ids\n            effort,state=",1)
    path.write_text(text)


def fifth(destination):
    path=destination/'hardware_actuators.py'
    text=path.read_text()
    text=text.replace('        self._hardware_graphs=OwnedCudaCache(2)',
                      '        self._hardware_graphs=OwnedCudaCache(2)\n        self._projection_graphs=OwnedCudaCache(2)\n        self._validity_graphs=OwnedCudaCache(2)',1)
    methods='''    def _without_state_graphs(self, function, *inputs):
        old_modes=[model._graph_passthrough for _,_,model in self.pairs]
        try:
            for _,_,model in self.pairs:model._graph_passthrough=True
            return function(*inputs)
        finally:
            for (_,_,model),old in zip(self.pairs,old_modes):model._graph_passthrough=old

    def _project_eager(self, desired, previous):
        desired=desired.clone()
        for key,ids,model in self.pairs:
            ids=self._pair_indices[key] if desired.is_cuda else ids
            _,_,valid,_=model.state(desired[:,ids])
            desired[:,ids]=torch.where(valid[:,None],desired[:,ids],previous[:,ids])
        return desired

    def project_valid_targets(self, desired, previous):
        return self._projection_graphs.call(lambda *values:self._without_state_graphs(self._project_eager,*values),desired,previous)

    def _invalid_eager(self, q):
        invalid=torch.zeros(q.shape[0],dtype=torch.bool,device=q.device)
        for key,ids,model in self.pairs:
            ids=self._pair_indices[key] if q.is_cuda else ids
            _,_,valid,_=model.state(q[:,ids]);invalid|=~valid
        return invalid

    def invalid_at_pose(self, q):
        # Telemetry passes the fresh post-PhysX pose, never the previous actuator pose.
        return self._validity_graphs.call(lambda *values:self._without_state_graphs(self._invalid_eager,*values),q)

'''
    text=text.replace('    def _hardware_kernel(',methods+'    def _hardware_kernel(',1)
    path.write_text(text)
    env=destination/'training/environment.py'
    text=env.read_text()
    target='''  for _,ids,model in self.actuator.pairs:
   _,_,valid,_=model.state(desired[:,ids])
   desired[:,ids]=torch.where(valid[:,None],desired[:,ids],self.targets[:,ids])'''
    validity='''  invalid=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
  for _,ids,model in self.actuator.pairs:
   _,_,valid,_=model.state(d.joint_pos[:,ids]);invalid|=~valid'''
    assert text.count(target)==1 and text.count(validity)==1
    text=text.replace(target,'  desired=self.actuator.project_valid_targets(desired,self.targets)',1)
    text=text.replace(validity,'  invalid=self.actuator.invalid_at_pose(d.joint_pos)',1)
    env.write_text(text)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('iteration', choices=['01', '02', '03', '04', '05'])
    args = p.parse_args()
    destination = here / ('iteration_' + args.iteration) / 'source/physical_v29'
    assert not destination.exists()
    previous = {'01':'baseline','02':'iteration_01','03':'iteration_02','04':'iteration_03','05':'iteration_04'}[args.iteration]
    shutil.copytree(here / previous / 'source/physical_v29', destination,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    {'01':first,'02':second,'03':third,'04':fourth,'05':fifth}[args.iteration](destination)
    freeze(destination)
    print(destination)
