"""Pose-dependent two-motor rod linkage from saved Fusion joint/eye frames.

q uses the anatomical URDF angles, theta the factory-geared motor-output angles.
Virtual work: theta_dot = J q_dot; tau_joint = J.T tau_motor.
This preserves the transmission without introducing URDF closed-loop constraints.
It does not animate passive visual rods or qualify rod-end contact/strength.
"""
import torch


def rotate(v, axis, angle):
    v, axis = torch.broadcast_tensors(v, axis)
    c, s = torch.cos(angle)[..., None], torch.sin(angle)[..., None]
    return v*c + torch.linalg.cross(axis, v, dim=-1)*s + axis*(axis*v).sum(-1,keepdim=True)*(1-c)


class CadDifferential:
    def __init__(self, spec):
        self.spec = spec
        self._tensors = {}
        self._prepared = {}

    def _geometry(self, q):
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

    def limit(self, q, qd, desired, torque_limit, speed_limit, efficiency=1.0):
        """Clip motor torques, then map back; no independent pitch/roll maxima.

        BLDC inputs use a user-assumed flat current limit below a hard speed cap.
        At/above the cap only braking torque is retained. This is not a measured
        voltage/current/thermal envelope. Near singular/unreachable poses effort
        is zero; a task must terminate/reset rather than train through these poses.
        """
        theta,J,valid,errors = self.state(q)
        safeJ = torch.where(valid[:,None,None],J,torch.eye(2,device=q.device,dtype=q.dtype)[None])
        # Analytic 2x2 inverse avoids a host-synchronizing CUDA linalg call.
        det=safeJ[:,0,0]*safeJ[:,1,1]-safeJ[:,0,1]*safeJ[:,1,0]
        motor=torch.stack((safeJ[:,1,1]*desired[:,0]-safeJ[:,1,0]*desired[:,1],
                           -safeJ[:,0,1]*desired[:,0]+safeJ[:,0,0]*desired[:,1]),1)/det[:,None]/efficiency
        motor_speed = (safeJ@qd[...,None]).squeeze(-1)
        applied = motor.clamp(-torque_limit,torque_limit)
        applied = torch.where((motor_speed.abs()>=speed_limit)&(applied*motor_speed>0),torch.zeros_like(applied),applied)
        applied = torch.where(valid[:,None],applied,torch.zeros_like(applied))
        output = efficiency*(safeJ.transpose(1,2)@applied[...,None]).squeeze(-1)
        return output,{'theta':theta,'J':J,'valid':valid,'closure_error_mm':errors,'motor_requested':motor,'motor_applied':applied,'motor_velocity':motor_speed}
