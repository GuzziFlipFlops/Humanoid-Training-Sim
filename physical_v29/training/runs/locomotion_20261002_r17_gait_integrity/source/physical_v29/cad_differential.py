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

    def state(self, q):
        """Return theta [N,2], analytic J [N,2,2], and closure validity [N]."""
        def t(v):
            key=(str(q.device),q.dtype,tuple(v))
            if key not in self._tensors:
                self._tensors[key]=torch.as_tensor(v,device=q.device,dtype=q.dtype)
            return self._tensors[key]
        p1,p2,a1,a2 = [t(self.spec[k]) for k in ('pitch_origin_mm','roll_origin_mm','pitch_axis','roll_axis')]
        p,r = q[:,0],q[:,1]
        def transform(point):
            return p1 + rotate(p2-p1+rotate(t(point)-p2,a2,r),a1,p)
        def inverse(point):
            return p2 + rotate(p1+rotate(t(point)-p1,a1,-p)-p2,a2,-r)
        def vector_transform(v):
            return rotate(rotate(v,a2,r),a1,p)
        a2world = rotate(a2,a1,p)
        p2world = p1+rotate(p2-p1,a1,p)
        angles,rows,valids,errors = [],[],[],[]
        for motor in self.spec['motors']:
            m,a,v,b = [t(motor[k]) for k in ('center_mm','axis','crank_vector_mm','bottom_eye_mm')]
            moving_motor = self.spec['motor_on_moving_link']
            bottom = b.expand(q.shape[0],3) if moving_motor else transform(b)
            ref = inverse(b) if moving_motor else bottom
            parallel = a*(a*v).sum()
            vp = v-parallel
            cross = torch.linalg.cross(a,v)
            d = m+parallel-ref
            ca,sa = (d*vp).sum(-1),(d*cross).sum(-1)
            amp = torch.sqrt(ca.square()+sa.square())
            value = (motor['rod_length_mm']**2-d.square().sum(-1)-vp.square().sum())/(2*amp.clamp_min(1e-9))
            phi = torch.atan2(sa,ca)
            ac = torch.acos(value.clamp(-1,1))
            candidates = torch.stack((phi+ac,phi-ac),-1)
            candidates = torch.atan2(torch.sin(candidates),torch.cos(candidates))
            theta = candidates.gather(1,candidates.abs().argmin(1,keepdim=True)).squeeze(1)
            tip_local = m+rotate(v,a,theta)
            tip = transform(tip_local) if moving_motor else tip_local
            tangent = rotate(cross,a,theta)
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
