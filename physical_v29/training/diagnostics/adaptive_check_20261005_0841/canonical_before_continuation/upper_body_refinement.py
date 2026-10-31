"""Training-only upper posture loss, restricted to the final actor output rows.

Inference, action mapping, PPO, exploration, critic and body physics are unchanged.
The auxiliary derivative updates only eight upper output rows; it never flows
through shared hidden features or directly updates the fifteen torso/leg rows.
Existing qualified priors retain their exact direct lower-body derivatives.
"""
import torch
from standing_retention import StandingRetention

UPPER_REFINEMENT_VERSION='upper_output_posture_v1'
UPPER_JOINTS=('head_yaw','head_pitch','left_shoulder_pitch','right_shoulder_pitch',
    'left_shoulder_roll','right_shoulder_roll','left_elbow_pitch','right_elbow_pitch')
ACTION_SCALE=.25
HEAD_WEIGHT=.15
ARM_WEIGHT=.025
SYMMETRY_WEIGHT=.025

def upper_indices(names):
    assert len(names)==23 and len(set(names))==23
    return [names.index(n) for n in UPPER_JOINTS]

class LocomotionOnlyRetention(StandingRetention):
    """Original .25 derivative/denominator on lower outputs; zero upper copying."""
    def __init__(self,*args,joint_names,**kwargs):
        self.upper=upper_indices(joint_names)
        super().__init__(*args,**kwargs)

    def _regularize(self,module,inputs,mean):
        observations=self._observations;self._observations=None
        if observations is None or not mean.requires_grad:return
        raw=observations['policy'];index=self.command_index if self.command_index is not None else raw.shape[-1]-80+6
        if self.command_range is None:selected=raw[:,index:index+3].norm(dim=1)<=.1
        else:
            speed=raw[:,index:index+2].norm(dim=1)
            selected=(speed>self.command_range[0])&(speed<=self.command_range[1]+1e-6)
        ids=selected.nonzero(as_tuple=False).flatten()
        if not ids.numel():return
        stride=max(1,(ids.numel()+self.max_samples-1)//self.max_samples);ids=ids[::stride][:self.max_samples]
        with torch.no_grad():
            target=self.teacher(observations[ids]).clamp(-3.,3.)
            difference=mean[ids].detach()-target;difference[:,self.upper]=0.
            derivative=2.*self.weight*difference/difference.numel()
            error=difference.square().sum()
        self._error=error if self._error is None else self._error+error
        self._samples+=difference.numel();self._batches+=1
        mean.register_hook(lambda gradient:gradient.index_add(0,ids,derivative))

    def read_audit(self):
        result=super().read_audit()
        result.update(output_scope='Original direct torso/leg derivative, zero head/arm copying',upper_excluded=list(UPPER_JOINTS))
        return result

def posture_loss(mean_upper):
    """Soft comfortable target bands in native default-offset action units.

    Head yaw/pitch neutral within3deg. Shoulders may swing freely within26deg
    pitch/10deg roll. Elbows retain the checked -.30rad standing reference with
    .30rad free deviation. Symmetry permits opposite pitch/roll arm motion and
    equal elbow flexion; no imposed gait clock, hand/root support or hard clamp.
    A Huber shoulder/head error has bounded slope above2 action units.
    """
    assert mean_upper.shape[1]==8
    tolerances=mean_upper.new_tensor([.0523599,.0523599,.45,.45,.174533,.174533,.30,.30])/ACTION_SCALE
    excess=(mean_upper.abs()-tolerances).clamp_min(0.)
    huber=torch.where(excess<=2.,.5*excess.square(),2.*(excess-1.))
    head=HEAD_WEIGHT*huber[:,:2].mean()
    arms=ARM_WEIGHT*huber[:,2:].mean()
    pairs=torch.stack([mean_upper[:,2]+mean_upper[:,3],mean_upper[:,4]+mean_upper[:,5],
        mean_upper[:,6]-mean_upper[:,7]],1)*.5
    symmetry=SYMMETRY_WEIGHT*torch.nn.functional.smooth_l1_loss(pairs,torch.zeros_like(pairs),beta=1.)
    return head+arms+symmetry,dict(head=head.detach(),arms=arms.detach(),symmetry=symmetry.detach())

class UpperBodyRefinement:
    def __init__(self,actor,joint_names,max_samples=256):
        self.actor=actor;self.names=list(joint_names);self.ids=upper_indices(self.names)
        self.layer=actor.mlp[-1];assert isinstance(self.layer,torch.nn.Linear) and self.layer.out_features==23
        self.max_samples=max_samples;self._features=None;self._pending={};self._batches=0;self._sum={}
        self._handles=[self.layer.register_forward_pre_hook(self._capture),
            actor.mlp.register_forward_hook(self._attach),
            self.layer.weight.register_hook(lambda g:self._parameter(g,'weight')),
            self.layer.bias.register_hook(lambda g:self._parameter(g,'bias'))]

    def _capture(self,module,args):
        self._features=args[0].detach() if module.training and torch.is_grad_enabled() else None

    def _attach(self,module,inputs,mean):
        features=self._features;self._features=None
        if features is None or not mean.requires_grad:return
        stride=max(1,(len(mean)+self.max_samples-1)//self.max_samples)
        rows=torch.arange(0,len(mean),stride,device=mean.device)[:self.max_samples]
        # A separate small leaf provides the explicit auxiliary derivative;
        # detached hidden features ensure it does not train the body trunk.
        with torch.enable_grad():
            upper=mean[rows][:,self.ids].detach().requires_grad_(True)
            loss,parts=posture_loss(upper)
            derivative=torch.autograd.grad(loss,upper)[0].detach()
        selected_features=features[rows]
        weight=derivative.transpose(0,1)@selected_features
        bias=derivative.sum(0)
        self._batches+=1
        for name,value in parts.items():self._sum[name]=value if name not in self._sum else self._sum[name]+value
        # This hook records the last-layer-only auxiliary gradient and returns
        # PPO's mean derivative unchanged. Parameter hooks add only upper rows.
        def remember(gradient):
            for name,value in [('weight',weight),('bias',bias)]:
                self._pending[name]=value if name not in self._pending else self._pending[name]+value
            return gradient
        mean.register_hook(remember)

    def _parameter(self,gradient,name):
        extra=self._pending.pop(name,None)
        if extra is None:return gradient
        result=gradient.clone();result[self.ids]+=extra
        return result

    def read_audit(self):
        assert not self._pending,'Unconsumed posture gradient'
        result=dict(version=UPPER_REFINEMENT_VERSION,batches=self._batches,
            max_samples=self.max_samples,head_weight=HEAD_WEIGHT,arm_weight=ARM_WEIGHT,
            symmetry_weight=SYMMETRY_WEIGHT,training_only=True,
            updated_rows=list(UPPER_JOINTS),shared_trunk_auxiliary_gradient=False,
            losses={k:float(v/max(1,self._batches)) for k,v in self._sum.items()})
        self._sum={};self._batches=0
        return result

    def close(self):
        for handle in self._handles:handle.remove()
        self._features=None;self._pending={}

def verify_upper_body_migration(prior,new_source,proof_path):
    """Strict exact full-state continuation, not an actor-only warm start."""
    import json,hashlib
    from pathlib import Path
    from curriculum_provenance import training_reward_signature
    prior=Path(prior);new_source=Path(new_source);proof_path=Path(proof_path)
    proof=json.loads(proof_path.read_text());state=json.loads((prior/'state.json').read_text())
    assert proof['schema']=='upper_body_output_refinement_v1'
    assert state['status']=='superseded_for_diagnostic_repair' and state.get('child_pid') is None
    assert Path(proof['prior_run']).resolve()==prior.resolve()
    assert Path(proof['checkpoint']).resolve()==Path(state['checkpoint']).resolve()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(state['checkpoint'])==proof['checkpoint_sha256']
    def inventory(root):
        return {p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file()
            and '__pycache__' not in p.parts and p.suffix!='.pyc' and 'usd' not in p.parts}
    old,new=inventory(prior/'source/physical_v29'),inventory(new_source)
    changed={k for k in old.keys()|new.keys() if k not in old or k not in new or old[k].read_bytes()!=new[k].read_bytes()}
    assert changed=={'training/worker_unitree.py','training/upper_body_refinement.py'},changed
    assert set(proof['implementation_hashes'])==changed
    for k,h in proof['implementation_hashes'].items():assert sha(new[k])==h
    assert training_reward_signature(new_source/'training')==proof['reward_signature']
    assert training_reward_signature(prior/'source/physical_v29/training')==proof['reward_signature']
    for key in ['cpu_tests','physical_validation','full_state_smoke']:
        record=proof[key];p=(proof_path.parent/record['path']).resolve()
        assert p.is_relative_to(proof_path.parent.resolve()) and sha(p)==record['sha256']
        assert json.loads(p.read_text())['passed'],key
    assert proof['actor_critic_Adam_exploration_retained'] and not proof['qualifies_new_posture']
    return state
