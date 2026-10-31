"""Experimental training-only DC regularizer; no inference or physical override.

Equal phase-bin weighting per exact command/frontier band removes sampling bias.
Only two final shoulder-pitch output rows receive auxiliary gradients. The new
gradient is additionally capped at 10% of that minibatch's actual PPO shoulder
final-layer gradient norm. Existing upper refinement remains untouched.
"""
import torch
WEIGHT=.025
MEAN_BAND_ACTION=(5.*torch.pi/180.)/.25
MAX_PPO_FRACTION=.1
PHASE_BINS=8
GROUPS=10

def phase_mean_loss(effective_shoulder, raw):
    assert effective_shoulder.shape==(len(raw),2) and raw.shape[1]==400
    frames=raw.reshape(-1,5,80)
    x=frames[:,-1,6];y=frames[:,-1,7]
    groups=torch.full_like(x,-1,dtype=torch.long)
    for i,speed in enumerate([.15,.35,.65,1.,1.5,2.,2.5,3.]):
        groups=torch.where((x-speed).abs()<1e-6,i,groups)
    groups=torch.where((x>3.+1e-6)&(x<3.25),8,groups)
    groups=torch.where((x>=3.25)&(x<=3.5+1e-6),9,groups)
    eligible=(groups>=0)&(y.abs()<1e-6)&((frames[:,:,6]-x[:,None]).abs()<1e-6).all(1)
    eligible&=(frames[:,:,7].abs()<1e-6).all(1)
    phase_norm=frames[:,-1,78:80].square().sum(1)
    eligible&=(phase_norm>.99)&(phase_norm<1.01)
    angle=torch.remainder(torch.atan2(frames[:,-1,78],frames[:,-1,79]),2*torch.pi)
    bins=(angle*(PHASE_BINS/(2*torch.pi))).long().clamp(0,PHASE_BINS-1)
    index=groups.clamp_min(0)*PHASE_BINS+bins
    count=raw.new_zeros(GROUPS*PHASE_BINS).index_add(0,index,eligible.to(raw.dtype))
    sums=effective_shoulder.new_zeros((GROUPS*PHASE_BINS,2)).index_add(0,index,effective_shoulder*eligible[:,None])
    count=count.reshape(GROUPS,PHASE_BINS)
    phase_means=(sums/count.flatten().clamp_min(1.)[:,None]).reshape(GROUPS,PHASE_BINS,2)
    complete=(count>=2).all(1)
    balanced=phase_means.mean(1)
    excess=(balanced.abs()-MEAN_BAND_ACTION).clamp_min(0.)
    huber=torch.where(excess<=1.,.5*excess.square(),excess-.5)
    loss=WEIGHT*(huber*complete[:,None]).sum()/(2*complete.sum().clamp_min(1))
    return loss,dict(balanced=balanced.detach(),counts=count.detach(),complete=complete.detach(),eligible=eligible.detach())

class ShoulderMeanRefinement:
    def __init__(self,actor,joint_names,apply=True,max_samples=4096):
        self.actor=actor;self.ids=[joint_names.index(n) for n in ['left_shoulder_pitch','right_shoulder_pitch']]
        self.upper_ids=[joint_names.index(n) for n in ['head_yaw','head_pitch','left_shoulder_pitch','right_shoulder_pitch','left_shoulder_roll','right_shoulder_roll','left_elbow_pitch','right_elbow_pitch']]
        self.layer=actor.mlp[-1];self.apply=apply;self.max_samples=max_samples
        self.pending={};self.features=None;self.batches=0;self.stats={};self.first=None
        self.handles=[self.layer.register_forward_pre_hook(self.capture),actor.mlp.register_forward_hook(self.attach),
            self.layer.weight.register_hook(lambda g:self.parameter(g,'weight')),
            self.layer.bias.register_hook(lambda g:self.parameter(g,'bias'))]
    def capture(self,module,args):
        self.features=args[0].detach() if module.training and torch.is_grad_enabled() else None
    def attach(self,module,inputs,mean):
        features=self.features;self.features=None
        if features is None or not mean.requires_grad:return
        raw=inputs[0].detach();assert raw.shape[1]==400
        stride=max(1,(len(mean)+self.max_samples-1)//self.max_samples)
        rows=torch.arange(0,len(mean),stride,device=mean.device)[:self.max_samples]
        with torch.enable_grad():
            leaf=mean[rows][:,self.ids].detach().requires_grad_(True)
            loss,details=phase_mean_loss(leaf.clamp(-3.,3.),raw[rows])
            derivative=torch.autograd.grad(loss,leaf)[0].detach()
        extra_w=derivative.T@features[rows];extra_b=derivative.sum(0)
        extra_norm=torch.sqrt(extra_w.square().sum()+extra_b.square().sum())
        # Passive reproduction of the unchanged original upper auxiliary on its
        # exact256-row sample; it is never added a second time.
        from upper_body_refinement import posture_loss
        old_stride=max(1,(len(mean)+255)//256)
        old_rows=torch.arange(0,len(mean),old_stride,device=mean.device)[:256]
        with torch.enable_grad():
            old_leaf=mean[old_rows][:,self.upper_ids].detach().requires_grad_(True)
            old_derivative=torch.autograd.grad(posture_loss(old_leaf)[0],old_leaf)[0].detach()
        old_w=old_derivative.T@features[old_rows];old_b=old_derivative.sum(0)
        old_norm=torch.sqrt(old_w.square().sum()+old_b.square().sum())
        old_shoulder_norm=torch.sqrt(old_w[2:4].square().sum()+old_b[2:4].square().sum())
        def remember(gradient):
            pw=gradient[:,self.ids].T@features;pb=gradient[:,self.ids].sum(0)
            pnorm=torch.sqrt(pw.square().sum()+pb.square().sum())
            scale=(MAX_PPO_FRACTION*pnorm/extra_norm.clamp_min(1e-12)).clamp(max=1.)
            if self.apply:
                for key,value in [('weight',extra_w*scale),('bias',extra_b*scale)]:
                    self.pending[key]=value if key not in self.pending else self.pending[key]+value
            self.batches+=1
            quantities=dict(loss=loss.detach(),proposed_gradient_norm=extra_norm,actual_PPO_shoulder_norm=pnorm,
                bounded_gradient_norm=extra_norm*scale,scale=scale,eligible_fraction=details['eligible'].float().mean(),
                complete_groups=details['complete'].sum(),existing_upper_gradient_norm=old_norm,
                existing_shoulder_auxiliary_gradient_norm=old_shoulder_norm)
            for key,value in quantities.items():self.stats[key]=value if key not in self.stats else self.stats[key]+value
            if self.first is None:
                self.first={**{k:float(v) for k,v in quantities.items()},
                    'phase_balanced_mean_deg':(details['balanced']*.25*180/torch.pi).cpu().tolist(),
                    'counts':details['counts'].cpu().tolist(),'complete':details['complete'].cpu().tolist(),
                    'effective_sample_clip_fraction':float((leaf.detach().abs()>3).float().mean())}
            return gradient
        mean.register_hook(remember)
    def parameter(self,gradient,key):
        extra=self.pending.pop(key,None)
        if extra is None:return gradient
        result=gradient.clone();result[self.ids]+=extra;return result
    def read_audit(self):
        assert not self.pending
        result=dict(version='experimental_phase_balanced_shoulder_mean_v1',applied=self.apply,batches=self.batches,
            max_samples=self.max_samples,weight=WEIGHT,mean_band_deg=5.,max_PPO_shoulder_gradient_fraction=MAX_PPO_FRACTION,
            groups=['.15','.35','.65','1','1.5','2','2.5','3','(3,3.25)','[3.25,3.5]'],
            phase_bins=PHASE_BINS,min_samples_each_bin=2,shared_trunk_direct_gradient=False,
            output_rows=['left_shoulder_pitch','right_shoulder_pitch'],
            statistics={k:float(v/max(1,self.batches)) for k,v in self.stats.items()},first_actual_batch=self.first)
        self.stats={};self.batches=0;self.first=None;return result
    def close(self):
        for h in self.handles:h.remove()
        self.pending={};self.features=None
