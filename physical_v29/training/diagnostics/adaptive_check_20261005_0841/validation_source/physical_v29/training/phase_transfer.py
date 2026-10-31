"""EXPERIMENTAL training-only gait transfer; never alters inference observations.

Frozen teacher alone receives diagnostic, physically tested head phase context.
Student receives its actual 400-value observations. Auxiliary gradient affects
only torso/leg rows of its final Linear; PPO and existing posture loss are intact.
"""
import torch
VERSION='diagnostic_neutral_head_running_rows_v1'
UPPER=('head_yaw','head_pitch','left_shoulder_pitch','right_shoulder_pitch',
 'left_shoulder_roll','right_shoulder_roll','left_elbow_pitch','right_elbow_pitch')

def context_inputs(raw, table, columns):
    result=raw.clone().reshape(-1,5,80)
    phase=torch.remainder(torch.atan2(result[:,:,78],result[:,:,79])/(2*torch.pi),1.)
    location=phase*len(table); lo=location.floor().long()%len(table); hi=(lo+1)%len(table)
    blend=(location-location.floor()).unsqueeze(-1)
    result[:,:,columns]=table[lo]*(1.-blend)+table[hi]*blend
    return result.reshape(-1,400)

class RunningRowTransfer:
    def __init__(self,actor,teacher,joint_names,context,max_samples=256,weight=.25):
        self.actor=actor;self.teacher=teacher.eval().requires_grad_(False)
        self.layer=actor.mlp[-1];assert isinstance(self.layer,torch.nn.Linear) and self.layer.out_features==23
        self.lower=[i for i,n in enumerate(joint_names) if n not in UPPER]
        self.columns=[start+joint_names.index(n) for start in [9,32,55] for n in ['head_yaw','head_pitch']]
        assert self.columns==context['columns'] and len(self.lower)==15
        self.table=torch.tensor(context['contexts']['1.5'],device=self.layer.weight.device,dtype=self.layer.weight.dtype)
        self.weight=weight;self.max_samples=max_samples;self._raw=None;self._features=None;self._pending={};self._audit=[]
        self._handles=[actor.register_forward_pre_hook(self._capture),self.layer.register_forward_pre_hook(self._features_hook),
            actor.mlp.register_forward_hook(self._attach),
            self.layer.weight.register_hook(lambda g:self._parameter(g,'weight')),
            self.layer.bias.register_hook(lambda g:self._parameter(g,'bias'))]

    def _capture(self,module,args):
        self._raw=args[0]['policy'].detach().clone() if module.training and torch.is_grad_enabled() else None

    def _features_hook(self,module,args):
        self._features=args[0].detach() if module.training and torch.is_grad_enabled() else None

    def _attach(self,module,inputs,mean):
        raw=self._raw;features=self._features;self._raw=None;self._features=None
        if raw is None or features is None or not mean.requires_grad:return
        frames=raw.reshape(-1,5,80)
        selected=((frames[:,:,6]-1.5).abs()<1e-6).all(1)
        selected&=(frames[:,:,78:80].square().sum(-1)>.5).all(1)
        selected&=(frames[:,-1,[self.columns[0],self.columns[1]]].abs()<.174533).all(1)
        rows=selected.nonzero().flatten()
        if not len(rows):return
        stride=max(1,(len(rows)+self.max_samples-1)//self.max_samples);rows=rows[::stride][:self.max_samples]
        with torch.no_grad():
            target=self.teacher({'policy':context_inputs(raw[rows],self.table,self.columns)}).clamp(-3.,3.)[:,self.lower]
            difference=mean[rows][:,self.lower].detach()-target
            # Same .25 strength and23-output denominator as retained priors,
            # with Huber slope capped at1 to bound auxiliary influence.
            derivative=self.weight*difference.clamp(-1.,1.)/(len(rows)*23.)
            features=features[rows]
            extra_w=derivative.T@features;extra_b=derivative.sum(0)
            self._audit.append(torch.stack([difference.square().mean(),
                (mean[rows][:,self.lower].detach().clamp(-3.,3.)-target).square().mean(),
                derivative.norm(),extra_w.norm(),mean.new_tensor(len(rows))]).detach())
        def remember(gradient):
            for name,value in [('weight',extra_w),('bias',extra_b)]:
                self._pending[name]=value if name not in self._pending else self._pending[name]+value
            return gradient
        mean.register_hook(remember)

    def _parameter(self,gradient,name):
        extra=self._pending.pop(name,None)
        if extra is None:return gradient
        result=gradient.clone();result[self.lower]+=extra
        return result

    def read_audit(self):
        assert not self._pending
        result=dict(version=VERSION,batches=len(self._audit),weight=self.weight,max_samples=self.max_samples,
          student_observations_truthful=True,inference_hook=False,teacher_context_only=True,
          auxiliary_shared_trunk_gradient=False,auxiliary_upper_rows=False,
          sample_scope='All5 frames exact1.5command, valid phase, actual latest head within10deg')
        if self._audit:
            values=torch.stack(self._audit).mean(0).cpu().tolist()
            result.update(dict(zip(['raw_lower_MSE','effective_clipped_lower_MSE','mean_output_aux_norm','mean_final_weight_aux_norm','mean_selected_samples'],values)))
        self._audit=[];return result

    def close(self):
        for handle in self._handles:handle.remove()
        self._pending={};self._raw=None;self._features=None
