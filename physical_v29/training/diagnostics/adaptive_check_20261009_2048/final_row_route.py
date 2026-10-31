"""Training-only teacher derivative route using actual detached last features.

Only the selected teacher contribution is restricted. PPO and all other
auxiliaries keep their own original paths. This does not insulate other commands
from final-layer changes, global clipping, optimizer moments or physical feedback.
"""
import torch

VERSION='neutral267_exact_command_final21_rows_v2'

class FinalRowRoute:
    def __init__(self, actor, head_rows):
        self.layer=actor.mlp[-1]
        assert isinstance(self.layer,torch.nn.Linear) and self.layer.out_features==23
        assert len(head_rows)==2 and len(set(head_rows))==2
        self.head=list(head_rows)
        self.features=None
        self.pending={}
        self.pending_counts={}
        self.batches=0
        self.selected=0
        self.consumed={'weight':0,'bias':0}
        self.invocations={'weight':0,'bias':0}
        self.handles=[self.layer.register_forward_pre_hook(self.capture),
            self.layer.weight.register_hook(lambda g:self.parameter(g,'weight')),
            self.layer.bias.register_hook(lambda g:self.parameter(g,'bias'))]

    def capture(self,module,args):
        self.features=args[0].detach() if module.training and torch.is_grad_enabled() else None

    def attach(self,mean,ids,derivative,chosen):
        features=self.features
        self.features=None
        assert features is not None and mean.requires_grad
        assert features.shape==(len(mean),self.layer.in_features)
        assert derivative.shape==(len(ids),23) and chosen.shape==(len(ids),) and chosen.dtype==torch.bool
        assert torch.isfinite(derivative).all() and not derivative[:,self.head].any()
        # Unselected teacher states retain their original mean/trunk derivative.
        ordinary=derivative.clone()
        ordinary[chosen]=0
        restricted=derivative.clone()
        restricted[~chosen]=0
        weight=restricted.T@features[ids]
        bias=restricted.sum(0)
        assert not weight[self.head].any() and not bias[self.head].any()
        self.batches+=1
        self.selected+=int(chosen.sum())
        def remember(gradient):
            for name,extra in [('weight',weight),('bias',bias)]:
                self.pending[name]=extra if name not in self.pending else self.pending[name]+extra
                self.pending_counts[name]=self.pending_counts.get(name,0)+1
            return gradient.index_add(0,ids,ordinary)
        mean.register_hook(remember)

    def parameter(self,gradient,name):
        extra=self.pending.pop(name,None)
        if extra is None:return gradient
        self.consumed[name]+=self.pending_counts.pop(name)
        self.invocations[name]+=1
        assert extra.shape==gradient.shape and torch.isfinite(extra).all()
        assert not extra[self.head].any()
        return gradient+extra

    def read_audit(self):
        assert not self.pending,'Unconsumed restricted teacher gradient'
        assert not self.pending_counts
        assert self.consumed['weight']==self.batches and self.consumed['bias']==self.batches
        result=dict(version=VERSION,batches=self.batches,selected_states=self.selected,
            parameter_contributions=dict(self.consumed),head_rows=self.head,
            parameter_hook_invocations=dict(self.invocations),
            selected_teacher_trunk_derivative=False,other_teacher_mean_path_unchanged=True,
            PPO_and_other_auxiliary_paths_unchanged_before_global_clipping=True,
            inference_changed=False,global_clipping_and_Adam_interactions_not_isolated=True)
        self.batches=0
        self.selected=0
        self.consumed={'weight':0,'bias':0}
        self.invocations={'weight':0,'bias':0}
        return result
