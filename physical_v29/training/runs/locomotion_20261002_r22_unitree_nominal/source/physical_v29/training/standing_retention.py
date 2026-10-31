"""Training-only stopped-command behavior regularization for residual PPO.

Inject the exact gradient of a bounded-sample mean-action MSE into PPO's mean
output backward pass. This avoids forking the installed RSL-RL update routine.
The frozen teacher never sends commands to the simulator. Inference has no
hook, no action blending and no access to the teacher.
"""
import torch

LEARNER_VERSION = 'standing_retention_v1'


def validate_teacher(infos, contract, expected_hashes):
    """Accept the already-qualified legacy actor with exact file-byte binding.

    Its original metadata predates optional motion/actuator fields. Asset and
    observation contract remain mandatory; present extra hashes must match.
    The chain independently requalifies this same teacher under current assets.
    """
    assert infos['contract']==contract,'Standing teacher observation contract mismatch'
    assert infos['asset_sha256']==expected_hashes['asset_sha256'],'Standing teacher asset mismatch'
    for key,value in expected_hashes.items():
        if key in infos:assert infos[key]==value,f'Standing teacher {key} mismatch'
    return sorted(set(expected_hashes)-set(infos))


class StandingRetention:
    def __init__(self, actor, teacher, weight=.25, max_samples=256, command_index=246):
        self.actor=actor;self.teacher=teacher.eval().requires_grad_(False)
        self.weight=weight;self.max_samples=max_samples;self.command_index=command_index
        self._observations=None;self._error=None;self._samples=0;self._batches=0
        self._handles=[actor.register_forward_pre_hook(self._capture),
                       actor.mlp.register_forward_hook(self._regularize)]

    def _capture(self, module, args):
        self._observations=args[0] if module.training and torch.is_grad_enabled() else None

    def _regularize(self, module, inputs, mean):
        observations=self._observations;self._observations=None
        if observations is None or not mean.requires_grad:return
        raw=observations['policy']
        ids=(raw[:,self.command_index].abs()<=.08).nonzero(as_tuple=False).flatten()
        if not ids.numel():return
        stride=max(1,(ids.numel()+self.max_samples-1)//self.max_samples)
        ids=ids[::stride][:self.max_samples]
        with torch.no_grad():
            target=self.teacher(observations[ids]).clamp(-3.,3.)
            difference=mean[ids].detach()-target
            derivative=2.*self.weight*difference/difference.numel()
            error=difference.square().sum()
        self._error=error if self._error is None else self._error+error
        self._samples+=difference.numel();self._batches+=1
        # d(weight * mean((actor_mean - frozen_teacher_mean)^2))/d(mean).
        # Other rows receive only their original PPO gradient.
        mean.register_hook(lambda gradient:gradient.index_add(0,ids,derivative))

    def read_audit(self):
        mse=float(self._error/self._samples) if self._samples else 0.
        result=dict(standing_prior_mse=mse,standing_prior_loss=self.weight*mse,
            weight=self.weight,batches=self._batches,max_teacher_samples_per_batch=self.max_samples,
            learner_version=LEARNER_VERSION,scope='Training gradients only; no teacher in inference or promotion')
        self._error=None;self._samples=0;self._batches=0
        return result

    def close(self):
        for handle in self._handles:handle.remove()
        self._observations=None
