"""Bounded read-only PPO diagnostics; no extra update, RNG draw or simulator."""
import torch

class PpoUpdateAudit:
    def __init__(self, algorithm, max_samples=512):
        self.algorithm=algorithm;self.max_samples=max_samples;self.latest={}
        self.original=algorithm.update
        algorithm.update=self.update

    def update(self):
        alg=self.algorithm;st=alg.storage
        with torch.no_grad():
            obs=st.observations.flatten(0,1)
            ids=torch.linspace(0,len(obs)-1,min(len(obs),self.max_samples),device=st.actions.device).long()
            sampled=obs[ids].clone()
            old_mean,old_std=[p.flatten(0,1)[ids].clone() for p in st.distribution_params]
            actions=st.actions.flatten(0,1)[ids].clone()
            old_log=st.actions_log_prob.flatten(0,1)[ids].squeeze(-1).clone()
            returns=st.returns.flatten(0,1)[ids].squeeze(-1).clone()
            values=st.values.flatten(0,1)[ids].squeeze(-1).clone()
            before=alg.actor(sampled).clone()
            before_std=alg.actor.distribution.log_std_param.exp().expand_as(before).clone()
            before_kl=torch.distributions.kl_divergence(torch.distributions.Normal(old_mean,old_std),torch.distributions.Normal(before,before_std)).sum(-1)
            dones=st.dones.flatten(0,1)[ids].squeeze(-1).bool()
            target_variance=returns.var(unbiased=False)
            command_index=sampled['policy'].shape[-1]-80+6
            stopped=sampled['policy'][:,command_index:command_index+3].norm(dim=1)<=.1
            initial=dict(sample_count=len(ids),pre_update_kl=float(before_kl.mean()),
                pre_update_kl_nonterminal=float(before_kl[~dones].mean()) if (~dones).any() else None,
                pre_update_kl_reset=float(before_kl[dones].mean()) if dones.any() else None,
                reset_sample_fraction=float(dones.float().mean()),
                return_mean=float(returns.mean()),return_std=float(returns.std(unbiased=False)),
                value_explained_variance=float(1-(returns-values).var(unbiased=False)/target_variance.clamp_min(1e-8)),
                normalized_advantage_std=float(st.advantages.std(unbiased=False)),
                rollout_action_clip_fraction=float((st.actions.abs()>3).float().mean()),
                stopped_sample_fraction=float(stopped.float().mean()),
                optimizer_lr_before=float(alg.optimizer.param_groups[0]['lr']))
            # This interface disables empirical normalization and dropout.
            # Before any optimizer step, the same recorded state must produce
            # the distribution that generated its action, apart from roundoff.
            assert initial['pre_update_kl']<1e-5, 'PPO_STALE_OBSERVATION_OR_DISTRIBUTION: '+str(initial)
        losses=self.original()
        with torch.no_grad():
            after=alg.actor(sampled)
            after_std=alg.actor.distribution.log_std_param.exp().expand_as(after)
            distribution=torch.distributions.Normal(after,after_std)
            kl=torch.distributions.kl_divergence(torch.distributions.Normal(old_mean,old_std),distribution).sum(-1)
            ratio=torch.exp(distribution.log_prob(actions).sum(-1)-old_log)
            displacement=(after-before).square().mean(-1).sqrt()
            initial.update(post_update_kl=float(kl.mean()),post_update_kl_p99=float(torch.quantile(kl,.99)),
                likelihood_clip_fraction=float(((ratio<1-alg.clip_param)|(ratio>1+alg.clip_param)).float().mean()),
                mean_action_update_rms=float(displacement.mean()),
                stopped_action_update_rms=float(displacement[stopped].mean()) if stopped.any() else None,
                moving_action_update_rms=float(displacement[~stopped].mean()) if (~stopped).any() else None,
                optimizer_lr_after=float(alg.optimizer.param_groups[0]['lr']),losses={k:float(v) for k,v in losses.items()},
                scope='Evenly spaced bounded rollout sample; diagnostic only. No PPO or inference changes.')
            self.latest=initial
        return losses

    def close(self):self.algorithm.update=self.original
