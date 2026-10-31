"""Command-conditioned PPO advantages; raw rewards/GAE/critic unchanged.

This is an experimental task-conditioned optimization scale, not CPO/PCGrad,
not an unbiased estimator of the original mixed-command clipped objective, and
not a physical retention guarantee. Only stored truthful issued commands label
samples. Exact evaluated commands and continuous frontier form distinct groups.
"""
import json
from pathlib import Path
import torch

VERSION = 'command_conditioned_advantage_v1'
ANCHORS = (0., .15, .35, .65, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.)
FRAME_WIDTH = 80
COMMAND_OFFSET = 6


def command_groups(policy_observations):
    assert policy_observations.shape[-1] == 400
    command = policy_observations[..., -FRAME_WIDTH + COMMAND_OFFSET]
    labels = torch.full_like(command, len(ANCHORS), dtype=torch.long)
    labels[command <= .1] = 0
    for i, speed in enumerate(ANCHORS[1:], 1):
        labels[(command - speed).abs() < .005] = i
    assert torch.isfinite(command).all()
    return labels


def normalize_by_command(raw_advantages, labels):
    assert raw_advantages.shape == labels.shape + (1,)
    raw = raw_advantages.squeeze(-1)
    result = torch.empty_like(raw)
    audit = []
    for label in torch.unique(labels, sorted=True):
        mask = labels == label
        values = raw[mask]
        n = len(values)
        mean = values.mean()
        # Match RSL's Bessel-corrected convention within each populated task.
        std = values.std() if n >= 2 else torch.zeros_like(mean)
        scaled = (values - mean) / (std + 1e-8) if n >= 2 else torch.zeros_like(values)
        result[mask] = scaled
        key = int(label)
        audit.append(dict(group=key, command_m_s=ANCHORS[key] if key < len(ANCHORS) else None,
                          kind='exact_command' if key < len(ANCHORS) else 'continuous_frontier',
                          samples=n, raw_mean=float(mean), raw_std=float(std),
                          conditioned_mean=float(scaled.mean()),
                          conditioned_std=float(scaled.std()) if n >= 2 else 0.,
                          constant_or_singleton=bool(std == 0)))
    assert torch.isfinite(result).all()
    return result.unsqueeze(-1), audit


class CommandAdvantage:
    def __init__(self, algorithm, out, enabled):
        self.algorithm = algorithm
        self.original = algorithm.compute_returns
        self.enabled = enabled
        self.out = Path(out)
        self.updates = 0
        self.latest = {}
        assert not algorithm.normalize_advantage_per_mini_batch
        algorithm.compute_returns = self.compute_returns

    def compute_returns(self, observations):
        self.original(observations)
        st = self.algorithm.storage
        with torch.no_grad():
            raw = st.returns - st.values
            labels = command_groups(st.observations['policy'])
            conditioned, groups = normalize_by_command(raw, labels)
            original = st.advantages.clone() if self.updates == 0 else None
            if self.enabled:
                st.advantages = conditioned
            self.latest = dict(version=VERSION, applied=self.enabled, update=self.updates,
                               groups=groups, global_raw_mean=float(raw.mean()),
                               global_raw_std=float(raw.std()),
                               actual_advantage_mean=float(st.advantages.mean()),
                               actual_advantage_std=float(st.advantages.std()),
                               original_GAE_returns_values_rewards_unchanged=True,
                               no_command_or_observation_or_action_write=True)
            if self.updates == 0:
                torch.save(dict(raw_advantage=raw.cpu(), old_global_normalized=original.cpu(),
                                conditioned_advantage=conditioned.cpu(), actual_advantage=st.advantages.cpu(),
                                labels=labels.cpu(), policy_observations=st.observations['policy'].cpu(),
                                returns=st.returns.cpu(), values=st.values.cpu(), rewards=st.rewards.cpu(),
                                dones=st.dones.cpu(), actions=st.actions.cpu()),
                           self.out / 'first_rollout_normalization.pt')
            with (self.out / 'command_advantage.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(self.latest) + '\n')
        self.updates += 1


def install(algorithm, out, enabled):
    return CommandAdvantage(algorithm, out, enabled)


def cpu_tests():
    checks = []
    obs = torch.zeros((2, 6, 400))
    obs[..., 326] = torch.tensor([0., .15, 1.5, 3.5, 3.2, 2.])
    labels = command_groups(obs)
    assert labels[0].tolist() == [0, 1, 5, 9, 13, 6]
    checks.append('truthful latest history command offset and frontier/exact separation')
    raw = torch.tensor([[[1.], [2.], [3.], [4.], [5.], [6.]],
                        [[3.], [6.], [9.], [12.], [15.], [18.]]])
    x, audit = normalize_by_command(raw, labels)
    shifted = raw.clone(); shifted[:, 1] += 10000.
    y, _ = normalize_by_command(shifted, labels)
    assert torch.equal(x, y)
    checks.append('one-command offset cannot shift other command advantages')
    scaled = raw.clone(); scaled[:, 2] = scaled[:, 2] * 3. + 10.
    z, _ = normalize_by_command(scaled, labels)
    torch.testing.assert_close(x, z)
    checks.append('positive affine task scaling invariance')
    global_x = (raw-raw.mean())/(raw.std()+1e-8)
    global_y = (shifted-shifted.mean())/(shifted.std()+1e-8)
    assert not torch.equal(global_x[:, 3], global_y[:, 3])
    checks.append('counterexample reproduces old cross-command normalization coupling')
    assert all(abs(a['conditioned_mean']) < 1e-6 and abs(a['conditioned_std']-1) < 1e-6 for a in audit)
    checks.append('per-task zero mean and Bessel unit scale')
    a, _ = normalize_by_command(torch.ones((1, 3, 1)), torch.tensor([[0, 1, 1]]))
    assert torch.equal(a, torch.zeros_like(a))
    checks.append('constant and singleton groups finite with zero signal')
    obs[:, :, 246] = 5.  # Previous history frame cannot replace actual issued command.
    assert torch.equal(labels, command_groups(obs))
    checks.append('previous command history does not mislabel action-time sample')
    edges = torch.zeros((1, 5, 400)); edges[0, :, 326] = torch.tensor([.1, .15, .356, 1.499, 5.])
    assert command_groups(edges).tolist() == [[0, 1, 13, 5, 12]]
    checks.append('stopped and exact tolerance boundaries')
    from types import SimpleNamespace
    class Fake:
        normalize_advantage_per_mini_batch = False
        def __init__(self):
            self.storage = SimpleNamespace(returns=raw.clone(), values=torch.zeros_like(raw),
                observations={'policy':obs.clone()}, rewards=raw.clone(), dones=torch.zeros_like(raw), actions=raw.clone())
        def compute_returns(self, incoming):
            self.storage.advantages=(raw-raw.mean())/(raw.std()+1e-8)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        for enabled in (False, True):
            alg=Fake(); before={k: getattr(alg.storage,k).clone() for k in ('returns','values','rewards','dones','actions')}
            hook=install(alg,d,enabled); alg.compute_returns(None)
            assert all(torch.equal(v,getattr(alg.storage,k)) for k,v in before.items())
            assert torch.equal(alg.storage.observations['policy'], obs)
            assert torch.equal(alg.storage.advantages, x if enabled else global_x)
            checks.append('enabled hook changes only advantages' if enabled else 'disabled hook bitwise original advantages')
    return dict(passed=True, checks=checks, version=VERSION, physical_retention_guarantee=False,
                original_mixed_objective_unbiasedness_claim=False)


if __name__ == '__main__':
    print(json.dumps(cpu_tests(), indent=2))
