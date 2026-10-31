"""Migrate only previously unused command/phase scales of a standing prior.

Never overwrite the source. Physical observation statistics and network weights
stay unchanged. Commands/phase remain zero at stand, so its outputs are exact.
The legacy reward tag deliberately triggers a fresh v2 critic and optimizer.
"""
import argparse, copy, hashlib, json
from pathlib import Path
import torch
import torch.nn.functional as F


def known_input_ids():
    return [frame * 80 + channel for frame in range(4) for channel in [6, 7, 8, 78, 79]]


def repair_state(original):
    actor = copy.deepcopy(original)
    ids = known_input_ids()
    for field, value in [('_mean', 0.), ('_var', 1.), ('_std', 1.)]:
        actor['obs_normalizer.' + field][:, ids] = value
    return actor


def predict(observations, actor):
    latent = (observations - actor['obs_normalizer._mean']) / (actor['obs_normalizer._std'] + .01)
    for layer in (0, 2, 4, 6):
        latent = F.linear(latent, actor[f'mlp.{layer}.weight'], actor[f'mlp.{layer}.bias'])
        if layer != 6:
            latent = F.elu(latent)
    return latent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    assert source.resolve() != output.resolve() and not output.exists()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    saved = torch.load(source, map_location='cpu', weights_only=False)
    original = saved['actor_state_dict']
    fixed = repair_state(original)
    ids = known_input_ids()
    keep = torch.ones(320, dtype=torch.bool)
    keep[ids] = False
    for key in original:
        if key in ['obs_normalizer._mean', 'obs_normalizer._var', 'obs_normalizer._std']:
            assert torch.equal(original[key][:, keep], fixed[key][:, keep])
        else:
            assert torch.equal(original[key], fixed[key]), key
    generator = torch.Generator().manual_seed(971)
    standing = original['obs_normalizer._mean'] + torch.randn((256, 320), generator=generator) * original['obs_normalizer._std'] * .15
    standing[:, ids] = 0.
    with torch.no_grad():
        difference = float((predict(standing, original) - predict(standing, fixed)).abs().max())
    assert difference == 0., difference
    moving = standing[:120].clone()
    phases = torch.arange(120) * 2 * torch.pi / 120
    for frame in range(4):
        moving[:, frame * 80 + 6] = .15
        moving[:, frame * 80 + 78] = torch.sin(phases)
        moving[:, frame * 80 + 79] = torch.cos(phases)
    old_normalized = (moving - original['obs_normalizer._mean']) / (original['obs_normalizer._std'] + .01)
    new_normalized = (moving - fixed['obs_normalizer._mean']) / (fixed['obs_normalizer._std'] + .01)
    report = dict(source=str(source),source_sha256=digest,output=str(output),
        standing_observation_trials=256,standing_max_action_difference=difference,
        physical_observation_statistics_unchanged=True,network_weights_unchanged=True,
        observation_count=int(original['obs_normalizer.count']),changed_channels=ids,
        old_command_phase_peak_normalized=float(old_normalized[:, ids].abs().max()),
        new_command_phase_peak_normalized=float(new_normalized[:, ids].abs().max()),
        scope='CPU input-scale migration only; standing must be revalidated in Isaac and walking remains unqualified')
    saved['actor_state_dict'] = fixed
    saved['infos']['actor_input_repair'] = report
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, output)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
