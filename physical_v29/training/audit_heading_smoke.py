"""CPU-only saved-artifact audit. Does not start Isaac or qualify a policy."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import psutil


HERE = Path(__file__).resolve().parent
OUT = HERE / 'diagnostics/r25_run_review_20261003_1919/heading_smoke_audit.json'


def main():
    active = json.loads((HERE / 'ACTIVE_RUN.json').read_text(encoding='utf-8'))
    run = Path(active['run'])
    assert run.parent.resolve() == (HERE / 'runs').resolve()
    assert run.name == 'locomotion_20261003_r27_heading_commands_checked'
    results = []
    for name in ('003_checkpoint_reload', '005_phase_retention_reload'):
        folder = run / name
        with np.load(folder / 'gait_diagnostics_50hz.npz', allow_pickle=False) as z:
            observed = z['observed_command']
            action = z['action_command']
            difference = float(np.max(np.abs(observed - action)))
            assert np.isfinite(observed).all() and np.isfinite(action).all()
            assert difference == 0., f'Observed/action command mismatch in {name}'
            results.append(dict(operation=name, trace_shape=list(action.shape),
                observed_action_command_max_difference=difference,
                heading_command_abs_max_rad_s=float(np.max(np.abs(action[..., 2]))),
                heading_target_spread_rad=float(np.ptp(z['heading_target'])),
                scope='Saved 16-robot 50 Hz command trace; not a full qualification'))
    training = []
    for name in ('002_ppo_smoke', '004_phase_retention_smoke'):
        rows = [json.loads(s) for s in (run / name / 'progress.jsonl').read_text().splitlines()]
        pre_kl = max(r['ppo']['pre_update_kl'] for r in rows)
        command_difference = max(r['behavior']['action_command_difference'] for r in rows)
        assert pre_kl == 0. and command_difference == 0.
        training.append(dict(operation=name, updates=len(rows),
            pre_update_kl_max=pre_kl, action_reward_command_difference_max=command_difference))

    # Import Torch only when the live worker has host-memory room. No CUDA is used.
    assert psutil.virtual_memory().available > 3 * 2**30, 'Retry actor audit after host memory recovers'
    import torch
    torch.set_num_threads(1)
    transfer = json.loads((run / '002_ppo_smoke/warm_start.json').read_text())
    source_path = Path(transfer['source'])
    baseline_path = Path(transfer['baseline_checkpoint'])
    source = torch.load(source_path, map_location='cpu', weights_only=False)
    baseline = torch.load(baseline_path, map_location='cpu', weights_only=False)
    src = source['actor_state_dict']
    dst = baseline['actor_state_dict']
    keys = sorted(src)
    assert keys and any('log_std' in k for k in keys)
    assert set(src) == set(dst)
    differences = {k: float((src[k] - dst[k]).abs().max()) for k in keys}
    assert all(torch.equal(src[k], dst[k]) for k in keys), 'Transferred actor changed'
    assert not transfer['critic_loaded'] and not transfer['optimizer_loaded']
    assert not transfer['qualification_reused']
    assert baseline_path != run / '002_ppo_smoke/candidate.pt'
    value = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), run=str(run),
        passed=True, command_traces=results, ppo_command_audit=training,
        actor_transfer=dict(source=str(source_path), baseline=str(baseline_path),
            source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
            exact_actor_and_exploration_tensor_count=len(keys),
            max_differences=differences, critic_loaded=False, optimizer_loaded=False,
            qualification_reused=False),
        scope='CPU saved-artifact consistency audit. Does not qualify locomotion, contact pressure or hardware duty.')
    OUT.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
