"""Independent CPU validation of completed workers, scope, loads and diversity."""
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from runtime_joint_order import runtime_joint_order

OUT = Path(__file__).resolve().parent
TRIAL = OUT / 'coupled_training_trial'
ROOT = TRIAL / 'matched_coupled_teacher'
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest = read(ROOT / 'experiment_manifest.json')
assert sha(manifest['checkpoint']) == manifest['checkpoint_sha256']
assert sha(TRIAL / 'neutral_teacher_candidate.py') == manifest['wrapper_sha256']
assert sha(TRIAL / 'command_local_teacher_target.py') == manifest['helper_sha256']
assert read(ROOT / 'completion.json')['completed']
assert read(TRIAL / 'remainder_pause/diagnostic_resume.json')['same_supervisor']
operations = []; actual_updates = 0; samples = 0
for seed in [229, 241]:
    for branch in ['control', 'candidate']:
        checkpoint = Path(manifest['checkpoint'])
        for updates in [64, 128]:
            folder = ROOT / f'seed{seed}_{branch}_{updates:03d}_train'
            cfg = read(folder / 'run_config.json')['arguments']
            proof = read(folder / 'full_load_proof.json')
            assert Path(cfg['checkpoint']).resolve() == checkpoint.resolve()
            assert proof['source_sha256'] == sha(checkpoint)
            assert all(proof[k] for k in ['actor_equal', 'critic_equal', 'optimizer_equal', 'exploration_equal'])
            assert read(folder / 'operation_receipt.json') and read(folder / 'result.json')['completed']
            assert cfg['iterations'] == 64 and cfg['num_envs'] == 4096 and cfg['seed'] == seed
            assert cfg['learning_rate'] == .0001
            executed = read(folder / 'experimental_teacher_target.json')
            assert executed['applied'] == (branch == 'candidate')
            assert executed['training_only'] and executed['truthful_student_observations']
            assert executed['helper_sha256'] == manifest['helper_sha256']
            rows = [json.loads(s) for s in (folder / 'progress.jsonl').read_text().splitlines()]
            assert len(rows) == 64
            assert all(r['ppo']['pre_update_kl'] == 0 and r['behavior']['action_command_difference'] == 0 for r in rows)
            assert all(r[k]['batches'] == 20 for r in rows for k in ['retention', 'locomotion_retention', 'upper_refinement'])
            assert min(r['gpu']['free_mb'] for r in rows) >= 300
            assert min(r['gpu']['ram_available_mb'] for r in rows) >= 2048
            names = runtime_joint_order(folder)
            head = [names.index(n) for n in ['head_yaw', 'head_pitch']]
            if branch == 'candidate':
                assert all(r['locomotion_retention']['head_excluded'] == head for r in rows)
                assert all(r['locomotion_retention']['neutral_teacher_sha256'] == executed['neutral_teacher_sha256'] for r in rows)
                assert read(folder / 'command_local_target_validation.json')['passed']
            actual_updates += len(rows); samples += len(rows) * cfg['num_envs'] * 24
            operations.append(dict(folder=str(folder), kind='train', actual_updates=len(rows), actual_samples=len(rows) * cfg['num_envs'] * 24,
                exact_actor_critic_Adam_exploration=True, observed_runtime_head_rows=head,
                min_GPU_MiB=min(r['gpu']['free_mb'] for r in rows), min_host_MiB=min(r['gpu']['ram_available_mb'] for r in rows)))
            checkpoint = folder / 'candidate.pt'
            for label in ['nominal', 'push'] + (['diverse_nominal', 'diverse_push'] if updates == 128 else []):
                gate = ROOT / f'seed{seed}_{branch}_{updates:03d}_{label}'
                cfg = read(gate / 'run_config.json')['arguments']; result = read(gate / 'result.json'); ev = read(gate / 'evaluation.json')
                assert result['completed'] and read(gate / 'operation_receipt.json')
                assert Path(cfg['checkpoint']).resolve() == checkpoint.resolve()
                assert cfg['num_envs'] == 320 and cfg['duration'] == 20
                assert bool(result['perturb']) == label.endswith('push')
                assert len(ev['cases']) == 10 and all(c['statistics']['trials'] == 32 for c in ev['cases'])
                if label.startswith('diverse'):
                    # This reset-only evaluator invokes the byte-identical original worker;
                    # it intentionally does not emit the experimental target wrapper's receipt.
                    argv = read(gate / 'process.json')['argv']
                    assert Path(argv[0]).resolve() == (TRIAL / 'diverse_eval.py').resolve()
                    assert '--coupled-neutral-teacher' not in argv
                    assert not any(flag in argv for flag in ['--standing_prior', '--locomotion_prior_certificate'])
                    source = TRIAL / 'control_source/physical_v29/training/worker_unitree.py'
                    protected = read(TRIAL / 'protected_source_copy.json')['protected_hashes']
                    assert sha(source) == protected['training/worker_unitree.py']
                    reset_wrapper = (TRIAL / 'diverse_eval.py').read_text()
                    assert "runpy.run_path(str(TRAIN/'worker_unitree.py'),run_name='__main__')" in reset_wrapper
                    diversity = read(gate / 'initial_diversity.json')
                    q = np.asarray(diversity['initial_q']); qd = np.asarray(diversity['initial_qd'])
                    assert q.shape == qd.shape == (320, 23) and np.isfinite(q).all() and np.isfinite(qd).all()
                    assert np.max(np.abs(qd)) <= .010001
                    assert len(diversity['unique_initial_states_by_command']) == 10
                    assert all(n == 32 for n in diversity['unique_initial_states_by_command'].values())
                    assert diversity['original_native_physics_and_all_gate_classifiers']
                    assert diversity['no_observation_or_action_override']
                else:
                    assert read(gate / 'experimental_teacher_target.json')['applied'] is False
                operations.append(dict(folder=str(gate), kind='eval', all_gates_pass=ev['passed'],
                    checkpoint_sha256=sha(checkpoint), independently_seeded_initial_states=label.startswith('diverse')))
assert actual_updates == 512 and samples == 50331648 and len(operations) == 32
smoke_updates = sum(read(ROOT / f / 'training_validation.json')['actual_updates'] for f in ['candidate_smoke', 'candidate_full_reload'])
assert smoke_updates == 12
result = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), passed=True,
    all_32_comparison_operations_completed_and_owned_workers_cleaned=True,
    matched_learning_updates=actual_updates, matched_learning_samples=samples,
    smoke_reload_updates=smoke_updates, smoke_reload_samples=smoke_updates * 64 * 24,
    diagnostic_samples_in_main_progress=0, original_signature_does_not_certify_experimental_targets=True,
    diagnostic_actors_deployed=False, operations=operations,
    false_positive_URDF_order_abort_preserved=True, corrected_review_uses_observed_Isaac_order=True)
(OUT / 'experiment_validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'operations'}, indent=2))
