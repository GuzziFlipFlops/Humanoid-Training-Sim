"""Independently validate all loads, samples, gates, source bytes and diversity."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from runtime_joint_order import runtime_joint_order
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_full_state'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=read(ROOT/'experiment_manifest.json')
assert read(ROOT/'completion.json')['completed'] and read(OUT/'diagnostic_resume.json')['same_supervisor']
assert sha(OUT/'worker_entry.py')==manifest['wrapper_sha256'] and sha(OUT/'diverse_eval.py')==manifest['diverse_eval_sha256']
protected=manifest['protected_source_copy']['protected_hashes']
assert all(sha(OUT/'control_source/physical_v29'/name)==digest for name,digest in protected.items())
assert manifest['whole_actor_critic_Adam_exploration_resumed'] and manifest['reward_physics_caps_observations_commands_PPO_and_auxiliaries_unchanged']
anchors=manifest['branches'];assert all(sha(v['checkpoint'])==v['sha256'] for v in anchors.values())
operations=[];actual_updates=0;samples=0
for seed in [317,331]:
    for branch in ['control','earlier']:
        checkpoint=Path(anchors[branch]['checkpoint'])
        for updates in [64,128]:
            folder=ROOT/f'seed{seed}_{branch}_{updates:03d}_train'
            cfg=read(folder/'run_config.json')['arguments'];proof=read(folder/'full_load_proof.json')
            assert Path(cfg['checkpoint']).resolve()==checkpoint.resolve() and proof['source_sha256']==sha(checkpoint)
            assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
            assert read(folder/'result.json')['completed'] and read(folder/'training_validation.json')['passed']
            argv=read(folder/'process.json')['argv'];assert Path(argv[0]).resolve()==(OUT/'worker_entry.py').resolve()
            assert cfg['iterations']==64 and cfg['num_envs']==4096 and cfg['seed']==seed and cfg['learning_rate']==.0001
            assert '--coupled-neutral-teacher' not in argv and '--final-row-route' not in argv
            assert not (folder/'experimental_teacher_target.json').exists()
            rows=[json.loads(x) for x in (folder/'progress.jsonl').read_text().splitlines()];assert len(rows)==64
            assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
            assert all(r[k]['batches']==20 for r in rows for k in ['retention','locomotion_retention','upper_refinement'])
            assert min(r['gpu']['free_mb'] for r in rows)>=300 and min(r['gpu']['ram_available_mb'] for r in rows)>=2048
            names=runtime_joint_order(folder);assert [names.index(k) for k in ['head_yaw','head_pitch']]==[11,16]
            actual_updates+=len(rows);samples+=sum(r['num_envs']*24 for r in rows)
            operations.append(dict(folder=str(folder),kind='train',source_sha256=sha(checkpoint),exact_full_state=True,updates=64))
            checkpoint=folder/'candidate.pt'
            for label in ['nominal','push']+(['diverse_nominal','diverse_push'] if updates==128 else []):
                gate=ROOT/f'seed{seed}_{branch}_{updates:03d}_{label}';cfg=read(gate/'run_config.json')['arguments']
                assert read(gate/'result.json')['completed'] and read(gate/'operation_receipt.json')
                assert Path(cfg['checkpoint']).resolve()==checkpoint.resolve() and cfg['num_envs']==320 and cfg['duration']==20
                ev=read(gate/'evaluation.json');assert len(ev['cases'])==10 and all(c['statistics']['trials']==32 for c in ev['cases'])
                assert bool(read(gate/'result.json')['perturb'])==label.endswith('push')
                if label.startswith('diverse'):
                    d=read(gate/'initial_diversity.json');q=np.asarray(d['initial_q']);qd=np.asarray(d['initial_qd'])
                    assert q.shape==qd.shape==(320,23) and np.isfinite(q).all() and np.isfinite(qd).all()
                    assert np.max(np.abs(qd))<=.010001
                    assert len(d['unique_initial_states_by_command'])==10 and all(n==32 for n in d['unique_initial_states_by_command'].values())
                    assert d['original_native_physics_and_all_gate_classifiers'] and d['no_observation_or_action_override']
                operations.append(dict(folder=str(gate),kind='eval',all_pass=ev['passed'],sha256=sha(checkpoint),independent_states=label.startswith('diverse')))
assert actual_updates==512 and samples==50331648 and len(operations)==32
smoke=sum(read(ROOT/(b+'_'+kind)/'training_validation.json')['actual_updates'] for b in ['control','earlier'] for kind in ['smoke','reload'])
assert smoke==16
assert read(ROOT/'current_baseline_parity.json')['passed']
assert read(OUT/'full_baseline_parity.json')['passed']
assert read(ROOT/'earlier_baseline_review.json')['all_prior_through3_pass']
result=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),passed=True,all_32_comparison_operations_complete=True,
    matched_learning_updates=actual_updates,matched_learning_samples=samples,smoke_reload_updates=smoke,smoke_reload_samples=smoke*64*24,
    diagnostic_samples_in_main_progress=0,protected_authored_source_files=len(protected),no_runtime_objective_override=True,
    original_main_learner_resumed=True,diagnostic_actor_deployed=False,
    bitwise_replayed_basic_and_synchronous_arrays=read(OUT/'full_baseline_parity.json')['array_count'],operations=operations)
(OUT/'experiment_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='operations'},indent=2))
