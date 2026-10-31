"""Close CPU records and verify the native heartbeat without changing training."""
import argparse
import hashlib
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
TRAIN = OUT.parents[1]
ROOT = TRAIN.parents[1]
AUTOMATION = Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml')
MARKER = 'adaptive_check_20261009_1047_record'
PREFIX_HASH = '46f1711b1bce88727bc847e7985a95992e127272ea59cc70e4174507927817c3'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def records():
    return [TRAIN/'ITERATION_HISTORY.md', ROOT/'docs/robot/TRAINING_JOURNAL_20261001.md',
        Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md'),
        Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),
        Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]

def case(pair, test, command):
    return next(c for c in pair[test+'_cases'] if abs(c['command']-command)<1e-8)

def prepare():
    handoff = read(TRAIN/'automation/NEXT_CHECK.json')
    old_summary = handoff['summary']
    v = read(OUT/'closeout_learning_verification.json')
    live = read(OUT/'closeout_cpu/live_handoff.json')
    pair = read(OUT/'closeout_gate_review.json')['latest']
    assert v['supervisor_pid']==39000 and len(v['actual_worker_pids'])==1
    assert v['frozen_files_verified']==55 and v['protected_authored_worker_files_equal']==53
    assert v['inherited_actual_main_updates']==27507 and v['hash_proven_recovered_main_updates']==50
    assert sha(pair['checkpoint'])==pair['sha256']
    assert all(c['trials']==32 and c['survival']==1 for t in ['nominal','pushed'] for c in pair[t+'_cases'])
    assert all(not case(pair,t,0.15)['passed'] and not case(pair,t,1.5)['passed'] for t in ['nominal','pushed'])
    a,b = [case(pair,t,3.5) for t in ['nominal','pushed']]
    stamp = datetime.fromisoformat(v['owned_snapshot_utc']).astimezone(timezone.utc).isoformat()
    block = v['blocks'][-1]
    assert block['updates']==64 and block['environments']==8192
    assert block['all_preupdate_KL_and_command_mismatch_zero'] and block['three_auxiliaries_batches_per_update']==20
    proof=block['full_load_proof']
    assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
    actor=Path(pair['checkpoint']).parent.name
    fast_status='passes both current cases' if a['passed'] and b['passed'] else 'still fails at least one current case'
    summary=(
        f'Closing owned observation {stamp}: the same R40 full-state learner and supervisor 39000 are running, with one actual worker, '
        f'55 verified frozen hashes and 53 unchanged authored worker files. It retains {v["completed_main_updates"]:,} completed plus '
        f'50 hash-proven recovered = {v["actual_retained_main_updates"]:,} current-run main updates, plus 27,507 inherited; '
        f'{v["observed_partial_main_updates"]} observed partial updates remain separate. Exact actor {actor}, SHA {pair["sha256"]}, '
        f'{fast_status} at command 3.5: nominal/pushed signed speed {a["signed_speed"]:.3f}/{b["signed_speed"]:.3f} m/s, '
        f'lateral speed {a["lateral_speed"]:.3f}/{b["lateral_speed"]:.3f} m/s. The complete stage remains unqualified: '
        '0.15 m/s path efficiency and 1.5 m/s geometric flight fail both matching tests. Every other stopped/prior case passes both. '
        'Saved gates apply only to this actor. Full all-command upper angle/target/velocity and bias QA remains explicitly bound to actor 1004; '
        'no newer posture or natural-motion certification is inferred. Its head RMS is 3.10 degrees, with held arm biases and unequal motion.\n\n'
        'The tolerance-width slow reward is REJECTED after exact full-state 64/128-update comparisons, two learning seeds and independently generated '
        'initial states. Both ordinary 128-update control nominal/pushed pairs pass 3.5 m/s direction; all four candidate cases fail, with '
        '0.26–0.33 m/s lateral speed versus 0.12–0.13 for controls. Explicit current-actor 4, 4.5 and 5 m/s tests fail gait/direction. '
        'The actor-specific 5 ms motor probe has zero driving-speed clips, so those failures do not establish a velocity ceiling or qualify the 4 m/s fallback. '
        'Paired PPO captures reproduce a local cross-command gradient change through global advantage normalization, even with identical fast-command returns. '
        'This is a research lead, not proof of the full cause of long-term regression. The inverse current-head/body plus qualified-arms physical test also '
        'fails the bounded screen. No diagnostic actor, composite, reward or optimizer reset entered production.\n\n'
        'The original supervisor resumed after the 5,449.8-second reward comparison and 1,028.5-second higher-speed diagnosis. '
        'Genuine main blocks load exact actor, critic, Adam and exploration; both teachers and upper refinement remain at 20 batches per update. '
        f'The latest completed 64-update block at 8,192 environments keeps at least {block["minimum_free_gpu_MiB"]} MiB GPU and '
        f'{block["minimum_host_available_MiB"]} MiB host memory free. Retained active/worker-effective/whole-chain rates are '
        f'{v["active_PPO_samples_s"]/1000:.2f}k/{v["worker_effective_samples_s"]/1000:.2f}k/{v["whole_chain_samples_s"]/1000:.2f}k samples/s; '
        'the denominators differ and documented suspensions are excluded through each endpoint. Original 35/1 gains, truthful 400-input/23-output inference, '
        'native CAD caps and iteration-4 execution remain. The CPU numeric-order reporting repair passes five checks and does not alter frozen training sources. '
        'Keep the same active five-hour heartbeat and verified 5 m/s or qualified 4 m/s plus causal-limit endpoint. Fusion stays closed, queued work and videos '
        'are preserved, and the computer stays on.'
    )
    compact=dict(recorded_utc=stamp,supervisor_pid=v['supervisor_pid'],actual_worker_pids=v['actual_worker_pids'],
        frozen_files_verified=55,protected_authored_worker_files_equal=53,completed_main_updates=v['completed_main_updates'],
        recovered_main_updates=50,retained_main_updates=v['actual_retained_main_updates'],inherited_main_updates=27507,
        observed_partial_updates=v['observed_partial_main_updates'],operation=v['operation_path'],checkpoint=pair['checkpoint'],sha256=pair['sha256'],
        cases={t:[{k:c[k] for k in ['command','passed','failures','trials','survival','signed_speed','lateral_speed','flight']} for c in pair[t+'_cases']]
            for t in ['nominal','pushed']},upper_QA_checkpoint=handoff['mandatory_all_speed_trace_checkpoint'],
        last_completed_main=block,gpu=v['gpu'],host=v['system_memory'],computer_on=True,endpoint=False)
    write(OUT/'closeout_compact.json',compact)
    for p in records():
        text=p.read_text(encoding='utf-8')
        assert text.count(f'<!-- {MARKER} -->')==1 and text.count(old_summary)==1
        p.write_text(text.replace(old_summary,summary,1),encoding='utf-8')
    evidence=read(OUT/'evidence.json')
    evidence.update(summary=summary,closeout_observation=str(OUT/'closeout_compact.json'),
        closeout_learning_verification=str(OUT/'closeout_learning_verification.json'),
        closeout_owned_snapshot=str(OUT/'closeout_cpu/live_handoff.json'),closeout_pair=pair,
        original_full_report_measurement_endpoint=evidence['owned_snapshot'])
    write(OUT/'evidence.json',evidence)
    handoff.update(summary=summary,closing_summary=summary,latest_observation=stamp,latest_liveness_utc=stamp,
        latest_liveness=str(OUT/'closeout_cpu/live_handoff.json'),latest_closeout=str(OUT/'closeout_compact.json'),
        completed_main_updates=v['completed_main_updates'],new_main_updates=v['completed_main_updates'],
        actual_retained_current_run_main_updates=v['actual_retained_main_updates'],partial_observed_main_updates=v['observed_partial_main_updates'],
        current_owned_operation=v['operation_path'],current_checkpoint=live['state']['checkpoint'],current_checkpoint_sha256=sha(live['state']['checkpoint']),
        latest_matching_pair=pair,selected_actor=pair['checkpoint'],selected_actor_sha256=pair['sha256'],
        reviewed_speed_case_passes_both=bool(a['passed'] and b['passed']),current_stage_qualified=False,
        fully_upper_reviewed_pair=str(OUT/'closing_gate_review.json'),closeout_pair_upper_QA_available=False,
        closeout_learning_verification=str(OUT/'closeout_learning_verification.json'),
        final_verification=str(OUT/'closeout_learning_verification.json'),
        retained_main_throughput={k:v[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s','documented_diagnostic_pause_seconds_excluded']},
        resources=dict(last_completed_main=block,target_environments=8192,reserve_MiB=300,healthy_training_ramp_seconds=180,evaluation_memory_is_not_capacity=True))
    write(TRAIN/'automation/NEXT_CHECK.json',handoff)
    report=OUT/'REPORT.md'
    text=report.read_text(encoding='utf-8').replace(
        'The user correctly identified the prolonged failure to qualify beyond3.5m/s.',
        'Training qualification has stalled at 3.5 m/s.')
    closing=(f'\n## Closing owned observation\n\n{summary}\n\n'
        '[Exact later gate and ownership data](closeout_compact.json), [later full-state verification](closeout_learning_verification.json) '
        'and [earlier fully inspected upper plots](final_upper/upper_joint_time_audit.json) have separate actor scopes. '
        'The original report tables and motor experiments retain their pinned checkpoint identities.\n')
    assert '## Closing owned observation' not in text
    report.write_text(text+closing,encoding='utf-8')
    args=read(OUT/'automation_update_args.json')
    prefix=args['prompt'].split('Latest dated five-hour check',1)[0]
    assert hashlib.sha256(prefix.encode()).hexdigest()==PREFIX_HASH
    args['prompt']=(prefix+'Latest dated five-hour check October 9, 10:47 UTC. '+summary+
        '\n\nRead diagnostics/adaptive_check_20261009_1047/REPORT.md, decision.json, matched_reward_review.json, experiment_validation.json, '
        'objective_advantage_comparison.json, takeoff_phase_comparison.json, inverse_arm_response_analysis.json, high_speed_extrapolation/analysis.json, '
        'final_upper plots/data and closeout_compact.json, then fresh ACTIVE_RUN and state.\n\nNext focused tests:\n'+
        '\n'.join(f'{i+1}. {s}' for i,s in enumerate(handoff['focused_tests']))+
        '\n\nPreserve the same active five-hour heartbeat, thread, notification preference and full neutral-head learner. '
        'Only verified 5 m/s, or qualified 4 m/s with causal native limits blocking 5, ends training. At that endpoint save exact evidence, '
        'pause training and automation, prevent restart and brief the user with the computer on.')
    write(OUT/'automation_update_args.json',args)
    print(json.dumps(dict(actor=actor,fast_status=fast_status,retained_updates=v['actual_retained_main_updates'],
        current_signed=[a['signed_speed'],b['signed_speed']],current_lateral=[a['lateral_speed'],b['lateral_speed']],
        failures={t:[dict(command=c['command'],failures=c['failures']) for c in pair[t+'_cases'] if not c['passed']] for t in ['nominal','pushed']}),indent=2))

def verify():
    before=tomllib.loads((OUT/'previous_automation.toml').read_text(encoding='utf-8-sig'))
    current=tomllib.loads(AUTOMATION.read_text(encoding='utf-8-sig'))
    args=read(OUT/'automation_update_args.json')
    for k in ['id','kind','name','rrule','target_thread_id','created_at']:
        assert current[k]==before[k],k
    assert current['status']=='ACTIVE' and current['prompt']==args['prompt']
    prefix=current['prompt'].split('Latest dated five-hour check',1)[0]
    assert hashlib.sha256(prefix.encode()).hexdigest()==PREFIX_HASH
    optional=[k for k in set(before)|set(current) if 'notification' in k or k in ['model','reasoning_effort','thinking']]
    assert all(current.get(k)==before.get(k) for k in optional)
    inventory=read(TRAIN/'history/RUN_INVENTORY.json')
    rows=inventory['physical_runs']
    differing=sum(p['actual_sha256'] is not None and not p['matches'] for r in rows for p in r['frozen_source_verification'])
    missing=sum(p['actual_sha256'] is None for r in rows for p in r['frozen_source_verification'])
    assert len(rows)==40 and differing==0 and not inventory['read_warnings']
    assert all(p.read_text(encoding='utf-8').count(f'<!-- {MARKER} -->')==1 for p in records())
    receipt=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),automation_id=current['id'],status=current['status'],
        same_thread=True,same_five_hour_cadence=True,durable_prompt_prefix_sha256=PREFIX_HASH,prompt_matches_exact_update=True,
        notification_and_model_preferences_preserved=True,record_marker_once_in_all_five_files=True,
        physical_runs=len(rows),physical_operations=sum(len(r['operations']) for r in rows),
        differing_hashes=differing,missing_historical_manifest_artifacts=missing,read_warnings=inventory['read_warnings'],
        inventory_generated_utc=inventory['generated_utc'],computer_on=True)
    write(OUT/'automation_inventory_verification.json',receipt)
    handoff=read(TRAIN/'automation/NEXT_CHECK.json')
    handoff.update(automation_verified_active=True,automation_verification=str(OUT/'automation_inventory_verification.json'),
        inventory_verification=str(OUT/'automation_inventory_verification.json'),history_inventory=str(TRAIN/'history/RUN_INVENTORY.json'))
    write(TRAIN/'automation/NEXT_CHECK.json',handoff)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','verify'],required=True);args=p.parse_args()
    {'prepare':prepare,'verify':verify}[args.mode]()
