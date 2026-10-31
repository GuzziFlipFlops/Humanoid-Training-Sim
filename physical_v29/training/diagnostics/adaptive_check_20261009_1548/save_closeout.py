"""Reconcile the newer saved pair without transferring earlier upper/motor QA."""
import hashlib
import importlib.util
import json
from pathlib import Path
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent
TRAIN = OUT.parents[1]
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
write = lambda p, value: Path(p).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
v = read(OUT / 'closeout_learning_verification.json')
live = read(OUT / 'closeout_cpu/live_handoff.json')
pair = read(OUT / 'closeout_gate_review.json')['latest']
prior_pair = read(OUT / 'closing_gate_review.json')['latest']
assert v['supervisor_pid'] == 39000 and len(v['actual_worker_pids']) == 1
assert v['frozen_files_verified'] == 55 and v['protected_authored_worker_files_equal'] == 53
assert v['actual_retained_main_updates'] == 22617 and v['inherited_actual_main_updates'] == 27507
assert Path(pair['checkpoint']).parent.name == '1064_run_350_train_352'
assert hashlib.sha256(Path(pair['checkpoint']).read_bytes()).hexdigest() == pair['sha256']
assert all(c['trials'] == 32 and c['survival'] == 1.0 for test in ['nominal', 'pushed'] for c in pair[test + '_cases'])
current = [next(c for c in pair[t + '_cases'] if c['command'] == 3.5) for t in ['nominal', 'pushed']]
assert all(c['passed'] for c in current)
assert any(not c['passed'] for c in pair['nominal_cases'][:-1])
flight = [next(c for c in pair[t + '_cases'] if c['command'] == 1.5)['flight'] for t in ['nominal', 'pushed']]
summary = (
    f"Closing newer observation {v['owned_snapshot_utc']}: the same R40 supervisor 39000 has one actual worker and all 55 frozen hashes/53 protected files. "
    f"It retains {v['completed_main_updates']:,} completed plus 50 recovered = {v['actual_retained_main_updates']:,} current-run main updates, plus 27,507 inherited; {v['observed_partial_main_updates']} observed partial updates in 1067 remain separate. "
    f"Exact actor 1064, SHA {pair['sha256']}, passes both 3.5 m/s cases at {current[0]['signed_speed']:.3f}/{current[1]['signed_speed']:.3f} signed and {current[0]['lateral_speed']:.3f}/{current[1]['lateral_speed']:.3f} lateral m/s. "
    f"Both retained 0.15 net travel, 1.5 geometric flight and 2.5 foot-slip tests still fail; 1.5 genuine flight is {100*flight[0]:.3f}%/{100*flight[1]:.3f}%. Every other saved case passes and all survive. "
    "The stage remains unqualified. Full all-command upper plots and 10 Hz CAD motor/COM-contact audits stay bound to 1061, not 1064. This check’s native 5 ms attribution stays bound to the earlier 0.15/1.5 physical cohorts. "
    "Original 1064 loads exact 1061 actor/critic/Adam/exploration, zero KL/command mismatch and all three auxiliaries at 20 batches/update; current 1067 preserves exact 1064 loading. No experimental actor, target, reward, gain, cap or optimizer reset is deployed. Same active five-hour endpoint; computer on."
)
assert v['partial_full_load_proof'] and all(v['partial_full_load_proof'][k] for k in ['actor_equal', 'critic_equal', 'optimizer_equal', 'exploration_equal'])
record = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), summary=summary,
    newer_matching_pair=pair, source_snapshot=str(OUT / 'closeout_cpu/live_handoff.json'),
    main_verification=str(OUT / 'closeout_learning_verification.json'),
    upper_and_population_motor_checkpoint=prior_pair['checkpoint'], upper_and_population_motor_sha256=prior_pair['sha256'],
    newer_posture_certified=False, production_change=False, endpoint=False)
write(OUT / 'closing_newer_pair.json', record)
evidence = read(OUT / 'evidence.json')
evidence.update(closing_newer_observation=record, closing_summary=summary)
write(OUT / 'evidence.json', evidence)
handoff = read(TRAIN / 'automation/NEXT_CHECK.json')
handoff.update(recorded_utc=record['recorded_utc'], closing_summary=summary, latest_observation=v['owned_snapshot_utc'],
    latest_liveness_utc=v['owned_snapshot_utc'], latest_liveness=record['source_snapshot'],
    current_owned_operation=v['operation_path'], current_checkpoint=live['state']['checkpoint'],
    current_checkpoint_sha256=hashlib.sha256(Path(live['state']['checkpoint']).read_bytes()).hexdigest(),
    selected_actor=pair['checkpoint'], selected_actor_sha256=pair['sha256'], latest_matching_pair=pair,
    latest_matching_pair_file=str(OUT / 'closing_newer_pair.json'), partial_observed_main_updates=v['observed_partial_main_updates'],
    resumed_learning_evidence=record['main_verification'], final_verification=record['main_verification'],
    reviewed_speed_case_passes_both=True, newer_posture_certified=False)
handoff['focused_tests'][0] = handoff['focused_tests'][0].replace('actors 1058/1061', 'actors 1058/1061/1064')
handoff['next_tests'] = handoff['focus'] = handoff['focused_tests']
write(TRAIN / 'automation/NEXT_CHECK.json', handoff)
paragraph = ('\n\n<!-- adaptive_check_20261009_1548_closeout -->\n\n' + summary +
             '\n\nRead `closing_newer_pair.json` and `closeout_learning_verification.json` beside the dated report. Upper timeline QA remains explicitly bound to 1061.\n')
spec = importlib.util.spec_from_file_location('saved_record_paths', OUT / 'write_records.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for index, path in enumerate(module.record_paths()):
    old = path.read_text(encoding='utf-8')
    assert 'adaptive_check_20261009_1548_closeout' not in old
    path.write_text(old + paragraph if index < 3 else paragraph + '\n' + old, encoding='utf-8')
report = (OUT / 'REPORT.md').read_text(encoding='utf-8')
assert 'adaptive_check_20261009_1548_closeout' not in report
(OUT / 'REPORT.md').write_text(report + paragraph, encoding='utf-8')
args = read(OUT / 'automation_update_args.json')
before_read = '\n\nRead diagnostics/adaptive_check_20261009_1548/REPORT.md'
assert before_read in args['prompt']
args['prompt'] = args['prompt'].replace(before_read, '\n\n' + summary +
    '\nRead diagnostics/adaptive_check_20261009_1548/closing_newer_pair.json and closeout_learning_verification.json; upper QA remains 1061.' + before_read)
args['prompt'] = args['prompt'].replace('actors 1058/1061,', 'actors 1058/1061/1064,')
write(OUT / 'automation_update_args.json', args)
print(json.dumps(dict(closeout_saved=True, current_case_passes_both=True, stage_qualified=False,
    upper_QA_actor='1061', newest_gate_actor='1064', retained_main=v['actual_retained_main_updates'],
    partial=v['observed_partial_main_updates']), indent=2))
