"""Bind inspected angle/target/velocity metrics to each exact diagnostic actor."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
comparison = read(OUT / 'matched_sequence_review.json')
coverage = read(OUT / 'sequence_coverage_analysis.json')
assert comparison['completed']
records = []
for row in comparison['rows']:
    folder = OUT / f"{row['branch']}_{row['updates']:03d}_upper"
    motion = read(folder / 'upper_joint_time_audit.json')
    qa = read(folder / 'visual_inspection.json')
    for inspected in qa['plots']:
        assert inspected['visually_inspected']
        assert hashlib.sha256(Path(inspected['path']).read_bytes()).hexdigest() == inspected['sha256']
    values = [r for r in motion['records'] if r['actor_tag'] == 'latest']
    assert values and all(r['sha256'] == row['sha256'] for r in values)
    cached = next(r for r in coverage['records']
                  if r['branch'] == row['branch'] and r['updates'] == row['updates'])
    assert cached['sha256'] == row['sha256']
    records.append(dict(branch=row['branch'], updates=row['updates'],
        checkpoint=row['checkpoint'], sha256=row['sha256'],
        actual_motion=values, physical_slow_bias=cached['physical'],
        latest_training_cycle_coverage=cached['actual_command_coverage'],
        actual_new_gradient_statistics=row['actual_gradient_statistics_mean'],
        first_actual_batch=row['first_actual_batch'], last_actual_batch=row['last_actual_batch'],
        visual_inspection=qa,
        all_current_and_prior_gates={k: row[k + '_cases'] for k in ['nominal', 'pushed']},
        independent_initial_states={k: row[k] for k in ['diverse_nominal', 'diverse_push'] if k in row}))
result = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), records=records,
    scope='Exact saved actors, actual physical upper timelines, all population motion statistics, and independently seeded locomotion gates. Latest training-cycle snapshots are a subset of the training distribution, not complete population/time coverage.',
    comparison_does_not_certify_natural_motion=True,
    mapped_speed_matched_Unitree_encoder_trajectories_available=False,
    optical_gaze_calibrated=False, diagnostic_actors_deployed=False)
(OUT / 'upper_matched_comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps([dict(branch=r['branch'], updates=r['updates'],
    fast_bias=[v for v in r['physical_slow_bias'] if v['command'] == 3.5]) for r in records]))
