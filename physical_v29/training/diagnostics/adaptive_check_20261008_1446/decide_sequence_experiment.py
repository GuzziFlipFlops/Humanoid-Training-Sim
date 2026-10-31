"""Record a verdict only after every controlled component is saved and validated."""
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
review = read(OUT / 'matched_sequence_review.json')
validation = read(OUT / 'experiment_validation.json')
motion = read(OUT / 'upper_matched_comparison.json')
assert review['completed'] and validation['passed'] and len(motion['records']) == 4
control = next(r for r in review['rows'] if r['branch'] == 'control' and r['updates'] == 128)
candidate = next(r for r in review['rows'] if r['branch'] == 'candidate' and r['updates'] == 128)
assert all(control[k + '_cases'][-1]['passed'] for k in ['nominal', 'pushed'])
assert all(not candidate[k + '_cases'][-1]['passed'] for k in ['nominal', 'pushed'])
assert not candidate['diverse_nominal']['passed'] and not candidate['diverse_push']['passed']
explanation = (
    'The individual completed-cycle shoulder-mean candidate is rejected for deployment. '
    'At 64 updates it reduces fast shoulder bias but increases 3.5 m/s lateral drift to '
    '0.461/0.442 m/s versus unchanged control 0.096/0.098, and adds a nominal 1 m/s support failure. '
    'At 128 updates its drift remains 0.381/0.358 versus control 0.078/0.109; '
    'the support failure persists and retained 1.5 m/s geometric flight still fails. '
    'Nominal fast shoulder slow RMS is 16.46/4.56 degrees versus control 15.71/6.17, '
    'so the earlier improvement does not remain superior on both sides. '
    'Actual 2.5 and 3 m/s target biases still reverse slowly. Independent initial-state tests '
    'preserve the failure evidence. Keep the exact original full-state learner, existing upper '
    'refinement and all gates; no diagnostic actor or new gradient enters production. '
    'This rejects the tested continuation, not every possible cycle-based refinement.')
next_tests = [
    'Read fresh exact current-speed and every prior gate first, preserving useful actors. '
    'The matched unchanged control passes 3.5 m/s but still fails 0.15 net travel and 1.5 true flight. '
    'Prioritize their per-command actual advantages, teacher effective targets and contact/landing events. '
    'If those failures persist, test one bounded exact-command retention rehearsal on the CURRENT full state '
    'against unchanged 64/128 controls and independent training/initial-state seeds, retaining current direction, '
    'all other speeds and posture. The earlier 44% exposure, course, fixed-frame, teacher-context and other '
    'rejected variants remain rejected absent new evidence; do not change arbitrary weights or relax gates.',
    'Inspect every actual head/shoulder pitch+roll/elbow angle, target and velocity timeline, bilateral and '
    'arm-leg phase, and noncancelling slow bias. The new individual-cycle mean candidate is rejected: '
    'early fast bias reduction weakens direction/support and is not sustained on both shoulders at 128. '
    'Quantify current lower-output sensitivity to arm-history variation on saved truthful states and '
    'distinguish physical arm/base coupling from learned feedback dependence in bounded normal-limit cohorts '
    'before another shoulder derivative. Latest cycle snapshots cover only a subset of robots/states; '
    'they do not certify deterministic evaluation coverage. Preserve head, gait, full state, both teachers, '
    '35/1 gains and existing upper 20 batches/update. No production observation substitution, imposed '
    'reference curve, gain/cap increase or repeat of the same failed mean candidate without new evidence.',
    'Verify the same supervisor 39000, one actual worker, 55 frozen hashes, local ground, exact '
    'actor/critic/Adam/exploration and both teachers after diagnostic resume. Count 27,507 inherited '
    'plus fresh completed and hash-proven recovered main updates; exclude unsaved R39, capacity, smoke '
    'and diagnostic learning. Exclude all three documented suspensions through each timing endpoint. '
    'Keep the 8,192 target, 300 MiB reserve, host guards/fallback and 180 continuous healthy TRAINING '
    'seconds before ramps; Fusion stays closed. Final 5 m/s or conditional 4 m/s requires every '
    'unchanged gate and 32 independently seeded states per command, with actor-specific native 5 ms '
    'CAD speed/torque and sustained driving-saturation evidence proving 5 blocked for the 4 m/s fallback.'
]
result = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), verdict='REJECT', deploy=False,
    explanation=explanation, next_tests=next_tests, matched_control=control['checkpoint'],
    diagnostic_candidate=candidate['checkpoint'], training_seeds=[149],
    independently_seeded_evaluations=[2309,1911], production_changed=False,
    keep_original_full_state=True, no_from_scratch_restart=True,
    no_endpoint_established=True, computer_on=True)
(OUT / 'decision.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(explanation)
