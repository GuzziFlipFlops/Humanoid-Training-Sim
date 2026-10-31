import json
from pathlib import Path
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent
read = lambda name: json.loads((OUT / name).read_text())
r = read('matched_shoulder_review.json')
u = read('upper_matched_comparison.json')
assert r['completed'] and read('experiment_validation.json')['passed']
candidate = next(v for v in r['rows'] if v['branch'] == 'candidate' and v['updates'] == 128)
control = next(v for v in r['rows'] if v['branch'] == 'control' and v['updates'] == 128)
features = {v['branch']: v for v in u['rows'] if v['updates'] == 128 and v['test'] == 'nominal' and v['command'] == 3.5}
assert features['candidate']['absolute_mean_bias_deg'] > features['control']['absolute_mean_bias_deg']
assert not candidate['nominal_cases'][-1]['passed']
assert all(not candidate[k]['cases'][-1]['passed'] for k in ['diverse_nominal', 'diverse_push'])
assert any(not c['passed'] for c in candidate['nominal_cases'][:-1])
cn, cp = candidate['nominal_cases'][-1], candidate['pushed_cases'][-1]
explanation = (
    'The phase-balanced shoulder-mean candidate is rejected for immediate deployment. '
    f'At128 updates its standard3.5m/s nominal/pushed lateral speeds are{cn["lateral_speed"]:.3f}/{cp["lateral_speed"]:.3f}m/s;nominal fails and the standard pushed case passes. '
    'Both independently seeded current cases still fail direction,with lateral0.156/0.154m/s. '
    'The0.15 net-travel and1.5 geometric-flight failures persist,and nominal2.5 still slips. '
    'At64 it added2.5 slip and3m direction failures where the matched control passed. '
    f'More decisively for its intended arm goal,the representative3.5 shoulder held-bias measure is{features["candidate"]["absolute_mean_bias_deg"]:.2f}deg versus{features["control"]["absolute_mean_bias_deg"]:.2f}deg in control. '
    'The noncancelling slow RMS confirms this is a real held offset,not just an averaging artifact. '
    'Some lower-speed bias,head and128-update3m retention results improve,so the hypothesis is not declared universally ineffective. '
    'It does not demonstrate the intended current-speed upper-motion benefit with complete retention. '
    'No diagnostic actor or new gradient enters production;the existing upper refinement and original full learner continue. '
    'Actual training-bin means differ from closed-loop output means;that is a distribution/feature-response lead,not a proved runtime fault. '
    'All planned components were independently validated despite the harness final overall-time failure,and the same supervisor resumed safely.')
next_tests = [
    'Upper motion remains a priority. The new phase-balanced ensemble shoulder-mean candidate is rejected after exact64/128 controls and independent states. At128 its3.5 held-bias measure19.60deg exceeds control16.76deg,even though some slower/head/3m cases improve. Compare actual per-robot cycle and slow target means with truthful training-state coverage,hidden-feature response and per-command auxiliary/PPO gradients. The last candidate minibatch3.25-3.5band mean(-17.72,+1.08deg) differs from final exact3.5 effective outputs(-29.10,+14.63deg);these are different states/times,not a proved bug. Do not blindly raise weights/caps or repeat the same ensemble band. Any individual-cycle or state-conditioned continuation needs narrow output-row derivatives,full-state64/128 controls,all independent prior/current/head/direction/flight gates and seeded replication. Preserve forward head,legs,both teachers,35/1 gains and full learner. Inspect actual angle/target/velocity plus noncancelling slow RMS/range;do not mistake slow reversals for natural swing or fabricate Unitree encoder curves.',
    'Compare newer exact original3.5 direction and all stopped/prior gates against524 and separately certified267/349. At524 signed3.388/3.398 and lateral0.184/0.163 remain failed;0.15 net travel,1.5 geometric flight,2.5 slip and3m lateral fail both. Direction improves from518 but retention is nonmonotonic. Use actual per-command advantage/teacher/effective-target and complete objective evidence to test a different bounded retention-preserving continuation before deployment. Existing course1,velocity/cycle filters,fixed-frame kernel,lateral exposure,teacher-context removal,stiffness70 and other rejected candidates stay rejected absent new evidence. No reward-identical waiting,spoof,hybrid,clock patch,tilted-head rollback or from-scratch reset.',
    'Verify same39000,one actual worker,55 hashes,local ground,full actor/critic/Adam/exploration and both teachers. Keep27507 inherited plus completed and50 hash-proven R40 recoveries;32 unsaved R39 updates excluded. Exclude this2782.9s suspension only through each timing endpoint;525 nominal wrapper includes it. The comparison finished all components but exited1 at3008.8/2600s final guard;retain failed log and plan future diagnostic budgets separately from saved-boundary waiting/drain rather than falsely claiming success. Keep8192target,300MiBreserve,host guards/fallback and180 continuous healthy TRAINING seconds;Fusion stays closed. Current6144 steady GPU is about1.8GiB free,so do not force8192 from evaluation memory. Only qualified5 or qualified4 plus controlled native limits proving5 blocked ends training/automation;every endpoint needs all unchanged gates and32 independently seeded states/cmd. Use actual actor-specific5ms CAD motor velocity/torque and sustained driving saturation if a high-speed plateau appears;isolated elbow clips or falls are not a ceiling.'
]
decision = dict(recorded_utc=datetime.now(timezone.utc).isoformat(),decision='REJECTED for immediate deployment',
    explanation=explanation,production_change=False,diagnostic_actors_deployed=False,
    current_full_learner_preserved=True,existing_upper_refinement_continues=True,
    no_optimizer_or_exploration_reset=True,next_tests=next_tests,
    human_or_Unitree_motion_certified=False,verified_5_or_4_endpoint=False)
(OUT / 'decision.json').write_text(json.dumps(decision, indent=2) + '\n')
print(explanation)
