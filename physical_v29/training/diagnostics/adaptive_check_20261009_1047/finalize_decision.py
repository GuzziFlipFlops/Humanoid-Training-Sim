"""Select from completed saved evidence; no runtime mutation or inference override."""
from pathlib import Path
from datetime import datetime, timezone
import json

ROOT = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
reward = read(ROOT / 'reward_selection.json')
high = read(ROOT / 'high_speed_extrapolation/analysis.json')
objective = read(ROOT / 'objective_advantage_comparison.json')
arms = read(ROOT / 'inverse_arm_response_analysis.json')
assert reward['candidate_status'] == 'REJECTED' and not reward['deploy']
assert high['completed'] and arms['completed'] and objective['completed']
assert not high['endpoint_proven'] and not high['causal_velocity_ceiling_proven']
normal = [r for r in reward['exact_two_seed_128_direction_and_prior_comparisons']
          if r['test'] in ['nominal', 'pushed']]
assert len(normal) == 4 and all(r['control_current_pass'] and not r['candidate_current_pass'] for r in normal)
high_cases = [dict(test=g['label'], command=c['command'],
                  signed_forward_m_s=c['measured_signed_forward_m_s'],
                  lateral_m_s=c['measured_absolute_lateral_m_s'],
                  survival=c['statistics']['survival'], failures=c['failures'])
              for g in high['gates'] for c in g['cases'] if c['command'] >= 3.5]
assert not all(not c['failures'] for c in high_cases if c['command'] == 5)
inverse = [r for r in arms['records'] if r['actor'] == 'current_body_teacher_arms']
fmt = lambda value, factor=1: 'unavailable' if value is None else f'{value*factor:.3f}'
arm_text = '; '.join(f"{r['name']}: signed {fmt(r['signed_forward_m_s'])} m/s, "
                    f"flight {fmt(r['genuine_flight_fraction'],100)}%, survival {r['survival']:.3f}"
                    for r in inverse)
if arms['bounded_physical_candidate_promising']:
    arm_focus = ('The inverse current-body/current-head with qualified-arm cohorts give a useful physical lead. '
                 'Test a command-local, state-conditioned arm-row transfer on truthful current states, keeping the current head and legs. '
                 'Require exact-current full-state 64/128 controls, two learning seeds and every unchanged gate before any versioned deployment. ')
else:
    arm_focus = ('The inverse current-body/current-head with qualified-arm cohorts do not meet the bounded physical screen. '
                 'Do not copy those arm rows into production. Resolve phase-specific leg targets, sole unloading and torso/arm feedback at 0.15/1.5 '
                 'using the qualified teacher and current truthful states before a coupled command-local training target. ')
next_tests = [
    ('The tests now explicitly extend to 4, 4.5 and 5 m/s, but current-policy extrapolation does not qualify those speeds. '
     'Preserve the full R40 learner and separately passing 992/605/908/917 current-speed actors, 267 all-prior and certified 349. '
     'Compare fresh exact 3.5 direction and every retained speed, especially slow tracking and 1.5 genuine/geometric flight. '
     'Freshly requalify the earlier neutral-head all-prior actor 267 in the current runtime and compare its effective targets, '
     'full physical return and compatibility with the current upper controller against qualified 349 at 0.15/1.5. '
     'Neither earlier actor supplies labels above its independently reverified certificate. '
     'Use that previous-checkpoint evidence and the new exact paired reward/critic/global-advantage evidence to select a retention-preserving continuation. '
     'The narrow instantaneous slow kernel is rejected in both learning seeds; do not repeat it, reset the learner, relax gates or simply skip retention. '
     'Any new reward must have explicit versioned provenance, finite/counterexample checks, unchanged full-state 64/128 controls and independent seeds.'),
    (arm_focus + 'Never use 349 labels above its 3 m/s certificate or deploy diagnostic composites. '
     'Every check still saves and inspects actual head, shoulder pitch/roll and elbow angles, targets, velocities, bilateral and arm-leg phase '
     'and noncancelling slow bias. Preserve forward head, both teachers, upper 20 batches/update and original 35/1 gains. '
     'No fabricated human/Unitree encoder curves, production observation substitution, imposed trajectory, hybrid or cap increase.'),
    ('Verify the same owned supervisor 39000, one actual CUDA worker, all 55 frozen hashes/53 authored files, original local ground '
     'and exact actor/critic/Adam/exploration plus teachers. Count 27,507 inherited plus completed and 50 proven R40 recoveries; '
     'exclude diagnostic/smoke/capacity samples and 32 unsaved R39 updates. Exclude both current bounded pauses only through each timing endpoint. '
     'Keep target 8,192, reserve 300 MiB, host guards/fallback and 180 continuous healthy training seconds; Fusion stays closed. '
     'The higher-speed 5 ms motor attribution excludes terminal/reset samples. Turning, gait failure and brief clips are not a causal limit. '
     'Only qualified 5 m/s, or qualified 4 m/s with controlled native limits proving 5 blocked, all gates and 32 individually seeded states per command, ends training.')
]
summary = ('The new tolerance-width slow reward is REJECTED after exact full-state 64/128 tests, two learning seeds and diverse initial states. '
           'Both original 128-update control nominal/pushed pairs pass 3.5 direction; all four corresponding candidate tests fail, '
           'with lateral speeds about 0.26-0.33 m/s versus controls about 0.12-0.13. Some slow-command gains do not safely retain fast gait; '
           '1.5 flight remains below the unchanged minimum. Fresh current-actor tests now cover 4/4.5/5 m/s with every original gate, '
           'plus 32 individually seeded states per command. No 4/5 qualification or causal actuator ceiling is established. '
           'Actual paired PPO objective captures and the current-head/body inverse arms-only physical test are preserved; '
           'neither supplies a production repair by itself. No diagnostic actor, reward, hybrid or optimizer reset is deployed.')
decision = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), candidate_status='REJECTED', deploy=False,
                explanation=summary, diagnostic_summary=summary,
                exact_reward_selection=str(ROOT/'reward_selection.json'), higher_speed_cases=high_cases,
                inverse_arm_physical_summary=arm_text, inverse_arm_physical_promising=arms['bounded_physical_candidate_promising'],
                first_PPO_pair_exact=objective['first_minibatch_paired_attribution_valid'],
                endpoint_proven=False, velocity_ceiling_proven=False, production_changed=False,
                next_tests=next_tests, original_full_learner_preserved=True, computer_on=True)
(ROOT/'decision.json').write_text(json.dumps(decision,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(dict(candidate_status=decision['candidate_status'], deploy=False,
                     inverse_arm_physical_promising=decision['inverse_arm_physical_promising'],
                     higher_speed_cases=high_cases),indent=2))
