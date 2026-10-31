"""Read-only training review and chronological recommendation; no process control."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
TRAIN = OUT.parents[1]
RUN = TRAIN / 'runs/locomotion_20261007_r40_goal_budget_full_state'
read = lambda path: json.loads(Path(path).read_text(encoding='utf-8-sig'))
state = read(RUN / 'state.json')
folders = ['1140_run_350_gate_378_nominal', '1141_run_350_gate_378_push']
tests = []
checkpoint = None
for name in folders:
    folder = RUN / name
    assert read(folder / 'result.json')['completed']
    cp = Path(read(folder / 'run_config.json')['arguments']['checkpoint'])
    if checkpoint is not None:
        assert cp == checkpoint
    checkpoint = cp
    evaluation = read(folder / 'evaluation.json')
    tests.append(dict(folder=str(folder), all_pass=evaluation['passed'],
                      cases=[dict(command=c['statistics']['command_m_s'], passed=c['passed'],
                                  failures=c['failures'], signed_speed=c['statistics']['reset_frame_forward_speed_m_s'],
                                  lateral_speed=c['statistics']['reset_frame_lateral_speed_m_s'],
                                  true_flight=c['statistics']['geometric_flight_fraction'],
                                  trials=c['statistics']['trials']) for c in evaluation['cases']]))
assert all(not t['all_pass'] for t in tests)
assert all(next(c['passed'] for c in t['cases'] if c['command'] == 3.5) for t in tests)
assert all([c['command'] for c in t['cases'] if not c['passed']] == [0.15, 1.5] for t in tests)
now = datetime.now(timezone.utc).isoformat()
record = dict(recorded_utc=now, trigger='User asks why training remains at 3.5 m/s and whether to end it.',
              run=str(RUN), supervisor_pid=state['supervisor_pid'], status=state['status'],
              stage=state['stage'], current_operation=state['operation_path'],
              current_checkpoint=state['checkpoint'], next_training_environment_count=state['next_training_environment_count'],
              total_training_iterations_field=state['total_training_iterations'],
              exact_reviewed_checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              tests=tests, frozen_supervisor_sha256=hashlib.sha256((RUN/'supervisor_unitree.py').read_bytes()).hexdigest(),
              supervisor_promotion_rule='All nominal/pushed evaluation reports must pass, including earlier commands, before advancing; failed retention preserves trusted policy and continues the unqualified candidate.',
              recommendation='Pause/end the unchanged production iteration, preserve complete learned state and evidence, and use a bounded earlier-neutral-head-checkpoint comparison before further prolonged training. Test a revised reward/retention objective separately if warranted.',
              uncertainty='The specific learning cause remains unresolved. Retention and direction failures are reproduced; no native locomotion actuator ceiling is proved.',
              higher_speed_evidence=str(TRAIN/'diagnostics/adaptive_check_20261009_1047/high_speed_extrapolation/analysis.json'),
              previous_controlled_comparison=str(TRAIN/'diagnostics/adaptive_check_20261009_2048/REPORT.md'),
              production_change=False, checkpoint_reset=False, training_paused=False, automation_paused=False,
              authorization_scope='User asked for a recommendation, not an explicit stop. No process or automation action taken.')
(OUT/'evidence.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n', encoding='utf-8')
report = f'''# Review of the 3.5 m/s plateau

Recorded {now}. The user asks whether to end this iteration.

The current supervisor remains running at the 3.5 m/s curriculum stage. Its source advances only when both full evaluation reports pass. The stage list already includes 4, 4.5 and 5 m/s. Failure of any earlier command keeps the candidate in the same stage; more iterations do not by themselves permit promotion.

Exact checkpoint **1139_run_350_train_377**, SHA **{record['checkpoint_sha256']}**, has matching completed nominal 1140 and pushed 1141 tests, 320 robots for 20 seconds and 32 robots per command. Both current 3.5 m/s cases pass: signed forward speeds **3.4506/3.4610 m/s**, lateral speeds **0.1230/0.1106 m/s**. Every other case passes except 0.15 m/s forward net travel and 1.5 m/s geometric flight. Both full reports therefore fail. The ordinary nominal resets are not independently seeded final-endpoint evidence. These gates do not certify newer partial learning or arm motion; the earlier full upper timeline review remains bound to actor 1115.

This is a reproduced failure to retain the full skill set, with intermittent direction regression in earlier matching pairs. The exact learning cause is not established. Earlier controlled reward, teacher, arm and sampling candidates did not preserve the whole contract and were rejected; no arbitrary replacement reward is deployed. The separate 4/4.5/5 m/s extrapolation tests fail gait/direction and prove no motor ceiling. Do not treat 3.5 m/s as a hardware maximum or successful 4 m/s fallback.

Recommendation: pause/end this unchanged production iteration, save its complete learned state and independent evidence, and compare the earlier neutral-head actor 267's complete state against current-state control in bounded tests. A revised reward/retention objective should be a separately versioned, controlled test if warranted. Preserve all useful actors, head/gait/teachers and native limits. Earlier-state CPU preparation is complete; its runtime continuation is not reported as executed.

No policy, reward, gain, cap, actor/critic/Adam/exploration or process change is made in this review. Training and the automation remain active because the user asked for advice, not an explicit stop. The recommendation is not achievement of the 5 m/s or conditional 4 m/s endpoint.

Sources: [exact evidence](evidence.json), [prior controlled comparisons](../adaptive_check_20261009_2048/REPORT.md), [higher-speed physical diagnostic](../adaptive_check_20261009_1047/high_speed_extrapolation/analysis.json), and the preserved supervisor at `{RUN.as_posix()}/supervisor_unitree.py`, promotion loop beginning at line 725.
'''
(OUT/'REPORT.md').write_text(report, encoding='utf-8')
history = TRAIN / 'ITERATION_HISTORY.md'
marker = '<!-- plateau_recommendation_20261010_0059 -->'
text = history.read_text(encoding='utf-8-sig')
if marker not in text:
    entry = f'''\n\n{marker}
## Plateau recommendation — {now}

User asks why 3.5 m/s is not exceeded and whether to end training. Fresh exact actor 1139 (SHA {record['checkpoint_sha256']}) passes both current cases at 3.4506/3.4610 m/s but fails both retained 0.15 m/s net-travel and 1.5 m/s geometric-flight cases. Frozen supervisor blocks promotion until every current/prior case passes; 4/4.5/5 stages exist. Recommend pausing/ending the unchanged production iteration while preserving full state, then a bounded earlier-neutral-head full-state comparison and separately justified objective revision. Specific learning cause unresolved; no motor ceiling or successful endpoint inferred. No source/reward/physics/native limit/learned-state changes, pause or automation change performed: this is a recommendation in response to a question. Existing learner remains active. [Review]({OUT.as_posix()}/REPORT.md).\n'''
    assert history.read_text(encoding='utf-8-sig') == text
    history.write_text(text+entry, encoding='utf-8')
print(json.dumps(dict(recorded_utc=now, checkpoint=str(checkpoint), failures=[[(c['command'],c['failures']) for c in t['cases'] if not c['passed']] for t in tests], recommendation=record['recommendation'], training_paused=False), indent=2))
