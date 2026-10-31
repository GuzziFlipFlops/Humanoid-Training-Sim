"""Record this user-requested status review without mutating training sources."""
import json
from pathlib import Path
from datetime import datetime, timezone

here = Path(__file__).resolve().parent
e = json.loads((here/'evidence.json').read_text(encoding='utf-8'))
pair = e['gates'][-2:]
assert len(pair)==2 and pair[0]['checkpoint']==pair[1]['checkpoint']
assert pair[0]['folder'].endswith('nominal') and pair[1]['folder'].endswith('push')
rows = []
for a,b in zip(pair[0]['cases'],pair[1]['cases']):
    assert a['command']==b['command'] and a['trials']>=32 and b['trials']>=32
    rows.append(f"| {a['command']:.2f} | {a['speed']:.3f} / {b['speed']:.3f} | {'pass' if a['passed'] else 'fail'} / {'pass' if b['passed'] else 'fail'} | {', '.join(sorted(set(a['failed']+b['failed']))) or 'none'} |")
a,b = pair[0]['cases'][-1],pair[1]['cases'][-1]
assert a['command']==1.5 and b['command']==1.5 and a['passed'] and b['passed']
report = f"""# R34 user-requested status review

Evidence observed {e['observed_utc']}. The supervisor is running, stage run_150,
with {e['state']['total_training_iterations']} completed new main updates.
Current operation {Path(e['state']['operation_path']).name}; selected training count
{e['state']['next_training_environment_count']}. One actual CUDA worker and all
46 frozen hashes were independently verified at 22:42 UTC; the owned snapshot
is preserved separately. State and immutable saved test results are fresher.

The exact checkpoint {Path(pair[0]['checkpoint']).parent.name}/candidate.pt
(SHA256 {e['latest_checkpoint_sha256']}) now passes the running case in both
full 192-robot, 20-second nominal and pushed tests, 32 independent trials per
command. Signed speed is {a['speed']:.6f}/{b['speed']:.6f} m/s; geometric flight
is {a['flight_percent']:.6f}%/{b['flight_percent']:.6f}%, above the unchanged 1.5%
minimum; survival is 100% throughout both test populations. Running stance
asymmetry is {a['stance_asymmetry']:.6f}/{b['stance_asymmetry']:.6f}, below the
unchanged 0.18 maximum. Nominal is close to that boundary, so robustness is
not established by this single matched pair.

| Command m/s | Actual signed speed, nominal / pushed | Case result | Failed checks |
|---|---|---|---|
{chr(10).join(rows)}

The complete 1.5 stage has **not** qualified: 0.15 and 0.35 m/s retained commands
overspeed. Stopped, 0.65 and 1.0 cases pass both. The independently qualified
baseline through walking 1.0 remains preserved. No 2.0 or 5.0 qualification is
claimed. Previous full pairs had zero/negligible flight; this pair is new
independent running progress, not a sustained trend yet.

Decision: leave rewards, PPO settings, retention gradients, commands, physical
model, motor caps and frozen sources unchanged now. Preserve this useful flight
actor and continue the already running chain, with promotion blocked by prior
speed gates. Compare the next matched pairs for simultaneous flight and slow
tracking. If the tradeoff persists, perform a controlled per-command effective
action/target and objective/retention-gradient diagnosis before a focused repair;
do not assume a weight increase or actor restart will solve it.

Resources: the 8192 operation034 hit the 300 MiB VRAM reserve, saved
memory_recovery.pt and resumed at6144. Last completed6144 block038 achieved
35,642.8 active collection-plus-PPO samples/s and30,352.2 worker-effective
samples/s (startup/exit included), with minimum1036 MiB GPU free and4764 MiB
host available. All recorded main-block pre-update KL and action/reward command
differences are zero. Capacity guards, full-state recovery, target8192 and the
180-second healthy-memory upward ramp remain. Eval/startup free memory is not
eligible for the ramp. The three-hour automation remains active; computer on.

This is a CPU-only status audit, with no simulator launch, live source edit,
training restart or parameter adjustment. Gate qualification pertains to the
unchanged frozen physical_v29 training contract; the separately documented
full-CAD visual/collision fidelity limitations remain.
"""
(here/'REPORT.md').write_text(report,encoding='utf-8')
note = f"""\n\n## October 4, 2026, 22:45 UTC user status review

R34 remains run_150 after449 new main updates, selected6144 after saved8192
VRAM fallback. Exact checkpoint038 now passes the running case in matching
nominal039/pushed040: signed1.476/1.477m/s, true flight1.8854%/1.7569%,
100% survival and all running checks. Complete stage remains unqualified:
retained0.15/0.35 overspeed at0.262/0.455 nominal and0.263/0.451 pushed.
Stopped,0.65,1.0 pass. Parameters unchanged because this is fresh independent
running progress; preserve this actor and compare simultaneous retention/flight
in subsequent full pairs before a controlled repair. Last6144 block gives
35.64k active/30.35k worker-effective samples/s; minimumGPU1036MiB,host4764MiB,
zero pre-update KL and command mismatch. CPU-only status evidence/report:
E:/Aditya/robot_sim/physical_v29/training/diagnostics/status_review_20261004_2238/REPORT.md.
"""
for path in [here.parents[1]/'ITERATION_HISTORY.md',
             Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    with path.open('a',encoding='utf-8') as f: f.write(note)
next_path = here.parents[1]/'automation/NEXT_CHECK.json'
handoff = json.loads(next_path.read_text(encoding='utf-8-sig'))
handoff['user_status_review'] = {'recorded_utc':datetime.now(timezone.utc).isoformat(),
    'report':str(here/'REPORT.md'),'checkpoint':pair[0]['checkpoint'],
    'checkpoint_sha256':e['latest_checkpoint_sha256'],
    'main_updates':449,'running_case_nominal_pushed_pass':True,
    'whole_stage_qualified':False,'failed_prior_commands':[0.15,0.35],
    'parameter_decision':'Unchanged: preserve new flight actor and compare subsequent matched prior-speed and running pairs; controlled diagnostics if the tradeoff persists.',
    'next_focus':['Preserve checkpoint038: both running cases pass, slow walking fails.',
      'Compare next matched pair for simultaneous flight, stance balance and 0.15/0.35 speed tracking.',
      'Keep6144 fallback,300MiB reserve and180-second healthy ramp; no arbitrary reward restart.']}
next_path.write_text(json.dumps(handoff,indent=2),encoding='utf-8')
print(json.dumps({'report':str(here/'REPORT.md'),'matched_checkpoint':pair[0]['checkpoint'],
                  'sha256':e['latest_checkpoint_sha256'],'parameters_changed':False}))
