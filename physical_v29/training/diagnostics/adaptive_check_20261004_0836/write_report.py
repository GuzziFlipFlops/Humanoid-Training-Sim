"""Build the dated report and focused next handoff from fresh saved evidence."""
import datetime
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
HERE=ROOT/'physical_v29/training'
OUT=Path(__file__).parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
e=read(OUT/'evidence.json');live=e['live_state'];run=Path(e['active_pointer']['run'])
old=e['prior_state_summary'];blocks=e['prior_training_blocks'][-6:]
nominal=next(g for g in reversed(e['prior_gates']) if g['folder'].endswith('_nominal'))
pushed=next(g for g in reversed(e['prior_gates']) if g['folder'].endswith('_push'))
table='| Command | Nominal measured forward | Pushed measured forward | Nominal failures | Pushed failures |\n| --- | --- | --- | --- | --- |\n'
for a,b in zip(nominal['cases'],pushed['cases']):
    table+=f"| {a['command']:.2f} m/s | {a['signed_forward_m_s']:.3f} m/s | {b['signed_forward_m_s']:.3f} m/s | {', '.join(a['failures']) or 'pass'} | {', '.join(b['failures']) or 'pass'} |\n"
capacity=read(run/'benchmarks.json') if (run/'benchmarks.json').exists() else []
smoke=run/'002_resource_ppo_smoke'
rows=[json.loads(x) for x in (smoke/'progress.jsonl').read_text().splitlines()]
load=read(OUT/'full_checkpoint_load.json')
parity=read(OUT/'evaluation_parity.json') if (OUT/'evaluation_parity.json').exists() else None
parity_summary=None if parity is None else dict(passed=parity['passed'],independent_cases_identical=parity['independent_cases_identical'],
    files={f['name']:dict(all_arrays_equal=f['all_arrays_equal'],arrays=len(f['arrays'])) for f in parity['files']})
recorded=e['recorded_utc']
post_nominal=next((g for g in reversed(e['current_gates']) if g['folder'].endswith('_nominal') and 'gate_001_' not in g['folder']),None)
post_summary=None if post_nominal is None else dict(folder=post_nominal['folder'],passed=post_nominal['passed'],cases=[dict(
    command_m_s=c['statistics']['command_m_s'],signed_forward_m_s=c['statistics']['reset_frame_forward_speed_m_s'],
    survival=c['statistics']['survival'],failures=c['failures']) for c in post_nominal['cases']])
post_push=next((g for g in reversed(e['current_gates']) if g['folder'].endswith('_push') and 'gate_001_' not in g['folder']),None)
post_push_summary=None if post_push is None else dict(folder=post_push['folder'],passed=post_push['passed'],cases=[dict(
    command_m_s=c['statistics']['command_m_s'],signed_forward_m_s=c['statistics']['reset_frame_forward_speed_m_s'],
    survival=c['statistics']['survival'],failures=c['failures']) for c in post_push['cases']])
main_summary=None if e['last_completed_main'] is None else {k:v for k,v in e['last_completed_main'].items() if k!='last_progress'}
report=f'''# Adaptive three-hour check — October 4, 08:36 UTC trigger

Recorded {recorded}. Read ACTIVE_RUN.json and fresh state for later progress.

## Result and continuation

R30 independently qualified 0.35 m/s under matching nominal/pushed and retained
command gates, then improved straight alternating walking at 0.65 m/s. Its
0.65 cases pass repeatedly; promotion remains blocked by retained 0.15 m/s
overspeed. This is a specific tracking/retention failure, not evidence of another
circling, hopping or stationary-step qualification loophole. Reward alone still
does not certify the policy.

R30 was drained only after another matching nominal slow-speed failure, at its
saved full **1,472-main-update** checkpoint, with the pushed evaluation allowed
to finish. All owned workers exited; no updating training worker was interrupted.
The source-frozen **R31 exact-command-rehearsal** continuation is now
**{live['status']}**, stage **{live['stage']}**, operation
`{Path(live['operation_path']).name if live.get('operation_path') else 'boundary'}`.
Current environment count: {live.get('environment_count')}; next training count:
{live.get('next_training_environment_count')}. New main updates:
{live['total_training_iterations']}. Preserve an improving candidate.

Initialization remains the exact original R30 checkpoint:
`{load['checkpoint']}`

SHA256 `{load['checkpoint_sha256']}`. CPU full-load equality confirms all
9 actor, 8 critic and 51 Adam tensors. Exploration and Adam moments are retained.
The unchanged worker restart rule bounds the saved 0.00015 learning rate to the
existing 0.0001 ceiling; it is not an exact-rate preservation claim. Small smoke
actors and benchmark actors are discarded before main learning.

## Independent behavioral evidence before repair

The last R30 matching 128-robot, 20-second nominal/pushed evaluations are
`{nominal['folder']}` and `{pushed['folder']}`. Every command has32 trials.
All latest moving cases survive and record22/23 alternating landings, no same-foot
or paired-hop landings, low foot slip and close heading alignment.

{table}

The 0.15 command repeatedly measured approximately0.27–0.30 m/s over late
checkpoints, even while the 0.65 case improved. The earlier 0.35 qualified actor
did pass0.15, so this is not established as a mechanical minimum walking speed.
Current1.0 m/s and running stages have not qualified. Earlier R25 passes under the
old direction-permissive verifier remain historical, not new straight-walk proof.

## Focused repair and scope

Old training already sampled earlier speeds, but only about1.8% of all training
commands fell within0.02 m/s of0.15 at the0.65 stage. Exact-command rehearsal now
gives0.15 about12.1% of the population, a measured6.71-fold exposure increase.
The total mixture remains32% stopped,24% prior moving and44% frontier moving.
Prior rows select exact independently evaluated speeds; frontier rows include its
exact command and retain the original0.8–1.0 range. RNG draw count and the
permanently stopped one-fifth cohort remain. This is a declared local training
adaptation, not a claimed upstream Unitree sampler or proven convergence cure.

Actor/action/observation/history/reset semantics, all reward kernels/weights,
geometry, mass/inertia, contacts, solver settings, CAD Jacobians/reductions and
hardware bounds are unchanged. Selected iteration4 execution optimizations stay.
Evaluation delegates to the original sampler. Removing only the pinned
training override reconstructs the original environment bytes and signature;
restoring the narrowly listed metadata/signature changes reconstructs the full
old worker, including PPO and checkpoint loading. All other authored runtime
files must match. Current frozen hashes verified: {len(e['active_frozen'])}.

The same saved checkpoint is re-evaluated before main learning. R30's matching
older-speed qualification is preserved for its exact trusted checkpoint; every
current and lower-speed command still must pass before further promotion.
No gate threshold was relaxed.

Latest completed post-rehearsal nominal independent result: {json.dumps(post_summary) if post_summary else 'Awaiting completion; do not infer improvement from the training reward.'}
Latest completed post-rehearsal pushed independent result: {json.dumps(post_push_summary) if post_push_summary else 'Awaiting completion.'}
The early slow-speed improvement is not monotonic: nominal0.15 moved from0.277
before rehearsal to0.258 at64 updates and0.268 at128. Current0.65 tracking improved
from0.732 to0.685 then0.658, retaining survival and alternating steps. Preserve
the useful progress, but do not describe the slow-control issue as solved.
Matching pushed/retention qualification remains mandatory before promotion.

## Tested extension toward5 m/s

The frozen curriculum now continues through1.0 walking and1.5/2.0/2.5/3.0/3.5/
4.0/4.5/5.0 m/s running. Existing prior speed cases/classifiers are unchanged.
At5, an additional actual signed reset-frame speed check requires at least4.95
m/s, declaring a1% measurement tolerance; commanding5 is never success.
Population grows to keep at least32 trials per command, reaching416 robots per
nominal/pushed final test. Old cadence, clearance, flight, direction, net travel,
anti-fall/slide and retention requirements remain.

The endpoint is independently qualified actual5 m/s or a reproduced relevant
actuator velocity ceiling. A ceiling still requires motor speeds through the
actual differential Jacobians, sustained phase-specific saturation, torque
headroom and a controlled speed-plateau diagnosis. None is established here.
At either endpoint save, pause training and this heartbeat, and brief the user;
leave the computer on. No shutdown occurred.

## Validation and fidelity

- 51 CPU continuation/resource/heading/ownership/balance tests pass. Scheduler
  fixtures were made coherent with their own frozen source; no guard was weakened.
- 24 evaluation sampler/RNG comparisons are bitwise identical. All182 saved
  original case classifications match, and10 protected-source/false-qualification
  counterexamples are rejected. The current proof pins these tests and5 allowed
  runtime files; there is no broad signature bypass.
- 32 phase/lift/lifetime/standing-retention-gradient/PPO-audit assertions pass.
  Eight running command speeds through5 favor matched cycles over in-place,
  backward, sliding and synchronized-hop fixtures. The goal classifier rejects
  a4.8 m/s counterexample that the generic old5-command gate would accept.
- 5,593 upstream G1/H1 counterfactual checks pass. Those upstream tests also
  reproduce limitations of raw rewards; fixtures do not qualify real gait.
- Fresh32-robot physical/contact/combined-transmission preflight passes, with
  zero invalid poses and337.61 N support against338.09 N estimated weight.
- Actual CUDA64-robot four-update PPO and four-second checkpoint reload complete.
  Every smoke pre-update KL is zero; full-load equality is separately saved.
  The short reload did not pass every gait gate and is never a qualification.
- Same-checkpoint independent physical replay: {json.dumps(parity_summary) if parity else 'Awaiting full baseline replay; CPU sampler/signature fidelity above is verified.'}
  Details: [evaluation parity](evaluation_parity.json).

Files: [CPU proof](cpu_test.json), [full checkpoint load](full_checkpoint_load.json),
[running tests](running_extension_tests.json), [migration proof](continuation_proof.json),
[drain](r30_drain.json), [raw evidence](evidence.json). Small fixture/build attempts
initially caught mixed-newline restoration, fixture tick dtype and stale capacity
fixtures. Two harmless drain monitors used the wrong nominal-folder index and
were stopped before touching training; the final owned-process drain succeeded.
Those failures were preparation/test problems, not erased production failures.

## Throughput and memory

Latest six R30 main6144 blocks average
{statistics.mean(b['active_ppo_samples_s'] for b in blocks):,.0f} active PPO samples/s
(samples divided by collection+learning time),
{statistics.mean(b['worker_effective_samples_s'] for b in blocks):,.0f} worker-effective
samples/s including worker starts/exits. Full observed R30 main samples/time is
{e['prior_full_chain_main_samples_s']:,.0f} samples/s including startup,
capacity and independent gates. These denominators are distinct.

Those blocks retain at least1692 MiB GPU free and10326 MiB host available.
Worker private memory grows only6–9 MiB per64 updates; Torch reserve stays
1630–1632 MiB. No new unbounded cache leak is demonstrated. One earlier8192
main block hit the300 MiB reserve and saved the full recovery state; subsequent
6144 workers were stable. Its35 interrupted updates are preserved and included
in measured sample totals even though the supervisor's main count omits them.

R31 capacity results are saved in benchmarks.json:
`{json.dumps(capacity)}`

Latest completed new main block: `{json.dumps(main_summary)}`.

Current worker completed updates at this observation: {e['current_worker_completed_updates']}.
Current collection+learning throughput: {e['current_active_PPO_samples_s']} samples/s.
Latest progress memory: {json.dumps(e['current_worker_progress']['gpu']) if e['current_worker_progress'] else 'Worker is between saved progress records.'}
Latest command coverage: {json.dumps(e['current_worker_progress'].get('command_coverage')) if e['current_worker_progress'] else 'Awaiting current worker progress.'}
These are training diagnostics, not an independent post-repair gait pass.

The target stays8192 with both host/GPU reserves, automatic fallback and180
continuously healthy training seconds before upward ramps. Stopped-command
capacity alone does not guarantee moving-command memory headroom. Unrelated
unsaved Fusion work is preserved. No second actual CUDA worker was started.

## Primary-source cross-check and next focus

[Unitree H1 configuration](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h1/velocity_env_cfg.py)
uses a widening velocity-command curriculum and nominal-offset joint actions;
[Isaac Lab command implementation](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py)
exposes sampled velocity commands and heading feedback. These support auditing
the command distribution; the exact retained-command rehearsal is our
evidence-based adaptation. Matching rewards alone does not make different robot
actuation, physics and implementation equivalent.

Next1–3 tests: compare post-rehearsal0.15/0.35/current-frontier independent speed
and survival with this baseline; preserve improvement and finish matching gates.
Monitor moving-command8192 memory/fallback and actual sample/time, retaining the
optimization. Before running promotion inspect cadence/flight/clearance and
actual motor velocity telemetry; if tracking stalls, measure per-command reward
gradient/response rather than repeatedly changing weights.
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
next_check=dict(recorded_utc=recorded,active_run=str(run),checkpoint=live.get('checkpoint') or load['checkpoint'],
    checkpoint_sha256=e['current_checkpoint_sha256'],
    status=live['status'],stage=live['stage'],source_hashes={x['name']:x['sha256'] for x in e['active_frozen']},
    independently_qualified_speeds=sorted(set(p['speed'] for p in live['promotions'])),qualification_scope='Exact trusted checkpoints; current candidate requires all matching gates',
    current_curriculum_max_m_s=5.,candidate_qualified=False,action_taken='Full-state R31 continuation with training-only exact-command rehearsal; prior reward/physics/gates preserved; tested5m/s extension',
    baseline_gates=[nominal,pushed],prior_main_updates=1472,prior_initialization_updates=256,
    throughput=dict(prior_recent_active_PPO_samples_s=statistics.mean(b['active_ppo_samples_s'] for b in blocks),
                    prior_recent_worker_effective_samples_s=statistics.mean(b['worker_effective_samples_s'] for b in blocks),
                    prior_full_chain_main_samples_s=e['prior_full_chain_main_samples_s'],denominators='collection+learning; full worker wall; full chain wall'),
    resource_snapshot=dict(gpu_smi=e['gpu_smi'],host_available_mb=e['host_available_mb']),capacity=capacity,
    current_worker_progress=e['current_worker_progress'],current_worker_completed_updates=e['current_worker_completed_updates'],
    current_active_PPO_samples_s=e['current_active_PPO_samples_s'],
    last_completed_main=e['last_completed_main'],latest_post_rehearsal_nominal=post_summary,
    latest_post_rehearsal_pushed=post_push_summary,
    slow_speed_trend='Nominal command0.15:0.277 before rehearsal,0.258 after64 and0.268 after128; not monotonic or qualified',
    current_speed_trend='Nominal command0.65:0.732 before rehearsal,0.685 after64 and0.658 after128; qualified case but retained-speed gates still block stage',
    next_focus=['Compare exact-command rehearsal slow-speed retention/current-frontier nominal+pushed gates and preserve improving actors.',
                'Measure moving-command memory and throughput; verify8192 fallback/progress and180s host/GPU ramp without changing limits.',
                'Audit running cadence/flight/clearance and actual CAD motor speeds before promotion; investigate per-command reward gradients if tracking stalls.'],
    report=str(OUT/'REPORT.md'),runbook=str(HERE/'automation/THREE_HOUR_CHECK.md'),
    endpoint='Actual5m/s independent qualification or reproduced locomotion-relevant velocity ceiling; save,pause,brief; computer stays on')
(HERE/'automation/NEXT_CHECK.json').write_text(json.dumps(next_check,indent=2),encoding='utf-8')
print(json.dumps(dict(report=str(OUT/'REPORT.md'),active_run=run.name,stage=live['stage'],status=live['status'])))
