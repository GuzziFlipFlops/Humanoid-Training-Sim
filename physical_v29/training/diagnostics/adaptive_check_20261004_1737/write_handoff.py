"""Record this check, its preserved failures, and the next adaptive priorities."""
from datetime import datetime,timezone
from pathlib import Path
import json,re,tomllib
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
status=read(OUT/'replacement_status.json');state=status['state'];deployment=read(OUT/'deployment.json')
old=Path(deployment['old_run']);e=read(OUT/'evidence.json');physical=read(OUT/'physical_response/result.json')
proof=read(OUT/'retention_continuation_proof.json');run=Path(status['active_pointer']['run'])
progress=state.get('progress',{});now=status['recorded_utc']
gates=[]
for folder in ['029_run_150_gate_005_nominal','030_run_150_gate_005_push','077_run_150_gate_021_nominal']:
 r=read(old/folder/'result.json');gates.append((folder,r['cases']))
table='| Actor / test | Actual 1.5-command signed speed | Genuine flight | Retention |\n| --- | ---: | ---: | --- |\n'
for folder,cases in gates:
 s=cases[-1]['statistics'];fail=[str(c['statistics']['command_m_s']) for c in cases[:-1] if not c['passed']]
 table+=f"| {folder} | {s['reset_frame_forward_speed_m_s']:.3f} m/s | {100*s['geometric_flight_fraction']:.4f}% | Failed {', '.join(fail) if fail else 'none'} |\n"
fresh=[]
for g in status['gates']:
 if 'requalification' in Path(g['folder']).name:fresh.append(dict(folder=g['folder'],passed=g['result']['passed'],cases=g['result']['cases']))
qualified=bool(len(fresh)>=2 and all(x['passed'] for x in fresh))
operation=Path(state.get('operation_path','')).name
main=read(OUT/'new_main_evidence.json') if (OUT/'new_main_evidence.json').exists() else None
execution=''
behavioral_brief=''
if main and main['blocks']:
 b=next((x for x in reversed(main['blocks']) if x['completed_result']),main['blocks'][-1])
 execution=f'''Latest main execution evidence: `{Path(b['folder']).name}` has
{b['observed_updates']} observed updates. Exact actor, exploration, critic and Adam
load checks pass. Every recorded pre-update KL and action-command mismatch is
zero. The qualified prior hook ran20 minibatches per update on the certified
0.1-1.0m/s command window; it is absent from inference. Minimum free VRAM was
{b['min_free_gpu_mb']}MiB, host available {b['min_host_available_mb']}MiB,
private-memory growth {b['private_growth_mb']}MiB and Torch allocation growth
{b['torch_allocated_growth_mb']}MiB. Active collection-plus-PPO throughput was
{b['active_samples_per_second']:.0f} samples/s. See `new_main_evidence.json` for
the exact scope, completed result and matching independent behavioral gates.
These execution measurements alone do not qualify running.
'''
 if b['worker_effective_samples_per_second']:
  execution+=f"The completed worker block measured {b['worker_effective_samples_per_second']:.0f} samples/s including its startup and shutdown; it used {b['completed_result']['num_envs']} environments. This is a different denominator from active collection-plus-PPO time.\n"
 behavior_block=None;matching=[]
 for candidate in reversed(main['blocks']):
  pair=[g for g in main['gates'] if candidate['checkpoint_sha256'] and g['checkpoint_sha256']==candidate['checkpoint_sha256']]
  if len(pair)>=2:
   behavior_block=candidate;matching=pair;break
 if len(matching)>=2:
  a,z=matching[-2:];ar,zr=a['cases'][-1],z['cases'][-1]
  cumulative=sum(x['completed_result']['iterations'] for x in main['blocks'][:main['blocks'].index(behavior_block)+1] if x['completed_result'])
  passing=[str(c['command_m_s']) for c,d in zip(a['cases'][:-1],z['cases'][:-1]) if c['passed'] and d['passed']]
  failing=[str(c['command_m_s']) for c,d in zip(a['cases'][:-1],z['cases'][:-1]) if not(c['passed'] and d['passed'])]
  outcome='passed' if a['passed'] and z['passed'] else 'remain unqualified'
  behavioral_brief=f"After {cumulative} new updates, matching nominal {Path(a['folder']).name} and pushed {Path(z['folder']).name} {outcome}. Running signed speed is {ar['forward_speed_m_s']:.3f}/{zr['forward_speed_m_s']:.3f}m/s; true flight is {100*ar['geometric_flight_fraction']:.4f}%/{100*zr['geometric_flight_fraction']:.4f}%, against the unchanged1.5% minimum. Earlier commands passing both: {', '.join(passing)}. Earlier commands still failing: {', '.join(failing) or 'none'}. Judge the repair against both retained tracking and geometric flight, not either alone. Compare subsequent complete pairs before attributing a tradeoff to the new gradient, and preserve both the selected flight actor and qualified walking teacher."
  execution+='\n'+behavioral_brief+'\n'
 main_table='| Test | Command | Signed speed | Geometric flight | Passed / failures |\n| --- | ---: | ---: | ---: | --- |\n'
 for g in main['gates']:
  for c in g['cases']:
   outcome='Passed' if c['passed'] else ', '.join(c['failed_checks'])
   main_table+=f"| {Path(g['folder']).name} | {c['command_m_s']:.2f} | {c['forward_speed_m_s']:.3f}m/s | {100*c['geometric_flight_fraction']:.4f}% | {outcome} |\n"
 execution+='\n'+main_table+'\n'
focus_tests=[
 'Compare matching full nominal/pushed running and stopped/prior-speed retention against R32 checkpoint269: genuine flight2.0833%/1.9236%, signed1.539m/s, prior0.15/0.35/0.65 tracking failures. Track whether the training-only walking prior reduces those failures without erasing flight; preserve useful actors.',
 'Inspect locomotion_retention MSE/loss and per-command clipped output/target response. The prior acts only on latest-frame commands above0.1 and at/below the certified earlier speed; no teacher exists in inference. If flight or retention plateaus, use controlled gradient/physics tradeoffs before another narrow repair, not arbitrary reward weights.',
 'Check moving6144/8192 capacity, free VRAM/host RAM and effective samples/time including the bounded teacher cost. Verify full actor/critic/Adam recovery and180-second healthy upward ramps. Retain actual CAD motor-speed telemetry; sampled spikes do not prove a velocity ceiling.'
]
if behavioral_brief:
 focus_tests[0]=behavioral_brief+' Next compare newer full matched pairs against this saved baseline. If retained tracking or genuine flight remains stalled, investigate whether the qualified walking-prior gradient conflicts with running PPO/flight objectives, including effective clipped action/target gradient contributions and controlled physical tradeoffs. A behavioral tradeoff alone does not establish gradient interference. Do not repeat reward-identical waiting indefinitely or erase useful actors.'
report=f'''# October 4, 17:37 UTC adaptive check: running regression and qualified walking retention

R33 is the source-frozen replacement for R32. The selected full actor had learned
genuine running at1.5m/s but lost slower walking; later R32 updates lost flight.
The current repair addresses that demonstrated retention problem. It does not
claim the stand-walk-run chain or5m/s goal is qualified.

Latest observation: {now}. Status **{state['status']}**, stage
**{state['stage']}**, operation `{operation}`, observed count
**{state.get('environment_count')}**, completed new main updates
**{state['total_training_iterations']}**. All **{len(status['hashes'])}** frozen
hashes verify. Actual workers at this observation: **{status['actual_worker_count']}**.
Fresh trusted baseline nominal/pushed requalification complete: **{qualified}**.
Read `replacement_status.json` and fresh ACTIVE_RUN/state rather than assuming
this dated observation remains current. Smokes and capacity are not learned progress.

{execution}

## Independent evidence and diagnosis

R32 is preserved at1352 completed main updates and its final full checkpoint
`076_run_150_train_020/candidate.pt`, SHA256
`{read(OUT/'diagnostic_pause.json')['checkpoint_sha256']}`. All source files,
logs, evaluations and intermediate policies remain. R32 had46.15k active PPO,
39.35k worker-effective and20.95k whole-chain completed-main samples/s at the
17:41 observation. These denominators differ; whole-chain time includes the
previous documented775.6-second user GUI pause. Subtract that pause for an active
comparison, and keep the original clock evidence intact. Latest completed6144
block had804MiB minimum free VRAM,8321MiB host available and5MiB private growth.
No pre-update KL or action-command mismatch was observed.

{table}
Every running case above used32 independent robots. The early269-update
checkpoint028 passed **all running-case checks in both nominal and pushed tests**,
including speed, direction, alternating steps, survival and geometric flight.
Its prior-speed cases failed, so the whole running stage never qualified. Flight
peaked early and returned to zero in nine later nominal tests; slow retention
also failed repeatedly. This is regression, not a crash or evidence that the
robot cannot take off. The qualified standing/walking baseline remains preserved.

A controlled96-robot10-second normal-gravity comparison reproduced the earlier
actor's genuine flight at2.2792% versus0% for the final actor, with all robots
surviving and no resets. Their signed-world forward speeds were1.546 and1.525m/s.
The earlier actor's reward rate was14.247 versus14.099 for the grounded final
actor. The qualified teacher tracked the0.15 command at0.180 versus0.272 for the
final actor and0.353 for the early flight actor. A bounded diagnostic knee pulse
on the final actor produced only0.285% flight and reduced reward; phase offsets
did not restore flight. No pulse, phase helper or root support was deployed.
See `physical_response/result.json` and its full reward-term/target/force telemetry.

The saved50Hz final trace had a loaded foot in every intended flight window.
Action reconstruction matched2240 saved samples within6.68e-6; command snapshots
matched.39.5% of sampled raw running actions exceeded the existing+/-3 clamp.
Exact CAD Jacobians/reductions show no sustained leg velocity ceiling at these
saved10Hz states. Rare right-ankle samples reached98.5% of their cap; this does
not attribute failure to clipping or exclude substep saturation.

## Narrow repair and source lineage

The old bounded training prior protected stopped-command standing only. Earlier
walking commands had ordinary reward/rehearsal but **no corresponding retention
gradient**. R33 adds a separate training-only mean-action gradient toward the
latest independently qualified prior locomotion actor, limited by the **latest**
observation command. Running-stage samples above the teacher's certified speed
receive no such gradient. The original stopped-command gradient is bitwise
unchanged. The teacher is frozen, bounded to256 samples per minibatch, uses the
existing0.25 regularization coefficient, and is absent from inference. This is
a local adaptation, and closed-loop improvement remains to be measured.

Teacher use fails closed unless exact checkpoint bytes, current observation,
asset/motion/actuator contract, reward signature, classifier version and matching
full20-second nominal/pushed gates bind the entire earlier command set with
at least32 trials each. New future running stages select their most recent
qualified prior speed, without imposing a walking teacher on running samples.

Research supports using trained teachers to transfer and consolidate behaviors;
it does not guarantee this robot's result. [Policy Distillation](https://arxiv.org/abs/1511.06295)
and [Kickstarting Deep Reinforcement Learning](https://arxiv.org/abs/1803.03835)
support that general method. The0.25 bounded Gaussian-mean MSE hook here is a
declared implementation, not their exact published loss. The earlier
[Periodic Reward Composition](https://arxiv.org/html/2011.01387v2) force/phase
review and [public G1 task](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py)
remain background; no reward kernel or weight was changed this check.

Selected full checkpoint: `{proof['selected_checkpoint']}`;
SHA256 `{proof['selected_checkpoint_sha256']}`;269 selected main updates.
Its actor, trained exploration, critic and Adam tensors load exactly. The existing
restart learning-rate ceiling bounds the saved2.25e-4 rate to1e-4; adaptive PPO
continues normally. Smoke/benchmark actors are discarded. The later1352-update
actor is preserved independently, rather than miscounted as the selected lineage.

Exactly three authored runtime files change: `standing_retention.py`,
`worker_unitree.py` and the supervisor.42 other frozen authored files, reward v7,
geometric verifier v6,400-input23-output inference, command distribution/history,
PPO algorithm/hyperparameters, geometry/contact/mass/CAD motor limits, iteration4
execution and all earlier/final gates remain identical. The new sources are
frozen in R33; R32 frozen bytes were never edited.

## Verification, including failed diagnostic attempts

22 targeted CPU checks pass: exact gradients, only certified prior rows, latest
frame selection, unchanged standing gradient, additive disjoint hooks, frozen
teacher, no inference hooks/behavior changes, qualification binding and negative
provenance tests. Nine further migration checks reject wrong selected/final
hashes, signatures, classifier/schema, active prior processes and protected source
changes. `cpu_tests.json` and `migration_validation_tests.json` contain the results.

The complete192-robot20-second nominal replay of checkpoint028 reproduces **all
case results and every saved physical/50Hz telemetry array bit for bit**, including
actions, targets, roots, joints, contacts, forces, COM and failures. See
`physical_parity.json`. Small real64-env running PPO and full-checkpoint reload
complete; all four pre-update KL and action-command mismatch values are zero.
`retention_fullstate_smoke_02/full_load_proof.json` confirms exact actor/critic/Adam
and exploration load. These smokes are execution evidence, not qualification.

The first CPU test incorrectly imported simulator-only evaluation utilities
without AppLauncher; it was corrected to AST-read the version constant. A later
new load-audit assertion compared Adam's CPU scalar step to a CUDA-loaded scalar;
the scalar values were unchanged. The failed diagnostic is preserved under
`retention_fullstate_smoke`, with its traceback and exact owned-worker cleanup.
It had left Kit open after failure; that failed process was closed, and the
retry passes after device-neutral exact comparison. There was a brief overlap
of the failed Kit context and retry startup; it never represented two training
workers. All subsequent simulator operations are sequential. No failed candidate
or short-reload behavioral result was accepted as qualification.

The old supervisor was suspended at a saved boundary and its evaluation worker
drained before controlled physics. Its exact identity was then retired after
validation. R33 was launched with full continuation, fresh physics/PPO/load,
capacity tests and matching trusted-baseline gates. The current candidate must
still pass **all** running and stopped/prior-speed gates to promote.

## Next decision and endpoint

1. {focus_tests[0]}
2. {focus_tests[1]}
3. {focus_tests[2]}

Keep one actual simulator worker, no stereo,8192 target,300MiB GPU reserve,
host guards, fallback and180 healthy seconds before upward ramps. The unchanged
curriculum proceeds through5m/s. Only measured signed>=4.95m/s plus matching
nominal/pushed and retained skill gates, or a causal demonstrated relevant
actuator velocity ceiling, satisfies the endpoint. No such endpoint is proven.
The same adaptive three-hour heartbeat remains active; at endpoint save, pause
owned training and the automation, and brief. **Leave the computer on.**
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
note=f'''<!-- adaptive_check_20261004_1737 -->
## October4 17:37 UTC: reproduced flight regression; R33 qualified prior retention

R32 is preserved at1352 updates. Its269-update checkpoint028 passed the1.5m/s
running case in matching nominal/pushed tests (1.539m/s;2.0833%/1.9236% real
geometric flight), but earlier walking failed, so no whole-stage qualification.
Later flight regressed to zero. A96-robot physical comparison reproduced this;
all survived. The standing prior had no gradient on moving prior commands.
R33 continues the exact028 actor/exploration/critic/Adam, with bounded training-only
qualified prior-speed teacher gradients; inference, reward v7, geometric gate v6,
all physics/motor constraints and iteration4 are unchanged. Teacher never controls
inference.22 CPU plus9 migration checks, real PPO/load and full192-robot20-second
bitwise replay pass. A failed diagnostic Adam-device comparison is preserved and
fixed only in the new load audit. Read [the dated report]({OUT.as_posix()}/REPORT.md),
proofs and fresh ACTIVE_RUN/state. Observed{now}: {state['status']} {state['stage']},
count{state.get('environment_count')}, new main{state['total_training_iterations']};
46 frozen hashes verify. Fresh baseline teacher requalification complete:{qualified}.
{behavioral_brief}
No5m/s or relevant velocity-ceiling endpoint is proven. Same three-hour checks
remain active; endpoint saves/pauses/briefs with the computer on.
<!-- /adaptive_check_20261004_1737 -->
'''
def insert(path,where='append'):
 text=path.read_text(encoding='utf-8-sig');text=re.sub(r'<!-- adaptive_check_20261004_1737 -->.*?<!-- /adaptive_check_20261004_1737 -->\s*','',text,flags=re.S)
 if where=='prepend':text=note+'\n'+text
 else:text=text.rstrip()+'\n\n'+note
 path.write_text(text,encoding='utf-8')
history=HERE/'ITERATION_HISTORY.md';text=history.read_text(encoding='utf-8')
start=text.index('**Current recorded state:**');end=text.index('| Read next',start)
text=text[:start]+f'''**Current recorded state:** R33 qualified prior walking retention continues the
full earlier R32 actor that passed1.5m/s running cases but lost walking. Later
R32 is preserved at1352 updates. The current training-only repair leaves inference,
rewards and all physical/independent gates unchanged. Read the October4 17:37
report and fresh state. Full running-stage and5m/s qualification remain incomplete.
The same adaptive three-hour check saves/pauses/briefs at the verified endpoint;
computer stays on.

'''+text[end:];text=text.replace('Every R1–R29 run','Every R1–R33 run');history.write_text(text,encoding='utf-8');insert(history)
insert(HERE/'automation/THREE_HOUR_CHECK.md')
insert(Path('docs/robot/TRAINING_JOURNAL_20261001.md'),'prepend')
repo=Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot')
insert(repo/'docs/robot/TRAINING_JOURNAL_20261001.md','prepend')
insert(repo/'Learning/HumanoidTraining/FILE_LOCATIONS.md','prepend')
insert(Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),'prepend')
nextcheck=dict(recorded_utc=now,run=str(run),status=state['status'],stage=state['stage'],
 checkpoint=state.get('checkpoint'),selected_seed=proof['selected_checkpoint'],selected_seed_sha256=proof['selected_checkpoint_sha256'],
 preserved_final_R32=deployment['old_final_checkpoint'],new_main_updates=state['total_training_iterations'],
 selected_prior_updates=269,environment_count=state.get('environment_count'),target_environments=8192,
 verified_hashes=46,actual_worker_count=status['actual_worker_count'],host_available_mb=status['host_available_mb'],
 gpu=status['gpu_used_free_utilization_mb_pct'],current_progress=progress,qualified_teacher_rechecked=qualified,
 new_main_execution=main,
 report=str(OUT/'REPORT.md'),next_focused_tests=focus_tests,source_scope='Training-only qualified prior retention; rewards/inference/physics/all gates unchanged',
 endpoint='Actual signed>=4.95m/s matching nominal/pushed/stopped/prior gates or causally demonstrated relevant velocity ceiling; save/pause/brief, computer on')
(HERE/'automation/NEXT_CHECK.json').write_text(json.dumps(nextcheck,indent=2)+'\n')
automation_path=Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml')
automation=tomllib.loads(automation_path.read_text());(OUT/'automation_before.json').write_text(json.dumps(automation,indent=2)+'\n')
prefix=automation['prompt'].split('Latest dated focus,')[0]
focus=f'''Latest dated focus, October4 17:37 UTC check; observed{now}: R32 is preserved at1352 main updates and final checkpoint076. It briefly learned real running: checkpoint028 at269 updates passed every1.5m/s running-case check in matching nominal029/pushed030, signed1.539m/s and genuine geometric flight2.0833%/1.9236%. Earlier0.15/0.35/0.65 commands failed, so the complete running stage never qualified. Later flight regressed to zero across nine nominal tests and slow-speed tracking again failed.96-robot10-second physical comparison reproduced2.2792% flight for028 versus0% final actor, all survived; teacher still tracked0.15 at0.180m/s. The old prior gradient protected stopped commands only. R33_qualified_walking_retention restores exact028 actor/exploration/critic/Adam with the existing1e-4 restart LR ceiling, adding separate bounded training-only gradients toward the latest qualified prior locomotion actor only for latest-frame earlier walking commands. Teacher is frozen and absent from inference. Qualification binds exact checkpoint/signature/current physical contract and matching full20-second nominal/pushed prior gates with>=32 trials per command. Original standing gradient and all running samples are unchanged. Reward v7, verifier v6,400-input23-output inference, actions, commands/history, mass/contact/CAD/motor limits, PPO algorithm settings, iteration4 and every independent gate remain identical. Exactly two frozen training files plus supervisor change;42 protected frozen authored files unchanged.22 CPU and9 migration tests pass; real64-env running PPO/full-load/reload passes with all actor/critic/Adam/exploration tensors equal and zero pre-update KL/command mismatch. Entire192-robot20-second same-actor nominal replay, every case and every saved array, is bitwise identical. Read diagnostics/adaptive_check_20261004_1737/REPORT.md, gate_trends.json, physical_response/result.json, physical_parity.json, retention_continuation_proof.json, deployment.json and fresh replacement_status.json. Preserve failed diagnostic Adam-device audit and exact cleanup; it was corrected before deployment, not a training fault. Fresh observation R33 status{state['status']},stage{state['stage']},count{state.get('environment_count')},new main{state['total_training_iterations']};46 hashes verify. Fresh prior teacher requalification complete:{qualified}. No running-stage or5m/s qualification is inferred from startup, the selected seed's running case, or smoke. R32 completed6144 blocks averaged46.15k active/39.35k worker-effective/20.95k whole-chain samples/s; last minGPU804MiB andhost8321MiB, stable memory. Previous775.6s user GUI pause is included in whole-chain clocks. Target8192/reserve300/fallback/180s healthy ramp remain. Next: {' '.join(focus_tests)} Read fresh ACTIVE_RUN/state, not this dated observation. Preserve improving actors and the independently qualified baseline. Keep the same adaptive three-hour check active until verified5m/s or demonstrated relevant motor velocity ceiling; then save, pause and brief with computer on.'''
prompt=(prefix+focus).replace('The source-frozen R31 curriculum now extends','The source-frozen curriculum extends')
if main and main['blocks']:
 b=next((x for x in reversed(main['blocks']) if x['completed_result']),main['blocks'][-1])
 prompt+=f" Fresh deployed main evidence recorded{main['recorded_utc']}: {Path(b['folder']).name}, {b['observed_updates']} observed updates; exact actor/exploration/critic/Adam load, zero maximum pre-update KL and command mismatch. Qualified moving-prior gradient runs20 batches/update on latest-frame0.1-1.0m/s commands. MinGPU{b['min_free_gpu_mb']}MiB,host{b['min_host_available_mb']}MiB,private growth{b['private_growth_mb']}MiB,Torch allocation growth{b['torch_allocated_growth_mb']}MiB; {b['active_samples_per_second']:.0f} active collection-plus-PPO samples/s. Read new_main_evidence.json for newer matching gates and completed-block denominators; this is not running qualification."
# Keep text ASCII because this host's automation backend has corrupted Unicode punctuation.
prompt=prompt.encode('ascii','replace').decode()
args=dict(mode='update',id=automation['id'],kind='heartbeat',destination='thread',name=automation['name'],
 status='ACTIVE',rrule=automation['rrule'],targetThreadId=automation['target_thread_id'],prompt=prompt)
if 'notification_policy' in automation:args['notificationPolicy']=automation['notification_policy']
(OUT/'automation_update_args.json').write_text(json.dumps(args,indent=2)+'\n')
print(json.dumps(dict(report=str(OUT/'REPORT.md'),observed=now,operation=operation,main_updates=state['total_training_iterations'],teacher_qualified=qualified),indent=2))
