"""Write the dated repair narrative and adaptive focus from preserved evidence."""
from datetime import datetime,timezone
import hashlib,json,re,tomllib
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
e=read(OUT/'evidence.json');s=read(OUT/'replacement_status.json');state=s['state']
seed=read(OUT/'selected_seed.json');drain=read(OUT/'drain.json');test=read(OUT/'cpu_tests.json')
deployment=read(OUT/'deployment_proof.json');probe=read(OUT/'physical_response/result.json')
run=Path(s['active_pointer']['run']);old=Path(drain['run']);audit=read(OUT/'running_audit.json')
now=datetime.now(timezone.utc).isoformat()
nom=read(old/'102_run_150_gate_022_nominal/result.json');push=read(old/'103_run_150_gate_022_push/result.json')
latest=read(old/'108_run_150_gate_024_nominal/result.json')
table='| Command | Selected 101 nominal/pushed signed speed | Selected prior cases | Last 107 nominal signed speed | Last failures |\n| --- | --- | --- | --- | --- |\n'
for n,p,l in zip(nom['cases'],push['cases'],latest['cases']):
    ns,ps,ls=n['statistics'],p['statistics'],l['statistics']
    table+=f"| {ns['command_m_s']:g} | {ns['reset_frame_forward_speed_m_s']:.3f}/{ps['reset_frame_forward_speed_m_s']:.3f} | {'Pass' if n['passed'] and p['passed'] else 'Fail'} | {ls['reset_frame_forward_speed_m_s']:.3f} | {', '.join(l['failures']) or 'Pass'} |\n"
groups='| Physical intervention | Actual forward speed | Both soles >1 mm and forces <1 N | Maximum minimum sole clearance | Old reward rate |\n| --- | --- | --- | --- | --- |\n'
for g in probe['groups']:
    groups+=f"| {g['name']} | {g['mean_forward_world_speed']:.3f} m/s | {g['unloaded_geometric_flight_fraction']:.3%} | {g['max_minimum_sole_clearance_m']*1000:.3f} mm | {g['mean_reward_rate']:.3f} |\n"
bench='Capacity testing is still in progress; a current worker count is not completed capacity.\n'
if s['completed_benchmarks']:
    bench='Completed saved capacity attempts:\n\n'
    for b in s['completed_benchmarks']:
        r=b['result'];bench+=f"- `{Path(b['folder']).name}`: {r.get('iterations')} updates, completed={r.get('completed')}, recycled={r.get('recycled',False)}, seconds={r.get('seconds')}.\n"
if (run/'benchmarks.json').exists():
    bench+='\n| Capacity count | Completed updates | Active PPO samples/s | Worker-effective samples/s | Minimum free GPU / host MiB |\n| --- | --- | --- | --- | --- |\n'
    for b in read(run/'benchmarks.json'):
        bench+=f"| {b['environments']} | {b['updates']} | {b['environment_steps_per_second']:,.0f} | {b['effective_environment_steps_per_second']:,.0f} | {b['minimum_free_mb']} / {b['minimum_host_available_mb']} |\n"
    bench+=f"\nSelected next training count: {state.get('next_training_environment_count')}. The 8192 attempt reached the VRAM reserve and was rejected, not accepted as capacity. Fresh promotions at this observation: {[(p['stage'],p['speed']) for p in state['promotions']]}.\n"
summary=f'''## October 4, 14:37 UTC adaptive check: running take-off repair

R31 was safely drained at a saved boundary after repeated 1.5 m/s running blocks
failed to develop genuine flight. Its final checkpoint107, 2,052 completed main
updates and all logs/qualifications remain preserved. The selected checkpoint101
passes every stopped and prior walking command in matching nominal/pushed gates102/103;
the latest107 regresses0.35/0.65. R32 retains101's exact mean and trained exploration,
with fresh critic/Adam for the changed running objective and fresh qualification.
Historical exact walking passes through1.0 m/s remain available; neither running
nor the5 m/s endpoint is qualified.

Repeated synchronized sole/force telemetry finds zero geometric flight despite
about1.43–1.45 m/s signed travel and100% survival. Small force-threshold-only
flight readings at101 are grounded contact gaps, not progress toward running.
A controlled96-robot normal-limit action/phase probe achieved brief small real
separation with a0.3rad knee-retraction pulse:0.285% geometric flight, maximum
minimum sole clearance1.318mm, all96 survivors, no resets. No training helper or
manual pulse was deployed. Its old total reward was lower, so the probe establishes
partial take-off feasibility, not that the whole objective favors that pulse.

Running v7 adds one bounded continuous phase-conditioned foot-force term,
weight0.5 and quarter-body-weight exponential scale. The flight bonus now requires
both actual soles above1mm and both normal forces below1N. Evaluation v6 adds
the same geometric evidence requirement with the existing1.5%–45% flight bounds;
the old flight gate and all prior gates remain. This is a local research-informed
adaptation, not an exact Unitree recipe or a convergence guarantee.

Forty-nine CPU checks pass, including600 bitwise standing/walking fixtures,
partial-unloading feedback, adversarial running cases through5m/s, grounded
false-flight rejection and source-bound mean/exploration transfer. A real64-env
four-update running PPO smoke and full checkpoint reload pass, all survivors,
pre-update KL0 and observation/action command difference0. Forty-one protected
authored runtime files are byte-identical; exactly3 files changed. Geometry,
mass/inertia, contacts, CAD differential constraints/motor limits, inference,
commands, PPO configuration and iteration4 execution remain unchanged.

Fresh observation {s['recorded_utc']}: R32 status={state['status']},
stage={state['stage']}, current worker environment count={state.get('environment_count')},
main updates={state['total_training_iterations']}. All46 frozen hashes verify;
{s['actual_worker_count']} actual worker, GPU used/free/utilization={s['gpu_used_free_utilization_mb_pct']},
host available{s['host_available_mb']}MiB. Count during preflight/capacity is not
the selected next PPO count ({state.get('next_training_environment_count')}). Fresh promotions are
{[(p['stage'],p['speed']) for p in state['promotions']]}, tied to the saved evaluated actor.
Target8192, reserve300MiB, host guards,
fallback and180 healthy seconds before an upward ramp remain enforced.

Read `E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261004_1437/REPORT.md`
and its replacement_status.json for fresh stage/qualification. Continue checking
true flight and retained speed tracking; do not count force unloading, smoke
updates or a5m/s command as qualification. The same three-hour heartbeat remains
active. No relevant sustained actuator velocity ceiling is demonstrated. At the
verified endpoint pause and brief; leave the computer on and preserve user work.
'''
report='# Adaptive training check — October 4, 14:37 UTC\n\n'+summary+'''

## Preserved actors and independent trends

'''+table+f'''

Selected initial actor: `{seed['checkpoint']}`.
SHA256 `{seed['sha256']}`. Last preserved actor: `{drain['checkpoint']}`,
SHA256 `{drain['checkpoint_sha256'] if 'checkpoint_sha256' in drain else sha(Path(drain['checkpoint']))}`.
R31 actor/critic/Adam/exploration remain intact on disk. The R32 transfer keeps
the selected mean/exploration exactly, discards value/moments because returns
changed, and discards smoke/benchmark-trained actors before qualification.
The actual v6 transfer is recorded in warm_start.json; the stock supervisor's
historical `v4 trained exploration` label is stale descriptive text, not its
source-bound transfer protocol. No qualification was reused.

R31 initial fresh evidence: {e['completed_main_active_PPO_samples_s']:,.0f} active
PPO samples/s, {e['completed_main_worker_effective_samples_s']:,.0f} worker-effective,
{e['full_chain_completed_main_samples_s']:,.0f} full-chain. These denominators differ.
An8192 block reached the300MiB VRAM guard and saved recovery; its remaining
updates finished at6144. Last saved6144 main block101 gave46,796 active/
40,057 worker-effective samples/s, minimum1,455MiB free VRAM and9,337MiB host
available, about6MiB private memory growth. Memory fallback preserved progress.
Do not describe all newer blocks as8192 or capacity tests as main throughput.

## Controlled physical experiment

'''+groups+'''

Eight groups of12 robots,10seconds at50Hz, same107 actor, normal gravity,
unfixed base, unchanged actuator limits. Interventions begin at3seconds.
Policy phase was shifted +/-0.08cycles, or a brief anatomically balanced
hip/knee/ankle retraction was applied only in intended flight windows. All96
robots survived; no resets. The0.3rad case has no both-sole clearance above5mm.
The pulse is a diagnostic intervention, absent from deployed action mapping.

The first diagnostic attempt failed on an inference-tensor ownership restriction;
its script/failure are preserved in probe_attempt_01. The probe was repaired by
cloning the action buffer before mutation and rerun successfully. Its own failed
worker was verified and stopped after preserving the exception; no unrelated
application or worker was stopped. This was a diagnostic failure, not a policy pass.

![Saved physical force/sole timing](physical_takeoff_timing.png)

Plot selects the maximum-clearance trial in each group; population statistics
are in result.json. It is telemetry, not rendered visual gait verification.

## Objective, physical and provenance limits

Old binary gait/flight feedback remains flat when a supported swing foot is
unloaded338→18N without crossing the contact classification. The added force
signal has a tested nonzero gradient under held posture/costs, positive bounded
range and no standing/walking reward change. The new aggregate objective's
physical effect and convergence still require independent learned rollouts.
The scripted retraction lowers the old average total reward even though it
slightly improves unloading. Do not claim the new0.5 weight reverses every cost
tradeoff, produces a full running stride or establishes hardware5m/s feasibility.

Saved actor-history reconstruction reproduces2240 actions within5.8e-6;
commands match. About35.1% of audited raw running outputs exceed the runtime
action clamp +/-3. Effective action/target clipping should continue to be audited.
Exact pose-dependent CAD Jacobian/reduction motor speeds show no sustained leg
velocity ceiling.10Hz sampled motor audit cannot exclude brief physics-substep
clipping, and isolated near-cap samples do not establish the endpoint.

The primary [Periodic Reward Composition paper](https://arxiv.org/html/2011.01387v2)
uses phase-conditioned force/velocity objectives with smoothed boundaries. It
supports testing force feedback; our positive exponential, quarter-weight scale
and0.5 rate are declared local adaptations. The public
[Unitree H1 velocity task](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/h1/velocity_env_cfg.py)
has walking phase duty0.55. Half-cycle alternation with that duty implies
overlapping support, not a running flight guarantee. Existing local running
duty0.43, motor limits and independent fixed-frame criteria remain.

## Tests, execution and next decision

'''+bench+'''

- [R31 initial evidence, gate inventory and memory/throughput](evidence.json).
- [Exact history, command, force/sole and CAD motor audit](running_audit.json).
- [Owned checkpoint drain](drain.json) and [selected retained actor](selected_seed.json).
- [Controlled96-robot results](physical_response/result.json), telemetry.npz and [probe code](physical_takeoff_probe.py).
- [49 CPU counterfactual/provenance checks](cpu_tests.json) and [test code](test_candidate.py).
- [Actual running PPO smoke](running_ppo_smoke/result.json), warm_start.json, progress.jsonl.
- [Full checkpoint reload diagnostic](running_checkpoint_reload/result.json):64envs/4sec, too small/short to qualify.
- [Three-file deployment and41-file protected equality proof](deployment_proof.json).
- [Replacement live state,46 hashes and owned worker](replacement_status.json).

The CPU-only handoff recorder originally rejected the owned supervisor because
Windows command arguments used forward slashes. Its output ownership check now
resolves the actual --output argument, accepting equivalent Windows path spellings
while rejecting another run, misleading name prefix or a checkpoint-only path.
Five ownership counterexamples pass. This diagnostic-only fix is outside the
frozen runtime inventory; it changes no simulator, policy or process behavior.
The prior recorder and failed diagnostic attempt are documented and preserved.

No stereo, external root forces, phase intervention, fixed base, enlarged motor
limits or relaxed gate were deployed. R31 frozen source and every checkpoint are
preserved. New R32 frozen source is independently checked. Fresh standing,
nominal/pushed walking and prior-speed retention must pass before running
promotion. Target5m/s remains actual signed>=4.95m/s with>=32 trials/command,
nominal/pushed and retained skills. Neither success nor a hardware ceiling is
proven. If the added feedback also plateaus, inspect effective target gradients,
force signs/distribution and costs per intended two-foot flight window with
controlled physical tests before another narrow repair; avoid arbitrary weights.
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
marker='adaptive_check_20261004_1437';wrapped=f'<!-- {marker} -->\n'+summary+f'<!-- end_{marker} -->\n\n'
paths=[HERE/'automation/THREE_HOUR_CHECK.md',HERE.parents[1]/'docs/robot/TRAINING_JOURNAL_20261001.md',
 Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md'),
 Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md'),
 Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md')]
for p in paths:
    text=p.read_text(encoding='utf-8-sig');text=re.sub(r'<!-- '+marker+r' -->.*?<!-- end_'+marker+r' -->\s*','',text,flags=re.S)
    text=text.rstrip()+'\n\n'+wrapped if p.name=='THREE_HOUR_CHECK.md' else wrapped+text
    p.write_text(text,encoding='utf-8')
p=HERE/'ITERATION_HISTORY.md';text=p.read_text(encoding='utf-8-sig')
text=re.sub(r'<!-- '+marker+r' -->.*?<!-- end_'+marker+r' -->\s*','',text,flags=re.S)
text=re.sub(r'\*\*Current recorded state:\*\*.*?(?=\n\| Read next)',
 '**Current recorded state:** R32 geometric-flight feedback is the fresh replacement\n'
 'for drained R31. It retains the exact better-retention101 actor/exploration,\n'
 'fresh critic/Adam and mandatory fresh gates. Running v7 adds continuous phase\n'
 'force feedback; verifier v6 requires true geometric flight. Historical walking\n'
 'qualification through1m/s is preserved. Read the October4 14:37 report and fresh\n'
 'state; current stage/count are observations, not qualification. The same\n'
 'three-hour checks remain active toward actual5m/s or a reproduced relevant\n'
 'velocity ceiling, then pause and brief with the computer left on.\n',text,flags=re.S)
anchor='## Physical-v29 runs, October 1–4';assert anchor in text
text=text.replace(anchor,wrapped+anchor,1);p.write_text(text,encoding='utf-8')
focus=f'''Latest dated focus, October4 14:37UTC check (replacement observed {s['recorded_utc']}): R31 was safely drained at2052 completed main updates after repeated1.5 running blocks had zero genuine sole-clearance flight, despite1.43–1.45m/s travel and100% survival. Preserve checkpoint107 and all old gates. Better-retention checkpoint101 passed every stopped/prior command in matching nominal102/pushed103; latest107 regressed0.35/0.65. Small force-only flight fractions at101 were grounded contact gaps, not learning progress. A96-robot10-second normal-limit phase/retraction probe produced brief true flight with0.3rad knee retraction:0.285% population flight, maximum minimum sole1.318mm, all96 survived/no resets; no flight above5mm. Old total reward decreased under that manual pulse. The pulse is diagnostic only and was not deployed. Read diagnostics/adaptive_check_20261004_1437/REPORT.md, running_audit.json, physical_response/result.json, cpu_tests.json, selected_seed.json and deployment_proof.json. R32_geometric_flight_feedback retains exact101 mean and exploration, fresh critic/Adam for changed returns, fresh qualification. Running v7 adds positive bounded phase-conditioned swing-foot unloading rate0.5 with quarter-body-weight exponential scale. Existing flight bonus now requires both actual soles>1mm and forces<1N; evaluation v6 adds that geometric flight gate with unchanged1.5%-45% bounds. Prior gates unchanged. Forty-nine CPU tests pass,600 walking/standing fixtures bitwise identical; actual64-env4-update running PPO/full-checkpoint reload passed, zero pre-update KL/command mismatch and all survivors. Exactly3 runtime files changed,41 protected authored files identical; geometry/CAD/motor limits/action/history/commands/PPO/iteration4 unchanged. Physical convergence is not yet proven. R32 currently {state['status']} at {state['stage']}, current count{state.get('environment_count')}, completed main{state['total_training_iterations']};46 frozen hashes verify and{s['actual_worker_count']} actual worker. Read fresh ACTIVE_RUN/state/replacement_status rather than assuming this observation remains current. Stock supervisor descriptive v4 transfer label is stale; warm_start.json and source-bound v6 protocol identify actual transfer. Target8192/reserve300/host guards/fallback/180s healthy ramp remain. R31 had a guarded8192 recovery followed by completed6144 updates, not all8192; about40.3k active/32.6k worker-effective/19.4k full-chain samples/s were its measured differing denominators. About35.1% raw running actions exceed +/-3 clamp; exact CAD sampled leg speeds show no sustained relevant velocity ceiling, substep clipping remains unexcluded. Next1 complete fresh matching standing/walking retention gates, preserve improving actor, then compare learned true geometric flight and slow speed tracking against the grounded baseline. Next2 if still plateaued audit two-foot intended-flight force signs/distribution, effective clipped targets and full objective/cost tradeoffs using controlled physics before another focused repair; the new continuous force gradient alone is not success. Next3 monitor moving8192 memory, saved fallback/full-state recovery,180s ramps and measured throughput; inspect cadence/clearance/flight/CAD motor speeds before promotion toward5m/s. Never accept force-only flight, a command, smoke updates or reward as qualification. Keep the same adaptive three-hour check active until actual5m/s or reproduced relevant velocity-ceiling endpoint, then pause and brief; leave computer on.'''
focus=f'''Latest dated focus, October 4, 14:37 UTC check; replacement observed {s['recorded_utc']}: R31 was safely drained at 2,052 completed main updates after repeated 1.5 m/s running blocks produced zero genuine flight despite 1.43–1.45 m/s travel and 100% survival. Preserve its final checkpoint107 and all walking qualification records. R32_geometric_flight_feedback uses checkpoint101, whose matching nominal102/pushed103 gates passed every stopped and prior walking case. Its exact mean and trained exploration are retained; critic/Adam are fresh because the running objective changed, and all qualifications must be repeated.

Read diagnostics/adaptive_check_20261004_1437/REPORT.md, selected_seed.json, running_audit.json, physical_response/result.json, cpu_tests.json, deployment_proof.json and replacement_status.json. Force-threshold flight at checkpoint101 was a grounded contact gap. A 96-robot normal-limit physical intervention produced only brief small true flight: 0.285% population fraction and 1.318 mm maximum minimum sole clearance; all robots survived without resets. The manual knee pulse reduced old total reward. It establishes partial take-off feasibility, not running qualification, and is absent from deployed control.

Running reward v7 adds bounded phase-conditioned foot-force unloading feedback, rate weight 0.5 and quarter-body-weight exponential scale. Its existing flight bonus now requires both soles above 1 mm and forces below 1 N. Evaluation v6 adds matching geometric flight with the existing 1.5%–45% bounds; all old and prior-speed checks remain. These are local research-informed adaptations, not proven convergence. Forty-nine CPU checks pass, including 600 bitwise standing/walking fixtures; actual 64-env running PPO and checkpoint reload pass with zero pre-update KL and command mismatch. Exactly three runtime files changed; 41 protected authored files, geometry, CAD/motor limits, actions, history, commands, PPO and iteration4 execution are identical. The CPU-only status recorder also gained tested exact output-path normalization for Windows slash spellings; it changes no runtime.

Fresh R32 observation: status {state['status']}, stage {state['stage']}, current operation count {state.get('environment_count')}, selected next training count {state.get('next_training_environment_count')}, main updates {state['total_training_iterations']}; promotions {[(p['stage'],p['speed']) for p in state['promotions']]}. All 46 hashes verify and {s['actual_worker_count']} actual worker is owned. Read fresh ACTIVE_RUN/state instead of assuming this dated stage. Completed capacity at 6144 gave about 46.1k active PPO and 32.0k worker-effective samples/s, with 1081 MiB minimum free GPU and 7790 MiB host available. The 8192 capacity attempt hit the 300 MiB reserve and was rejected; target8192, fallback and the 180-second healthy-memory ramp remain. Capacity is not learned-policy progress or full-chain throughput.

Next priorities: (1) complete matching fresh standing/walking/stopped/prior-speed gates, preserve improving actors, then compare learned geometric flight and retained speed tracking against the grounded baseline; (2) if genuine flight or retention still plateaus, test intended two-foot flight force signs/distribution, effective clipped action/target response and full objective/cost tradeoffs in controlled physics before another focused repair; (3) monitor moving capacity, memory, saved fallback/full-state recovery, healthy ramps, throughput and exact CAD motor velocities. About 35.1% of audited raw running actions exceeded the +/-3 clamp. Sampled exact CAD leg speeds show no sustained relevant velocity ceiling; substep clipping is unexcluded. Force unloading, smoke updates, reward or a 5 m/s command never qualify. Keep this same adaptive three-hour heartbeat active until verified actual 5 m/s or a reproduced relevant actuator velocity ceiling, then save, pause and brief. Leave the computer on.'''
next_check={'recorded_utc':now,'active_run':str(run),'status':state['status'],'stage':state['stage'],
 'checkpoint':state.get('checkpoint'),'checkpoint_sha256':s.get('checkpoint_sha256'),
 'initial_actor_checkpoint':seed['checkpoint'],'initial_actor_sha256':seed['sha256'],
 'source_hashes':s['hashes'],'main_updates':state['total_training_iterations'],
 'current_environment_count':state.get('environment_count'),'fresh_promotions':state['promotions'],
 'historical_exact_qualified_speeds':[0.,.35,.65,1.],
 'historical_qualification_scope':'Exact preserved old checkpoints only; new reward/flight gates require fresh qualification',
 'resources':{'gpu_used_free_utilization_mb_pct':s['gpu_used_free_utilization_mb_pct'],'host_available_mb':s['host_available_mb']},
 'throughput_scope':'R32 capacity/startup is not main PPO throughput; use fresh progress and attempts',
 'report':str(OUT/'REPORT.md'),'tests':{'cpu_checks':len(test['checks']),'running_ppo_load_passed':True},
 'focus':focus,'next_tests':['Complete fresh independent standing/walking/retention, then compare true geometric running flight.',
 'If flight/retention stalls, controlled intended-flight force/objective/effective-target diagnosis before focused repair.',
 'Verify moving capacity, memory fallback/full-state recovery/180s ramps and exact CAD motor speed trends.'],
 'endpoint':'Actual signed >=4.95m/s matching nominal/pushed and retained skills, >=32 trials/command, or reproduced relevant velocity ceiling; save/pause/brief; computer stays on'}
(HERE/'automation/NEXT_CHECK.json').write_text(json.dumps(next_check,indent=2)+'\n',encoding='utf-8')
automation=Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml')
config=tomllib.loads(automation.read_text(encoding='utf-8'))
(OUT/'automation_before.json').write_text(json.dumps(config,indent=2)+'\n')
args=read(HERE/'diagnostics/adaptive_check_20261004_1137/automation_update_args.json')
assert config['id']==args['id']
args.update(name=config['name'],status='ACTIVE',rrule=config['rrule'],prompt=config['prompt'].split('Latest dated focus,',1)[0]+focus)
for key in ('targetThreadId','notificationPolicy'):
    if key in config:args[key]=config[key]
args['targetThreadId']=config.get('target_thread_id',args['targetThreadId'])
(OUT/'automation_update_args.json').write_text(json.dumps(args,indent=2)+'\n')
print(json.dumps({'report':str(OUT/'REPORT.md'),'status':state['status'],'stage':state['stage'],'count':state.get('environment_count'),'documents':len(paths)+2,'automation_keys':list(config)},indent=2))
