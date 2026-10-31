"""Durable dated evidence, rejected iteration and next experiment; CPU only."""
import json,hashlib,shutil,tomllib,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
live=read(HERE/'diagnostics/walk_regression_20261003/live_handoff.json')
run=Path(live['active_run']['run']);s=live['state'];pause=read(OUT/'diagnostic_pause.json');resume=read(OUT/'diagnostic_resume.json')
assert run.name=='locomotion_20261004_r35_upper_body_refinement' and s['status']=='running'
assert s['supervisor_pid']==resume['supervisor_pid'] and len(live['actual_worker_pids'])==1
assert len(live['frozen_source_integrity'])==47 and all(live['frozen_source_integrity'].values())
write(OUT/'owned_observation_at_report.json',live);write(OUT/'state_at_report.json',s)
comparison=read(OUT/'matched_comparison.json');physical=read(OUT/'head_causal_analysis.json')
baseline=read(OUT/'latest_main_baseline.json');parity=read(OUT/'physical_parity.json');cpu=read(OUT/'cpu_command_audit.json')
assert parity['all_bitwise_equal'] and parity['cases_equal'] and comparison['full_reload']['passed']
decision={'selected':'preserve_same_R35','candidate':'gait_consistent_running_frontier_v2',
    'deployed':False,'why':'Matched candidate loses nominal0.35 retention and has less independently measured flight than control; posture also worsens slightly.',
    'same_checkpoint_source':pause['checkpoint'],'same_checkpoint_sha256':pause['checkpoint_sha256'],
    'diagnostic_actors_deployed':False,'training_source_changed':False,'reward_or_caps_changed':False,
    'evidence':'matched_comparison.json','qualification_inferred':False}
write(OUT/'decision.json',decision)
gates=[]
for f in sorted(run.glob('*_gate_*')):
    if not (f/'evaluation.json').is_file():continue
    ev=read(f/'evaluation.json');cfg=read(f/'run_config.json')
    gates.append({'folder':str(f),'checkpoint':cfg['arguments']['checkpoint'],'perturb':ev['perturb'],
        'cases':[{'command':c['statistics']['command_m_s'],'speed':c['statistics']['reset_frame_forward_speed_m_s'],
            'flight':c['statistics']['geometric_flight_fraction'],'survival':c['statistics']['survival'],
            'passed':c['passed'],'failures':c['failures']} for c in ev['cases']]})
write(OUT/'gate_trends_final.json',gates)
pairs=[]
for cp in dict.fromkeys(x['checkpoint'] for x in gates):
    both=[x for x in gates if x['checkpoint']==cp]
    nominal=[x for x in both if not x['perturb']];pushed=[x for x in both if x['perturb']]
    if nominal and pushed:pairs.append([nominal[-1],pushed[-1]])
pair=pairs[-1]
main=[]
for f in run.glob('*_train_*'):
    if not (f/'result.json').is_file() or not (f/'progress.jsonl').is_file():continue
    result=read(f/'result.json')
    if not result.get('completed'):continue
    rows=[json.loads(x) for x in (f/'progress.jsonl').read_text().splitlines()]
    wrapper=next((a for a in s['attempts'] if Path(a['folder']).resolve()==f.resolve()),None)
    if wrapper:main.append((rows,wrapper['seconds']))
completed_samples=sum(sum(x['num_envs']*24 for x in rows) for rows,_ in main)
active_seconds=sum(sum(x['collect_seconds']+x['learn_seconds'] for x in rows) for rows,_ in main)
worker_seconds=sum(seconds for _,seconds in main)
all_pauses=[]
for p in HERE.glob('diagnostics/*/diagnostic_pause.json'):
    a=read(p);r=p.parent/'diagnostic_resume.json'
    if a.get('run')==str(run) and r.is_file():all_pauses.append(read(r)['pause_seconds'])
elapsed=live['recorded_unix']-s['started_unix']-sum(all_pauses)
throughput={'completed_main_samples':completed_samples,'completed_main_blocks':len(main),
    'active_collection_PPO_samples_s':completed_samples/active_seconds,
    'main_worker_effective_samples_s':completed_samples/worker_seconds,
    'full_chain_completed_main_samples_s_excluding_all_R35_diagnostic_pauses':completed_samples/elapsed,
    'excluded_diagnostic_pause_seconds':sum(all_pauses),
    'scope':'Completed main blocks only. Active excludes startup/exit/gates; worker-effective includes main-worker startup/exit; whole-chain includes capacity/gates/resource failures but excludes the two owned diagnostic pauses. Small matched diagnostic samples are excluded.'}
evidence={'recorded_utc':datetime.now(timezone.utc).isoformat(),'current_run':str(run),
    'latest_matched_production_pair':pair,'current_main_updates':s['total_training_iterations'],
    'comparison':comparison,'head_causal_behavior':physical['post_transient_behavior'],
    'decision':decision,'throughput':throughput,'latest_main_baseline':baseline,
    'physical_parity':parity,'same_supervisor_resume':resume,'running_qualified':False,'velocity_ceiling_proven':False}
write(OUT/'evidence.json',evidence)
stamp=datetime.now(timezone(timedelta(hours=-4))).strftime('%B %d %H:%M EDT')
report=OUT/'REPORT.md'
text=f'''# Adaptive training check — {stamp}

**R35 continues on its original supervisor, preserving the forward head and full learned state. The proposed running-command exposure change was tested and rejected. Running and all-prior retention remain incomplete.** No training runtime, reward, observation, action, CAD, motor cap or optimizer setting was deployed this check.

## Fresh production evidence

The check began with2560 completed new R35 updates;2688 were preserved at the pause boundary, plus1473 prior R34 updates. Exact saved checkpoint135 SHA256 `{pause['checkpoint_sha256']}` remains intact. The latest matched full192-robot,20-second nominal/pushed pair available at report creation is shown below. Each command has32 independent trials. Original qualified standing/walking actors are still preserved; their certificates do not qualify today's learner.

| Command m/s | Nominal signed m/s | Pushed signed m/s | Complete pair |
|---|---|---|---|
'''
for a,b in zip(pair[0]['cases'],pair[1]['cases']):
    text+=f"| {a['command']:.2f} | {a['speed']:.3f} | {b['speed']:.3f} | {'Pass' if a['passed'] and b['passed'] else ', '.join(sorted(set(a['failures']+b['failures'])))} |\n"
text+='''
All recent cases survive, but flight and some earlier commands remain unqualified. At the2560-update matched pair130/131, head combined yaw/pitch RMS at1.5 was3.798/3.801 degrees, versus51.71 before upper refinement;90.0%/90.7% of sampled frames put both head axes within5degrees. Other moving commands range roughly4.3–5.6degrees RMS. The5degree band is diagnostic, not a qualification gate. Arm compactness improved previously, but complete natural arm swing is not certified. Current upper progress must not be discarded simply to retrieve an older flight actor. See [upper audit](upper_audit.json), [production gate history](gate_trends_final.json) and [owned evidence](owned_observation_at_report.json).

## Closed-loop physical separation of head feedback and head motion

Training paused at an owned saved boundary, supervisor45868 was suspended, and the eval worker drained. One diagnostic simulator at a time used unchanged gravity, base freedom, self collision, contacts and CAD/motor constraints.128 robots,32 per group, ran20seconds. Four groups compared currentR35, the preservedR34 flight actor038, old body with native-zero head targets and actual feedback, and the same native-zero head control with ONLY head q/qd/previous-action history supplied from the simultaneous original-head cohort. After5seconds both neutral-head groups have exactlyzero head actions and targets. No root support or pose writes were used.

| Policy/control diagnostic | Head RMS after5s | Actual flight | Absolute lateral m/s | Final heading error |
|---|---|---|---|---|
'''
for row in physical['post_transient_behavior']:
    text+=f"| {row['group']} | {row['head_rms_deg_after5s']:.2f}° | {100*row['flight_fraction_after5s']:.3f}% | {row['abs_lateral_speed_after5s']:.3f} | {row['final_abs_yaw_deg']:.2f}° |\n"
text+='''
Every robot survives. Under the same native-zero physical head control, changing head-feedback history changes final heading from70.25degrees to0.89degrees and lateral speed from1.409 to0.109m/s, while preserving about1.79% genuine flight in the latter. This is causal evidence that the old body controller depends strongly on its original head-feedback distribution. Actual head dynamics also differ slightly after body trajectories diverge; this does not prove mechanical effects absent. The experiment is a diagnostic, not a qualified hybrid controller or permission to falsify production observations. The original actor's51.8degree head pose is not restored in production. All diagnostic controllers/inputs are discarded.

[Result](head_causal_response/result.json), [post-transient and motor analysis](head_causal_analysis.json), [reviewed plot](head_causal_comparison.png), [full saved trace](head_causal_response/telemetry.npz).

The exact50Hz CAD Jacobian/reduction audit validates motor-speed units by directional finite differences. Current right ankle motor2 has72.8% p99 of cap and0.281% samples at/above cap; original flight actor has84.0% p99 and0.309% samples at/above cap. Other relevant leg motors are farther below their caps. Isolated spikes, including transient dynamics above nominal cap, do not prove a sustained locomotion ceiling; substep clipping/torque attribution remains untested. All caps are unchanged.

## Training-command hypothesis, controlled comparison and rejection

The old1.5 stage samples32% stopped,24% earlier commands and44% frontier. Only about24% of all robots receive exact1.5 running rewards: the remaining20% frontier samples lie between1.2 and1.5 and use walking duty0.55, while1.5 uses running duty0.43. Production block135 confirms23.980% running-command exposure. This is a reproduced command/reward boundary, not proof it causes the plateau.

A separately staged candidate moves that20% frontier into the current running gait, retaining all stopped/prior draws and unchanged evaluation/inference.55 CPU checks pass; every <=1.0 and >=1.875 sampler output is bitwise equal, as are all prior/stopped cohorts. A same-actor192-robot20-second replay under the candidate is bitwise identical in all saved startup/physical/gait arrays and all evaluation cases. No geometry, reward weight/kernel, action semantics or gate changed.

Matched control and candidate each load checkpoint135's EXACT actor, critic, Adam and exploration, use4096 environments,64 PPO updates, seed71, the same qualified prior teacher and upper loss, and the same existing restart LR ceiling. Adaptive PPO LR remains adaptive. Full192-robot20-second nominal901/pushed1301 evaluations include32 trials per command. Every update has zero pre-update KL and observed/action-command difference. A further actual64-env4-update candidate reload proves actor/critic/Adam/exploration equality. These tests check safe execution, not convergence.

| Matched experiment | Running exposure | Nominal/pushed genuine flight | Nominal0.35 | Running head RMS nominal/pushed |
|---|---|---|---|---|
'''
for name in ['control','candidate']:
    a=comparison['arms'][name];n=a['nominal'][-1];p=a['pushed'][-1]
    text+=f"| {name} | {a['training']['mean_running_fraction']*100:.2f}% | {n['flight_fraction']*100:.3f}% / {p['flight_fraction']*100:.3f}% | {'Pass' if a['nominal'][2]['passed'] else 'Fail'} | {n['head_RMS_deg']:.2f}° / {p['head_RMS_deg']:.2f}° |\n"
text+=f'''
**Reject the candidate.** It reduces flight relative to control, loses a nominal0.35 retention pass, and slightly worsens head posture. Neither arm qualifies the complete running stage.64updates on one matched seed do not rule out every future exposure strategy, but they do not justify deploying this one now. Both diagnostic learned actors are discarded; the exact production checkpoint and source stay preserved. Candidate code/logs/checkpoints remain nearby for reproducibility; they are not a new active training revision.

[CPU tests](cpu_command_audit.json), [physical parity](physical_parity.json), [full matched comparison](matched_comparison.json), [decision](decision.json), [candidate reload proof](candidate_full_reload/full_load_proof.json).

## Research, interpretation and next narrow experiment

[Siekmann et al., Periodic Reward Composition](https://arxiv.org/html/2011.01387v2) describes complementary force/velocity behavior in swing and stance, with gait parameters supplied to the policy. It motivated auditing this robot's gait boundary, not copying numerical coefficients or asserting that extra running exposure must help. The controlled comparison rejects that simple remedy.

[Ross, Gordon and Bagnell, DAgger](https://proceedings.mlr.press/v15/ross11a.html) addresses imitation under the observation distributions induced by a learner's own actions. It supports investigating the measured change in head-feedback distribution and collecting teacher/student counterfactuals on current states. This is a research lead; no DAgger algorithm or teacher conditioning was deployed and the paper does not guarantee this robot's convergence.

Next prioritize a bounded training-only posture-aware locomotion transfer test on current neutral-head states, with matched control and full exact-state preservation. First reproduce the head-feedback effect at retained0.35/0.65/1.0 commands and verify any teacher conditioning under normal physics before trusting its targets. Then measure teacher/student lower-output and effective target errors, objective costs, per-command gradient contributions, and matched independent direction/flight/retention/head behavior. Production observations must remain truthful. Do not roll back to038, deploy action hybrids or reference-input rewrites, increase motor caps, or repeatedly tune arbitrary weights. Do not repeat the rejected exposure change without new evidence. Preserve the qualified walking teacher and038 as separate evidence.

## Resume, memory, actual throughput and provenance

The SAME owned supervisor resumed at{resume['recorded_utc']} after{resume['pause_seconds']:.1f}seconds. Checkpoint135 and source bytes are unchanged; no diagnostic actor was deployed. The eval136 wrapper duration includes this pause and must not be treated as simulation throughput. At the owned report snapshot{live['recorded_utc']}, status{s['status']}, stage{s['stage']}, {s['total_training_iterations']} completed new main updates, operation `{Path(s['operation_path']).name}`. One actual CUDA worker,47 matching authored source hashes. GPU sample `{live['gpu']}`; host available{live['system_memory']['available_mb']}MiB.

Completed-main throughput is{throughput['active_collection_PPO_samples_s']:.0f} active collection+PPO samples/s,{throughput['main_worker_effective_samples_s']:.0f} main-worker-effective samples/s, and{throughput['full_chain_completed_main_samples_s_excluding_all_R35_diagnostic_pauses']:.0f} whole-chain completed-main samples/s excluding both R35 diagnostic pauses({throughput['excluded_diagnostic_pause_seconds']:.1f}s). Denominators differ; diagnostic A/B samples are excluded. Latest completed production block135 gives{baseline['active_samples_per_sec']:.0f} active samples/s, minimumGPU{baseline['min_GPU_MiB']}MiB andhost{baseline['min_host_MiB']}MiB; pre-update KL0. This is not the old~9k bottleneck. Running failure still requires behavior/learning diagnosis.

Target8192,300MiB reserve, host guards, fallback and180continuously healthy training-second upward ramp remain. Iteration4 execution optimizations, environmentv8/verifierv6,23outputs/400inputs, full checkpoint recovery and all independent current/prior gates remain untouched. The physical model still uses simplified collision envelopes; earlier full-CAD frame QA and fulfilled recordings do not certify every detailed CAD collision. Unrelated unsaved Fusion work is preserved. No GUI repeat was requested or performed this check.

[Pause](diagnostic_pause.json), [resume](diagnostic_resume.json), [fresh source/process evidence](owned_observation_at_report.json), [throughput/evidence](evidence.json). A stale diagnostic-recorder path lookup failed once and was corrected to the canonical training/record_owned_observation_handoff.py; no simulator or training fault resulted. Nonfatal Kit shader-cache messages and all experiment logs are preserved. Neither actual5m/s nor a relevant persistent motor ceiling is proven. Keep the same three-hour heartbeat active; save/pause/brief at its verified endpoint, leave the computer on.
'''
report.write_text(text,encoding='utf-8')
nextfile=HERE/'automation/NEXT_CHECK.json';nextcheck=read(nextfile)
priorities=[
    'Posture-aware locomotion transfer: reproduce head-history causal effect at prior speeds; validate teacher targets under normal physics, then bounded training-only current-state A/B with exact full state. No production observation spoof, rollback or hybrid.',
    'Compare matching nominal/pushed flight AND0.15/0.35/0.65/1.0 retention/head/arm traces. Preserve improvement; do not repeat rejected24-to44% exposure without new evidence.',
    'Moving6144/8192 memory, full-state fallback/recovery and180healthy ramp; exact CAD motor attribution and actual samples/time.'
]
nextcheck.update(recorded_utc=evidence['recorded_utc'],active_run=str(run),report=str(report),
    latest_observation=str(OUT/'owned_observation_at_report.json'),selected_actor={'checkpoint':pause['checkpoint'],
    'sha256':pause['checkpoint_sha256'],'full_actor_critic_Adam_exploration':True},running_qualified=False,
    focused_tests=priorities,latest_verified_state=live,latest_repair_decision=decision,
    throughput=throughput,new_main_updates=s['total_training_iterations'])
write(nextfile,nextcheck)
marker='adaptive_check_20261005_0540'
entry=f'''\n\n<!-- {marker} -->
## {stamp} — R35 exact resume, causal head-feedback diagnosis, rejected running exposure

Read [the dated report]({report.as_posix()}). R35 at2688 new updates preserves exact135 checkpoint788837ad7b431073575a2e3aa3ad0460fe1f97b3d480336a8e8c6e01576d06fb. No deployed source/reward/CAD/cap/optimizer reset. Head remains about3.4–3.8deg RMS at1.5, not fully natural arms or complete running.128-robot20s causal test: same native-zero head actions/targets in both neutral-head cohorts; original head-history feedback restores heading70.25→0.89deg and lateral1.409→0.109m/s with1.793% actual flight after5s. Diagnostic inputs only; no production spoof, no mechanic-only conclusion. Actual1.5 running commands occupy24% despite44% frontier; separately staged44% exposure passes55CPU and entire192-robot20s same-actor bitwise replay, exact4096env64update A/B plus4update full reload. Candidate loses nominal0.35 retention and has0/0.0208% flight versuscontrol0.656/0.757%; reject and discard both diagnostic actors. Current learner/supervisor45868 resumes after{resume['pause_seconds']:.1f}s, one actual worker47hashes,6144production;8192target/300reserve/180healthy ramp unchanged. Reported throughput{throughput['active_collection_PPO_samples_s']:.0f}active/{throughput['main_worker_effective_samples_s']:.0f}worker-effective/{throughput['full_chain_completed_main_samples_s_excluding_all_R35_diagnostic_pauses']:.0f}whole-chain completed-main samples/s excluding allR35diagnostic pauses. Research sources:Periodic Reward Composition andDAgger, hypotheses not guarantees. Next controlled posture-aware teacher/student transfer on current truthful observations and retained commands, before justified full-state continuation. All exact gates mandatory,5m/s/velocity-ceiling endpoint unproven, same3h automationactive, computeron.
'''
for p in [HERE/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),
          Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    assert p.is_file(),p
    if marker not in p.read_text(encoding='utf-8-sig'):
        with p.open('a',encoding='utf-8') as f:f.write(entry)
brief=f'''<!-- {marker}_current -->
## Latest R35 causal head-input check: {stamp}

Read {report} and fresh ACTIVE_RUN/state. Same exact learner resumed; no source/reward/limit reset. Head about3.4–3.8deg at1.5, running/retention incomplete. Same native-zero head control plus original head-feedback history restores old actor heading70.25→0.89deg in diagnostic physics; no production observation rewrite. Matched24→44% running-exposure candidate worsens flight/retention and is rejected. Next focused posture-aware teacher/student transfer and matched independent gates; preserve full state.47hashes,6144production,target8192/reserve300/180healthy ramp. Same3h endpoint and computer-on scope remain.

'''
for p in [HERE/'automation/THREE_HOUR_CHECK.md',
    Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),
    Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]:
    assert p.is_file(),p
    original=p.read_text(encoding='utf-8-sig')
    if marker+'_current' not in original:p.write_text(brief+original,encoding='utf-8')
configpath=Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml')
cfg=tomllib.loads(configpath.read_text(encoding='utf-8'));write(OUT/'automation_before.json',cfg)
focus=f'''Latest dated focus, {stamp} ({evidence['recorded_utc']}): R35_upper_body_refinement remains the same full-state learner, preserved at2688 new main updates/checkpoint135 SHA{pause['checkpoint_sha256']}. Head at1.5 remains about3.4–3.8degRMS versus51.71originally; arms not fully natural. Latest saved production matched pairs survive but running and some prior0.65/1.0 cases fail. Read diagnostics/adaptive_check_20261005_0540/REPORT.md, evidence.json, head_causal_analysis.json, matched_comparison.json, decision.json and owned_observation_at_report.json. Read freshACTIVE_RUN/state instead of assuming this dated actor/stage persists.

A128-robot20s closed-loop causal test holds native-zero head actions/targets in both neutral-head groups. With truthful current head feedback the original038 body actor turns70.25deg and lateral1.409m/s; with only original head q/qd/previous-action histories supplied diagnostically, heading0.89deg/lateral0.109 and actualflight1.793% after5s. Heads stay near2deg and all survive. This reproduces learned head-feedback distribution dependence, not proof mechanics absent. No input spoof, head override, hybrid or diagnostic actor is deployed. Exact50Hz CAD legmotor samples show isolated ankle cap crossings, no sustained ceiling; substep clipping remains unexcluded.

The1.5 training seam is audited:24%allrobots receive running rewards while20%frontier1.2–<1.5 receive walking duty/costs. A separately staged44%running-exposure candidate passes55CPU checks, exact192-robot20s same-actor replay and exact full-state4096env64update matchedPPO/reload with0KL/command mismatch. It worsens nominal0.35 retention and flight:0/0.0208% nominal/pushed versuscontrol0.656/0.757%; headRMSalso slightlyworse. Candidate REJECTED, canonical/live frozen commands/rewards unchanged, both diagnostic learned actors discarded. Do not repeat this exposure change without new evidence or cite the seam as proven primarycause. Current head progress and qualified walking teacher/038 evidence preserved.

Same supervisor45868 resumes after{resume['pause_seconds']:.1f}s, same checkpoint/source/full learned state. Pause is excluded from throughput; eval136 wrapper includes it. Owned snapshot{live['recorded_utc']} shows stage{s['stage']}, {s['total_training_iterations']}completednewmain,6144env,oneactualworker,47hashes. Production completed-main throughput about{throughput['active_collection_PPO_samples_s']/1000:.1f}kactive/{throughput['main_worker_effective_samples_s']/1000:.1f}kworker-effective/{throughput['full_chain_completed_main_samples_s_excluding_all_R35_diagnostic_pauses']/1000:.1f}kwhole-chain excluding bothdiagnostic pauses; denominators differ. Latest6144freeGPUabout1.7GiB/host7.5GiB, originaltarget8192/reserve300/fallback/180healthy ramp remain. No runtime/reward/geometry/cap/optimizer reset thischeck, no GUI repeat pending.

Next one to three priorities: (1) prioritize bounded training-only posture-aware locomotion transfer on CURRENT neutral-head states: reproduce head-feedback effect at0.35/0.65/1.0, physically validate any teacher conditioning/targets, measure clipped lower-output/target errors and gradients, then exact-full-state matchedcontrol/PPO/gates before deployment. DAgger observation-distribution research is a lead, not an installed algorithm or convergence guarantee. Never falsify production observations, restore tilted-head policy, deploy hybrids or raise caps. Do not spend another check solely reward-identical waiting. (2) compare newer matching nominal/pushed actualflight ANDallprior speed/head/arm retention; preserve independently improvingactors. (3) moving6144/8192memory, full-state fallback/recovery/180healthy ramp and actualCADvelocity attribution. All existing gates/endpoints remain. No5m/s or relevantvelocityceilingproven; same adaptive3hheartbeatACTIVE, save/pause/brief withcomputeron at verifiedendpoint.'''
args={'mode':'update','id':cfg['id'],'kind':'heartbeat','name':cfg['name'],
    'prompt':cfg['prompt'].split('Latest dated focus',1)[0]+focus,'status':'ACTIVE','rrule':cfg['rrule'],
    'targetThreadId':cfg['target_thread_id']}
if 'notification_policy' in cfg:args['notificationPolicy']=cfg['notification_policy']
write(OUT/'automation_update_args.json',args)
print(json.dumps({'report':str(report),'automation_args':str(OUT/'automation_update_args.json'),
    'decision':decision,'throughput':throughput},indent=2))
