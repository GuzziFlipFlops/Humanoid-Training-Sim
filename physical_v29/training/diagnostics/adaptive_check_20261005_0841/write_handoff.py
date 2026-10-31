"""Refresh concise current references without rewriting older history."""
import json,tomllib,time
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
live=read(HERE/'diagnostics/walk_regression_20261003/live_handoff.json');run=Path(live['active_run']['run']);s=live['state']
assert run.name=='locomotion_20261005_r36_posture_conditioned_priors' and s['status']=='running'
assert s['stage']=='run_150' and len(live['actual_worker_pids'])==1
assert len(live['frozen_source_integrity'])==49 and all(live['frozen_source_integrity'].values())
write(OUT/'final_owned_observation.json',live)
decision=read(OUT/'decision.json');deploy=read(OUT/'deployment.json');progress=live.get('worker_progress') or {}
current=Path(s['operation_path'])
main=[Path(a['folder']) for a in s['attempts'] if a['operation']=='train' and '_run_150_train_' in Path(a['folder']).name and a.get('completed')]
if '_run_150_train_' in current.name and current not in main:main.append(current)
assert main,'No genuine new main operation yet'
first=main[0];proof=read(first/'full_load_proof.json')
assert proof['source_sha256']==deploy['checkpoint_sha256'] and all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
rows=[json.loads(r) for r in (first/'progress.jsonl').read_text().splitlines()]
assert len(rows)>=4 and all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
assert all(r['locomotion_retention']['teacher_context_version']=='teacher_head_phase_context_v1' for r in rows)
post=dict(recorded_utc=live['recorded_utc'],folder=str(first),observed_main_updates=len(rows),completed=first in [Path(a['folder']) for a in s['attempts'] if a.get('completed')],
  full_load=proof,maximum_pre_update_KL=0,maximum_command_difference=0,
  conditioned_prior_samples=sum(r['locomotion_retention']['teacher_phase_conditioned_samples'] for r in rows),
  minimum_free_GPU_MiB=min(r['gpu']['free_mb'] for r in rows),minimum_host_MiB=min(r['gpu']['ram_available_mb'] for r in rows),
  private_memory_growth_MiB=rows[-1]['gpu']['process_private_mb']-rows[0]['gpu']['process_private_mb'],
  torch_allocation_growth_MiB=rows[-1]['gpu']['torch_allocated_mb']-rows[0]['gpu']['torch_allocated_mb'],
  environments=rows[-1]['num_envs'],active_samples_s=sum(r['num_envs']*24 for r in rows)/sum(r['collect_seconds']+r['learn_seconds'] for r in rows),
  scope='Actual deployed main updates, not capacity/smoke or independent locomotion qualification.')
write(OUT/'new_main_evidence.json',post)
active=read(HERE/'ACTIVE_RUN.json');active['status']='running';write(HERE/'ACTIVE_RUN.json',active)
focused=[
 'Compare R36 matching nominal and pushed stopped and 0.15/0.35/0.65/1.0 retention, direction and head/arm telemetry after 64, 128 and 256 updates. Preserve improved actors. One diagnostic 64-update pass does not establish long-term stability or qualify the current running actor.',
 'If prior retention holds but genuine flight remains stalled, run normal-limit phase, take-off and full objective/target/cost diagnostics on the current truthful neutral-head states. Inspect effective clipped outputs and CAD response. Running-only row transfer lost its flight gain by 128 updates; do not repeat it or the rejected 44% exposure change without new evidence, spoof observations, roll back posture or deploy hybrids.',
 'Verify moving 6,144/8,192 memory, exact full-state fallback/recovery and 180 continuous healthy training seconds. Report active, worker-effective and whole-chain denominators. Preserve the 300 MiB reserve, 8,192 target, one actual worker and native CAD motor caps; sampled cap spikes do not prove a ceiling.'
]
brief=f'''<!-- adaptive_check_20261005_0841_current -->
## Latest R36 posture-conditioned walking-prior continuation: October05

Read E:\\Aditya\\robot_sim\\physical_v29\\training\\diagnostics\\adaptive_check_20261005_0841\\REPORT.md and fresh ACTIVE_RUN/state. R36 preserves exact R35checkpoint213 actor/critic/Adam/exploration after4352new plus1473inheritedmainupdates. Only walking-teacher targets at0.15/0.35/0.65/1.0 are conditioned for reproduced tilted-head feedback dependence; student observations/inference are truthful and head/arm loss,rewards,gates,geometry/CADcaps/PPO/iteration4 remain. Three runtime files change,44 protected unchanged,49 authoredmanifesthashes verify. Running-only finalrow transfer loses initial flight by128updates and is rejected; no diagnostic actor deployed. All-prior-only matched4096env64update comparison passes every stopped/prior case in full192robot20s nominal/pushed tests and improves headRMS, but running flight0/0.0174% remains unqualified.27CPU+12migration checks, actualPPO/fullload and39array whole-replay parity pass. Deployed main evidence: {post['observed_main_updates']}observedupdates at{post['environments']}, exact213fullstate, zeroKL/command mismatch. Read new_main_evidence.json; partial learning is not qualification. Source-frozen R35 remains preserved;3731.7s diagnosticpause excluded. Keep sameadaptive3hcheck and computer-on endpoint. Next retention stability, then controlled current-state flight/objective diagnosis; target8192/reserve300/180healthy ramp unchanged.

'''
for p in [HERE/'automation/THREE_HOUR_CHECK.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md'),
 Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md')]:
 text=p.read_text(encoding='utf-8-sig')
 if '<!-- adaptive_check_20261005_0841_current -->' not in text:p.write_text(brief+text,encoding='utf-8')
entry=f'''\n\n<!-- adaptive_check_20261005_0841 -->
## October05 04:41 EDT — R36 walking teacher-target correction with exact learned state

[Dated report]({(OUT/'REPORT.md').as_posix()}). R35 preserved at4352newmain+1473priorupdates, checkpoint213 SHA{deploy['checkpoint_sha256']}. Normal-limit closed-loop640robot20s plus0.15cohorts reproduce older teacher head-history dependence; diagnostic phase context restores teacher gait with actual neutral head. No diagnostic spoof/override/hybrid deployed. Three isolated learning iterations: running-row imitation shows0.490/0.576%flight after64 but loses it by128(reject); partial-prior+running variant overspeeds0.15(reject); all-four prior targets only passes stopped and all walking cases through1.0 in matched4096env64update/full192robot20s nominal+pushed gates, improvesheadRMS3.1–3.5deg, still0/0.0174%runningflight. Selected adaptation changes teacher labels, not student truthful observations/inference; exact .25 MSE formula/23denominator and originalhead/armloss preserved, numerical targets/gradients intentionally changed. Three runtimefiles,44protected unchanged;49manifesthashes verify.27CPUcounterexamples,12strictmigrationchecks, actual64envPPO/fullload and entire39array192robot20s same-actor replay pass. R36 startsoriginal213 full actor/critic/Adam/exploration with existing restartLRceiling1e-4, not anA/Bactor. Owned old supervisor safely stopped after fresh freeze; diagnosticpause3731.7s. Deployed main {post['observed_main_updates']}observedupdates at{post['environments']}, zeroKL/command mismatch, exactfullload; training remains unqualified forrunning/5m/s. OldR35 throughput43.53kactive/37.49kworker-effective/24.48kwholechainexpauses; newpartialactive{post['active_samples_s']:.0f}is not completedblock/wholechainrate. MinGPU{post['minimum_free_GPU_MiB']}MiB/host{post['minimum_host_MiB']}MiB. ActualCAD50Hzspeeds do not prove relevantceiling. Failedpreparations and0.15metadataerratum preserved. ExistingfullCADvideos fulfilled,no newGUIclaim. Same3hheartbeat nextpriorstability/current-stateflightdiagnosis; target8192/reserve300/hostfallback/180healthyhold and unchangedallgates. Computeron.
'''
for p in [HERE/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),
 Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
 if '<!-- adaptive_check_20261005_0841 -->' not in p.read_text(encoding='utf-8-sig'):
  with p.open('a',encoding='utf-8') as f:f.write(entry)
nextpath=HERE/'automation/NEXT_CHECK.json';n=read(nextpath)
if not (OUT/'next_check_before.json').exists():write(OUT/'next_check_before.json',n)
n.update(recorded_utc=live['recorded_utc'],active_run=str(run),report=str(OUT/'REPORT.md'),latest_observation=str(OUT/'final_owned_observation.json'),
 selected_actor=dict(checkpoint=s['checkpoint'],initial_checkpoint=deploy['exact_continued_checkpoint'],initial_sha256=deploy['checkpoint_sha256'],full_actor_critic_Adam_exploration=True),
 running_qualified=False,prior_run_main_updates_preserved=4352,inherited_main_updates_preserved=1473,
 new_main_updates=s['total_training_iterations'],observed_partial_main_updates=post['observed_main_updates'],focused_tests=focused,
 latest_verified_state=dict(recorded_utc=live['recorded_utc'],status=s['status'],stage=s['stage'],operation=str(current),environments=s['environment_count'],
 actual_workers=live['actual_worker_pids'],frozen_hashes_verified=49,full_snapshot=str(OUT/'final_owned_observation.json'),gpu=live['gpu'],host=live['system_memory']),
 latest_repair_decision=decision,deployment=str(OUT/'deployment.json'),new_main_evidence=str(OUT/'new_main_evidence.json'),
 throughput=dict(previous_R35=read(OUT/'throughput.json'),new_partial_main=post))
write(nextpath,n)
cfgpath=Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml')
cfg=tomllib.loads(cfgpath.read_text(encoding='utf-8'));write(OUT/'automation_current_before_update.json',cfg)
assert cfg['id']=='humanoid-training-research-and-repair-every-four-hours' and cfg['kind']=='heartbeat' and cfg['rrule']=='FREQ=HOURLY;INTERVAL=3'
prefix=cfg['prompt'].split('Latest dated focus',1)[0].rstrip()
focus=f'''Latest dated focus, October 5, {live['recorded_utc']}: R36_posture_conditioned_priors continues the exact full R35 checkpoint 213, SHA {deploy['checkpoint_sha256']}, after 4,352 new and 1,473 inherited main updates. Read diagnostics/adaptive_check_20261005_0841/REPORT.md, evidence.json, decision.json, all_prior_comparison.json, matched_extension_comparison.json, teacher_context_continuation_proof.json, deployment.json and new_main_evidence.json. Read fresh ACTIVE_RUN and state before assuming this dated run persists. Head and arm progress is retained; no diagnostic learned actor is deployed.

Controlled physical cohorts reproduce the older walking teacher's dependence on its tilted-head feedback at 0.15, 0.35, 0.65 and 1.0 m/s. Phase-conditioned teacher context restores its gait with an actual neutral head. Only training labels are corrected; the production student receives truthful observations and inference has no teacher, observation spoof, head override or hybrid. Three learning variants are preserved. Running-only final-body-row imitation gained 0.490%/0.576% flight at 64 updates but returned to zero by 128 and retained a 0.65 failure; reject it. Partial-prior plus running transfer oversped at 0.15; reject it. The selected all-four walking-prior correction, with no new running imitation, passes all stopped and prior cases through 1.0 in a matched 4,096-environment, 64-update test and independent 192-robot, 20-second nominal/pushed tests. Signed speeds are 0.180/0.181, 0.382/0.377, 0.682/0.680 and 0.995/0.996 m/s. Head RMS improves to about 3.1–3.5 degrees; natural arm motion is not certified. The 1.5 command produces 1.420/1.419 m/s but true flight is 0%/0.0174%, below the unchanged 1.5% minimum. This repairs retention; it does not qualify running or establish long-term stability.

The original 0.25 prior MSE formula, 23-output denominator, stopped teacher and upper-only posture gradient remain. Numerical walking labels and gradients intentionally change. Teacher conditioning requires exact commands in all five history frames, valid phase and current head within 10 degrees. Three runtime files change; 44 protected files are byte-identical and 49 frozen hashes verify. Reward v8, verifier v6, commands, history, actions, CAD physics, native motor caps, PPO, iteration 4 execution and all gates remain. Twenty-seven CPU checks, twelve strict migration checks, real PPO/full-state reload and a bitwise 39-array, 192-robot, 20-second unchanged-actor replay pass. R36 starts the original 213 actor, critic, Adam and exploration with the existing 1e-4 restart learning-rate ceiling. The old teacher is freshly requalified; current and prior gates must still pass on the actual learned actor.

Fresh main evidence: {post['observed_main_updates']} actual updates at {post['environments']} environments, exact full-state load, zero maximum pre-update KL and command mismatch, and conditioned prior targets active during training. Minimum free GPU memory is {post['minimum_free_GPU_MiB']} MiB and host available is {post['minimum_host_MiB']} MiB. Partial active throughput of {post['active_samples_s']:.0f} samples/s is not completed-block or whole-chain throughput. One actual worker and 49 hashes verify. Exclude the 3,731.7-second diagnostic pause and earlier R35 pauses from timing; eval 214's wrapper includes the pause. R35 completed-main rates were 43.53k active, 37.49k worker-effective and 24.48k whole-chain samples/s excluding pauses. Target 8,192, the 300 MiB reserve, host fallback and 180-second healthy-training ramp remain. Exact CAD motor speeds sampled at 50 Hz do not prove a persistent relevant ceiling; substeps remain untested. Full-body videos are already fulfilled and no recording request is pending.

Next focused tests: (1) {focused[0]} (2) {focused[1]} (3) {focused[2]} Keep this same adaptive three-hour heartbeat active until verified 5 m/s or a causally established relevant actuator ceiling. At that endpoint save, pause and brief with the computer on.'''
args=dict(mode='update',id=cfg['id'],kind='heartbeat',name=cfg['name'],status='ACTIVE',rrule=cfg['rrule'],
 targetThreadId=cfg['target_thread_id'],prompt=prefix+'\n\n'+focus)
if 'notification_policy' in cfg:args['notificationPolicy']=cfg['notification_policy']
write(OUT/'automation_update_args.json',args)
with (OUT/'REPORT.md').open('a',encoding='utf-8') as f:
 f.write(f"\n\n## Fresh deployed main evidence\n\nSnapshot {live['recorded_utc']}: {post['observed_main_updates']} genuine main updates at{post['environments']}, exact213 full actor/critic/Adam/exploration, zero KL/command mismatch, {post['conditioned_prior_samples']} conditioned teacher samples, min GPU{post['minimum_free_GPU_MiB']}MiB andhost{post['minimum_host_MiB']}MiB. One actual CUDA worker and49 hashes verified. Startup/capacity updates are excluded. These observations do not qualify running or current posture. [Main evidence](new_main_evidence.json), [full owned observation](final_owned_observation.json).\n")
print(json.dumps(dict(recorded_utc=live['recorded_utc'],run=str(run),main=post,automation_args=str(OUT/'automation_update_args.json')),indent=2))
