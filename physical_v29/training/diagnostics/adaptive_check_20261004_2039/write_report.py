"""Durable evidence and focused self-updating heartbeat handoff, CPU only."""
import json,tomllib
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1];ROOT=HERE.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
deployment=read(OUT/'deployment.json');validation=read(OUT/'validation_proof.json')
report=f'''# Three-hour check: bilateral flight unloading, October 4, 2026, 20:39 UTC

R33 recovered stopped and all walking commands through 1.0 m/s, but repeatedly regressed to a grounded fast walk at the 1.5 m/s running command. Its final full checkpoint is preserved at **1,401 new main updates**. R34 is a fresh frozen continuation from the earlier R33 checkpoint021 mean and trained exploration, selected because matching full nominal/pushed tests retain every earlier walking case and it still reproduces genuine flight. Running is not qualified. No 5 m/s endpoint or actuator velocity ceiling has been established.

## Reproduced failure and focused repair

Reward v7 averaged separate exponentials of intended swing-foot forces. During the two-foot flight window, redistributing the same 338.085 N total load from equal feet to one loaded foot raises that component from 0.135335 to 0.509158. This rewards an asymmetric grounded configuration without unloading the robot. In a 96-robot normal-actuator physical replay, the final actor had zero geometric flight; in its flight windows the larger foot force averaged 370.278 N and the smaller force zero, while the old quality averaged 0.526313.

Reward v8 applies the same exponential, scale and weight to the **sum of intended swing-foot loads**. Constant total load now gives the same score regardless of redistribution; single-foot swing feedback is bitwise unchanged. This is a local adaptation, not a claim to copy a published running reward exactly. Every previous reward weight, walking/standing term, flight definition, motor limit, physics byte, history/action/command contract and independent classifier remains unchanged. The narrow reward correction does not by itself guarantee learned take-off.

The [Periodic Reward Composition paper](https://arxiv.org/html/2011.01387v2) describes phase-conditioned force and foot-velocity costs, with both feet contributing to bipedal gait specification. It motivates the force-based diagnosis; the bounded simultaneous-load transformation here is our declared local adaptation.

## Physical and gradient evidence

- Saved 20-second final nominal gate071 passes stopped, 0.15, 0.35, 0.65 and 1.0 m/s; current running signed speed is 1.472773 m/s with zero genuine flight. Earlier matching gate065/066 also retains all prior cases and has zero/0.09375% flight.
- In the controlled 96-robot, 10-second experiment, all robots survive with zero resets, normal gravity, normal motor limits, no pose writes or root impulses. R33 checkpoint021 reproduces 1.3770% geometric flight; a temporary 0.30-rad coordinated retraction raises it to 3.4188%. The final actor remains grounded even under the tested pulses. The original R32 flight actor reproduces 2.2792%. These 12-robot groups are diagnostic, not qualification; no pulse is deployed.
- Re-scoring actual force/phase/posture traces reproduces the frozen unloading term within 9.99e-8. The corrected term removes grounded load-transfer credit. The retraction still reduces total reward despite increasing flight: other objective tradeoffs remain relevant if the narrow repair plateaus. Reward dominance alone is not physical convergence.
- A passive capture of one actual PPO minibatch gives a parameter-gradient cosine of **+0.113962** between running PPO and walking-prior retention, with norms 5.2480 and 0.08456. There is no direct prior derivative on running rows. This sample does not demonstrate opposing gradients; it also does not exclude conflict in other batches or long-term interference. Preserve the effective walking retention rather than removing it without evidence.
- Final actor reconstructed 2,240 saved actions within 3.82e-6. Commands match; 37.66% of sampled raw running outputs exceed the existing +/-3 action clamp. Exact pose-dependent CAD motor speeds show no sustained locomotion-relevant ceiling; sampled ankle spikes and elbow motion are not a velocity-limit endpoint. Substep clipping remains unexcluded.

## Tests and deployment

Eleven focused CPU checks reproduce/fix the redistribution incentive and verify monotonic force derivatives, boundedness, symmetry and zero no-swing credit. **600 standing/walking full reward fixtures are bitwise identical.** Eight full-cycle suites through 5 m/s rank ideal alternating running above fast walking, sliding, in-place movement, backwards movement and synchronous hopping. Six negative transfer checks reject changed contract, reward signature/version, asset, motion and actuator metadata.

Real 64-environment running PPO actor-transfer and subsequent full actor/critic/Adam/exploration reload pass. All eight smoke pre-update KL values and action-command differences are zero. Smoke actors are discarded. A same-actor **192-robot, 20-second** physical nominal replay is bitwise identical in every saved array and every case result. This demonstrates unchanged execution/physics and inference; it does not qualify the new learning actor.

Exactly two authored runtime files changed: stage_rewards.py and reward_warm_start.py. The latter strictly permits the audited v7 signature and physical contract; it does not broaden acceptance to arbitrary actors. Forty-two protected authored files remain byte-identical. Selected mean and trained exploration are exact; critic/Adam are fresh because the objective changed. All nominal/pushed/stopped/prior-speed qualification must be repeated under R34.

R33 supervisor was suspended at a saved boundary, its eval worker completed and drained, and its full checkpoint was preserved. It was retired only after the diagnostics and candidate tests passed. R34 launched at {deployment['recorded_utc']} in `{Path(deployment['run']).name}`. Target 8192, 300 MiB GPU reserve, host guards, automatic fallback and a continuous 180-second healthy-memory upward ramp remain. The separate frozen worker and automatic curriculum continue through actual 5 m/s with unchanged limits and gates.

## Throughput and resource observations

R33 had two GPU-reserve recoveries and fell back to 4096. Across logged main attempts, active collection-plus-PPO throughput is 35.48k samples/s and worker-effective throughput 29.43k samples/s; these differ from whole-chain throughput, which includes initialization and gates. The last completed 4096 block is 26.38k active and 21.42k worker-effective, with 1437 MiB minimum free GPU, 9154 MiB minimum host available and 6 MiB private growth. Do not extrapolate earlier 8192 speed to these 4096 blocks. R34 capacity must be measured anew rather than reused across a changed objective.

## Diagnostic corrections and scope limits

Initial re-scoring used an incorrect reward-term label, then action-time phase instead of the post-step reward phase, and finally an unexpanded phase mask. Those rejected attempts are preserved in diagnostic_attempts.json; no conclusion or runtime repair used their failed results. The corrected calculation is verified against the actual saved reward term. Earlier failures and smoke evidence remain intact.

The separate [video-body report](../video_body_review_20261004/REPORT.md) explains the missing-looking body: clips render the simplified box training asset, while the source CAD export still has its 140 meshes. A full-visual playback asset is prepared but not rendered or installed. Simplified collision envelopes also underrepresent some physical body surfaces, so present gates are simulation-contract qualification, not full-CAD collision clearance certification. Do not silently migrate current CADv32 into the frozen v29 training robot. Preserve unrelated unsaved Fusion work.

## Next focused check

1. Compare matched full v8 nominal/pushed flight and every prior-speed case against the preserved checkpoint021 baseline; preserve improving actors. Complete qualification before promoting. If simultaneous unloading still plateaus, test full objective/cost tradeoffs and effective clipped leg targets in controlled physics rather than repeating reward-identical waiting or arbitrary weights.
2. Monitor actual moving 6144/8192 memory, saved fallback/full-state recovery, 180-second ramps and active/worker/whole-chain denominators. The old reserve failure inflated marginal VRAM estimates; measure capacity before changing the guard.
3. Inspect running force distributions, cadence, geometric flight, direction and exact CAD motor speeds, plus additional measured PPO/prior gradient batches if interference is suspected. Retain the separate full-body visual and collision-fidelity audit as a documented limitation.

Evidence: initial_live_handoff.json, gate_trends.json, diagnostic_pause.json, physical_response/result.json and telemetry.npz, physical_objective_audit.json, gradient_analysis.json, running_audit.json, bilateral_counterfactuals.json, candidate_cycle_tests.json, candidate_proof.json, validation_proof.json, selected_seed.json, throughput.json and deployment.json. Fresh observation is saved separately in replacement_status.json and live_handoff.json.
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
attempts={'rejected_cpu_audits':[{'issue':'Incorrect term label term_running_phase_unload; actual saved term is running_unload','runtime_changed':False},
    {'issue':'Action-time phase used to reconstruct post-step reward; maximum disagreement 0.248764. Withheld scoring until exact post-step counter reconstruction','runtime_changed':False},
    {'issue':'Phase mask initially had one environment dimension; expanded to actual robot population before per-group indexing','runtime_changed':False}],
    'corrected_audit':'physical_objective_audit.json','corrected_max_term_error':read(OUT/'physical_objective_audit.json')['frozen_term_reconstruction_max_error']}
(OUT/'diagnostic_attempts.json').write_text(json.dumps(attempts,indent=2)+'\n')
note='''\n\n## October 4, 2026, 20:39 UTC check - R34 bilateral swing unloading

R33 is preserved at 1401 new main updates. All stopped/prior walking commands through 1.0 recovered, but 1.5 m/s genuine flight plateaued at zero. CPU and actual-physics tests reproduced a v7 load-redistribution incentive during intended two-foot flight; v8 exponentiates the summed swing loads with the existing weight and scale. Single-foot swing and 600 standing/walking fixtures are bitwise unchanged. One real PPO parameter-gradient sample did not show running/prior opposition; walking retention remains. R34 retains exact checkpoint021 mean/exploration, fresh critic/Adam for the changed objective and mandatory fresh qualification. Two runtime files changed, 42 protected authored files identical; real PPO/full reload and all 192-robot/20-second replay arrays/cases match. Normal-limit probes reproduce flight; no helper pulse is deployed. Target8192/reserve300/180s healthy ramps and all gates remain. See E:/Aditya/robot_sim/physical_v29/training/diagnostics/adaptive_check_20261004_2039/REPORT.md and replacement_status.json. No running-stage/5m/s/velocity-ceiling completion inferred. The separate video-body audit documents box-only visuals and simplified collision-fidelity limits.\n'''
for path in [HERE/'ITERATION_HISTORY.md',HERE/'automation/THREE_HOUR_CHECK.md',
    ROOT/'docs/robot/TRAINING_JOURNAL_20261001.md',
    Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md'),
    Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md'),
    Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md')]:
    assert path.is_file(),path
    old=path.read_text(encoding='utf-8-sig')
    if '20:39 UTC check - R34 bilateral swing unloading' not in old:path.write_text(old+note,encoding='utf-8')
toml=Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml')
config=tomllib.loads(toml.read_text(encoding='utf-8-sig'))
(OUT/'automation_before.json').write_text(json.dumps(config,indent=2)+'\n')
focus='''Latest dated focus, October4 20:39 UTC check: R33 is preserved at1401 new main updates, final checkpoint070. It recovered every stopped/prior walking command through1.0m/s in matching full nominal/pushed tests, but running regressed to zero genuine flight across repeated tests despite about1.47m/s travel and100% survival. Read diagnostics/adaptive_check_20261004_2039/REPORT.md, bilateral_counterfactuals.json, physical_objective_audit.json, gradient_analysis.json, validation_proof.json, deployment.json and replacement_status.json. A reproduced v7 incentive averages separate swing-foot force exponentials: constant338N total load redistribution raises quality from0.1353 to0.5092. Actual flight-window final-actor forces average0/370N with quality0.5263. R34_bilateral_swing_unloading fixes the aggregation to exponentiate summed intended swing loads, retaining the existing0.5 weight and quarter-body-weight scale. Single-foot swing and600 standing/walking fixtures are bitwise identical. Eight running-cycle suites through5m/s and six negative transfer checks pass. A96-robot10s normal-limit physical test reproduces1.377% flight for R33checkpoint021,3.419% with a diagnostic-only0.30rad pulse,2.279% for the original flight actor, and0% for the final actor. No pulse or root support is deployed. One actual PPO parameter-gradient sample has running/prior cosine+0.114 and norm5.248/0.0846, not evidence of opposition; other batches remain unexcluded. Qualified prior retention remains unchanged. The correction removes a demonstrated defect; total reward still penalizes the tested retraction and convergence is unproven. R34 retains exact021 mean/trained exploration, fresh critic/Adam and requires fresh full nominal/pushed/stopped/prior-speed qualification. Two runtime files change,42 protected authored files identical; real64-env PPO and full-state reload pass with all eight pre-update KL/command mismatches zero. Entire same-actor192-robot20s physical nominal replay arrays/cases are bitwise identical. Smoke actors are discarded. R33 reserve breaches forced4096; last main block gave26.38k active/21.42k worker-effective samples/s,minGPU1437MiB,host9154MiB,private growth6MiB. New capacity is measured, target8192/reserve300/fallback/180s hold retained. Next compare new matched flight AND all earlier-speed retention against checkpoint021, preserving improvement; if stalled investigate full objective/cost balance and effective clipped leg-target response, not arbitrary weights or reward-identical waiting. Monitor moving capacity, recoveries and actual samples/time; inspect more actual PPO/prior gradient samples if conflict is suspected. Saved final actor action reconstruction matches2240 samples within3.82e-6,commands agree,37.66% raw actions exceed the existing+/-3 clamp. Exact CAD motor samples do not establish a relevant velocity ceiling; substep clipping is unexcluded. The separate diagnostics/video_body_review_20261004 report confirms videos used box visuals while140 original CAD meshes remain; full-visual playback asset is prepared but not rendered, and simplified collision envelopes limit physical full-body certification. Preserve that limitation and unsaved Fusion work; do not silently migrate currentCADv32 into frozen trainingv29. Read fresh ACTIVE_RUN/state, not dated status. Keep this same adaptive three-hour heartbeat active until verified5m/s or a causally demonstrated relevant motor velocity ceiling; save,pause,brief with computer on.'''
prompt=config['prompt'].split('Latest dated focus,')[0]+focus
args={'id':config['id'],'mode':'update','kind':'heartbeat','name':config['name'],'prompt':prompt,
    'rrule':config['rrule'],'status':config['status'],'destination':'thread','targetThreadId':config['target_thread_id']}
if 'notification_policy' in config:args['notificationPolicy']=config['notification_policy']
(OUT/'automation_update_args.json').write_text(json.dumps(args,indent=2)+'\n')
next_check={'recorded_utc':datetime.now(timezone.utc).isoformat(),'automation_id':config['id'],'active_run':deployment['run'],
    'report':str(OUT/'REPORT.md'),'latest_observation':str(OUT/'replacement_status.json'),'selected_actor':read(OUT/'selected_seed.json'),
    'running_qualified':False,'qualification_scope':'Read exact saved actor matching full gates; startup/smoke is not qualification',
    'prior_run_main_updates_preserved':1401,'focused_tests':['Compare matched v8 genuine flight and all retained speeds vs selected021 baseline; preserve measurable progress',
    'If plateaued, physical full-objective/cost/clipped-target tradeoffs; additional actual PPO/prior gradients before removing retention',
    'Moving6144/8192 memory, checkpoint-bound fallback and180s healthy ramps; actual CAD motor speeds, separate full-body collision-fidelity limitation'],
    'endpoint':'Actual fixed-frame>=4.95m/s at5 command with all matching gates, or demonstrated relevant velocity ceiling; save,pause training and same automation,brief; leave computer on'}
(HERE/'automation/NEXT_CHECK.json').write_text(json.dumps(next_check,indent=2)+'\n')
print(json.dumps({'report':str(OUT/'REPORT.md'),'handoff':str(HERE/'automation/NEXT_CHECK.json'),'automation_args_prepared':True}))
