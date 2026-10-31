"""Evidence-backed local report and same-thread adaptive check priorities."""
import argparse,json,hashlib,shutil,tomllib
from pathlib import Path
from datetime import datetime,timezone
from dateutil.tz import gettz
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1];ROOT=HERE.parents[1]
p=argparse.ArgumentParser();p.add_argument('--finalize',action='store_true');args=p.parse_args()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
now=datetime.now(timezone.utc);local=now.astimezone(gettz('America/New_York'))
deployment=read(OUT/'deployment.json');run=Path(deployment['run']);state=read(run/'state.json')
load=read(OUT/'running_fullstate_smoke/full_load_proof.json');physical=read(OUT/'physical_validation.json')
current=Path(state['operation_path']) if state.get('operation_path') else None
progress=read(current/'progress.json') if current and (current/'progress.json').exists() else None
main=[]
for folder in sorted(run.glob('*_run_*_train_*')):
 if (folder/'progress.jsonl').exists():
  rows=[json.loads(s) for s in (folder/'progress.jsonl').read_text().splitlines()]
  if rows:main.append(dict(folder=str(folder),observed_updates=len(rows),first=rows[0],latest=rows[-1],
   full_load=read(folder/'full_load_proof.json'),maximum_pre_update_KL=max(r['ppo']['pre_update_kl'] for r in rows),
   maximum_command_difference=max(r['behavior']['action_command_difference'] for r in rows),
   minimum_gpu_free_mb=min(r['gpu']['free_mb'] for r in rows),minimum_host_available_mb=min(r['gpu']['ram_available_mb'] for r in rows),
   active_samples_per_second=sum(r['num_envs']*24 for r in rows)/sum(r['collect_seconds']+r['learn_seconds'] for r in rows)))
snapshot=dict(recorded_utc=now.isoformat(),run=str(run),status=state['status'],stage=state['stage'],operation=str(current),
 completed_main_updates=state['total_training_iterations'],prior_updates_retained=state.get('prior_training_iterations'),
 current_environment_count=state.get('environment_count'),checkpoint=state.get('checkpoint'),
 current_progress=progress,main_blocks=main,new_posture_qualified=False)
write(OUT/'fresh_training_status.json',snapshot)
failed=[dict(phase='candidate preparation',failure='Two reversible text guards caught an incorrect supervisor needle and a two-occurrence copy-list match before production edits',evidence='attempt01_prepare_guard/prepare_candidate.py',repaired=True),
 dict(phase='physical replay verification',failure='Diagnostic code assumed push/nominal share the same ordinal folder prefix; resolve exact checkpoint/seed/speed configuration instead',evidence='verify_validation_attempt01.py',repaired=True),
 dict(phase='CPU status metadata',failure='ACTIVE_RUN initially used selected_full_checkpoint instead of the recorder initialization field; added initial_full_checkpoint with the same verified SHA',repaired=True),
 dict(phase='manifest inventory',failure='New upper module was copied/frozen and migration-proof-bound, but absent from the legacy launcher hash list. Added its exact hash to generated metadata; canonical future launcher now includes it. No live frozen source changed.',evidence='inventory_supplement.json',repaired=True),
 dict(phase='report timestamp',failure='Windows Python lacks stdlib zoneinfo tzdata; use the existing dateutil America/New_York database. No training/environment install or source change.',repaired=True)]
write(OUT/'failed_attempts.json',failed)
report=f'''# Upper-body fine-tuning while preserving learned locomotion

Recorded {local:%B %d, %Y %H:%M %Z} ({now.isoformat()} UTC).

The user's request is implemented as training-only head/arm refinement in R35.
It continues the exact current R34 full checkpoint at the existing 1.5 m/s stage,
with all {deployment['retained_main_updates']} completed main updates preserved.
The actor, critic, Adam moments and trained exploration were loaded exactly.
No training from scratch, actor-only transfer, critic reset or rollback to an
older flight actor was used. The eight-update diagnostic actors are discarded.

## Reproduced diagnosis

The fully restored CAD videos show actual policy posture, not missing head/arm
parts. At the drained actor the full-population 1.5 m/s head combined RMS yaw/pitch
error is {physical['baseline_upper'][-1]['head_combined_rms_deg']:.2f} degrees.
Saved 50 Hz running samples command about +41.56 degrees yaw and -28.43 degrees
pitch; the physical head follows within about 0.14/0.24 degrees mean error, with
95th-percentile torques only 0.60/0.27 Nm. This supports an unwanted policy target
in these samples, not a demonstrated neck motor ceiling. Read head_target_audit.json.

The qualified standing/walking teacher also copied all eight head/arm outputs.
It could therefore preserve the undesirable upper posture. The native yaw/pitch
zero frames have identity rotation relative to the torso and align head +X with
torso +X; this is a relative head-forward target, not gaze stabilization against
arbitrary torso lean. Original compatible CAD visuals remain intact.

## Narrow adaptation and limits

Version upper_output_posture_v1 adds a bounded Huber auxiliary loss only to the
eight final actor output rows for head yaw/pitch, shoulder pitch/roll and elbows.
Detached hidden features prevent its derivative from updating the shared trunk
or fifteen torso/leg output rows. Ordinary PPO still trains the whole actor.
The qualified prior retains the exact original lower-output derivative and
23-output denominator; upper copying is removed from both stopped and moving
teachers. Shared PPO updates, global gradient clipping and physical coupling can
still change locomotion. This is not a guarantee of identical future leg motion.

Head targets use a 3-degree neutral band. Shoulder pitch may swing within about
26 degrees and shoulder roll within 10 degrees; elbows retain the checked -0.30
rad nominal flexion with 0.30 rad tolerance. A mild symmetry loss allows opposite
arm pitch/roll and equal elbow flexion. Head/arm/symmetry weights are 0.15/0.025/
0.025 with at most256 sampled observations per PPO minibatch and bounded slope.
These are declared local training adaptations, not Unitree's exact reward.
No hard inference pose, action override, teacher blending or gait helper is used.

The official [Unitree G1 velocity task](https://raw.githubusercontent.com/unitreerobotics/unitree_rl_lab/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py)
penalizes arm-joint deviation from nominal. This supports posture shaping as a
design pattern, not the numerical choices or convergence of this adaptation.
The [PyTorch hook contract](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.register_hook.html)
and actual local Torch2.7 CPU/CUDA tests support the implemented gradient routing.

Environment reward v8, verifier v6, all old gait/flight/direction/retention gates,
400-input23-output inference, observation history/actions, command distribution,
PPO algorithm configuration, geometry/mass/collision/CAD Jacobians, motor limits
and iteration4 execution are unchanged. Only the worker and new upper-loss module
change authored training runtime. The launcher routes strict full-state migration
and freezes the module. Existing frozen R34 and R35 source bytes remain immutable.

## Verification and remaining uncertainty

Thirty-three CPU checks pass: counterexamples, bounded extreme gradients, bitwise
unchanged initial actions, zero auxiliary hidden/lower gradients, exact lower
teacher derivatives and an isolated auxiliary update that reduces posture loss
without changing lower actions. Actual64-env8-update running PPO and full-state
reload pass; all8 pre-update KL and observed/action command differences are zero.
Both teachers and the upper loss run on genuine PPO minibatches. A192-robot,
20-second unchanged-actor nominal replay matches all cases and39 saved arrays
bit for bit. Subsequent smoke rollout has100% survival across32 trials at each
of six commands. This is execution validation, not a new posture qualification.

A matched no-refinement eight-update PPO control also drifts in slow speed.
The refined smoke narrowly fails0.65 speed at0.753 versus0.738 m/s control; both
fail0.35 speed. Neither achieves genuine running flight. Head error improves
slightly at several walking commands, but remains about51.7 degrees at1.5.
This does not show a completed upper-body correction. All old retention checks
remain mandatory; no smoke actor or relaxed gate is deployed. Read matched_control.json.

Diagnostic text guards and path/metadata mistakes are preserved in
failed_attempts.json and saved attempt scripts. The initially omitted new-module
hash was already bound to the migration proof; generated inventory now covers
47 authored hashes and the future canonical launcher includes it automatically.
The active frozen launcher's bytes were not edited. Preserve this manifest
supplement when comparing launcher versions in a later maintenance pass.

## Exact continuation and fresh status

Run: {run}

Continued checkpoint: {deployment['exact_continued_checkpoint']}

SHA256: {deployment['checkpoint_sha256']}

Training was temporarily drained at an owned saved boundary for the physical
tests and resumed via the new frozen continuation after
{deployment['training_pause_seconds']:.1f} seconds. Checkpoint and source lineage
are bound by upper_continuation_proof.json, full_state_smoke.json and deployment.json.
The existing1e-4 restart LR ceiling applies; the tested actual adaptive LR is1e-5.
No learned optimizer moments or exploration state were reset.

Fresh status {now.isoformat()}: {state['status']}, stage {state['stage']}, operation
{current}, count {state.get('environment_count')}, completed new main updates
{state['total_training_iterations']}. Startup/capacity tests are not main learning.
Read fresh_training_status.json and ACTIVE_RUN/state for newer evidence. Target
8192, reserve300MiB, host guards, fallback and180s healthy live-training ramps
remain. A capacity choice is not standing, walking or running qualification.

Next check must compare full-population per-command head RMS/p95, actual arm
angles/velocities/symmetry and full-CAD frames on the same exact actors, together
with nominal/pushed stopped/prior-speed retention and genuine running flight.
Use audit_upper_body.py on saved full-population telemetry. Do not declare natural
arms/head from reduced batch loss or a video alone, and do not restart an improving
policy merely to force faster posture change. Preserve checkpoint038 flight actor
and the qualified walking teacher. If upper posture stalls, inspect effective
clipped upper targets and actual auxiliary/PPO gradients before another repair.
The existing adaptive three-hour heartbeat remains; verified5m/s or a causally
established locomotion-relevant velocity ceiling still saves/pauses/briefs with
the computer on. No such endpoint is established.
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
focus=f'''Latest dated focus: user-directed upper-body update, {local:%B %d %H:%M %Z} ({now.isoformat()} UTC): R35_upper_body_refinement is an exact full-state continuation of R34 checkpoint086 after1473 main updates, SHA6611cb45edce83bf91abe42e971f8413716f77001d8732b7ff07c49c22f85702. The user explicitly requests keeping torso/leg/running progress while making the head forward and arms less unnatural; do not start over or roll back to an older actor to implement posture. Actor, critic, Adam and exploration are exact; the current adaptive LR remains1e-5 under the existing ceiling. Read diagnostics/upper_body_refinement_20261004/REPORT.md, physical_validation.json, matched_control.json, upper_continuation_proof.json, deployment.json and fresh_training_status.json. The qualified teacher's original direct lower gradient is preserved and upper copying excluded. A bounded head/arm/symmetry auxiliary derivative trains only eight final actor output rows, with no shared-trunk or direct lower-row derivative, no inference hooks/action override. Ordinary PPO, global clipping and physical coupling can still affect legs; gates remain mandatory. Environment v8 rewards, verifier v6, all physics/CAD/motor constraints, commands/history/actions and iteration4 are unchanged. This is a local training adaptation, not copied Unitree numerical weights. Head native zero is torso-forward. Saved physical targets reproduce unwanted+41.6deg yaw/-28.4deg pitch with small tracking error; sampled motor torque does not establish a ceiling.

33 CPU checks, actual64-env8-update PPO/full-state reload and entire192-robot20s unchanged-actor replay pass;39 arrays and all cases bitwise identical. Every smoke KL/command mismatch is zero. No smoke actor is deployed. Matched control/refinement small-batch tests both drift in speed; refinement narrowly fails0.65 at0.753 versus0.738 control and both fail0.35, all survive. Running head RMS remains51.7deg after eight smoke updates; posture correction and running remain unqualified. Preserve both completed and failed diagnostic attempts.47 authored hashes now include the new module through an explicit generated-manifest supplement; active frozen launcher bytes unchanged, canonical future launcher fixes its inventory coverage. Read fresh sources/proof rather than confusing that metadata-only canonical difference with a live physics change.

Fresh observed status: {state['status']}, stage{state['stage']}, operation{current}, count{state.get('environment_count')}, new completed main{state['total_training_iterations']}. Do not assume startup/capacity is running progress. Read ACTIVE_RUN and actual fresh state. Same target8192/reserve300/host fallback/180s healthy live-training ramp, one actual CUDA worker, no stereo. Existing full-body videos are complete; no repeat pending recording request.

Next priorities: (1) Preserve exact useful state and compare post-refinement matching nominal/pushed per-command head RMS/p95, forward-neutral fraction, actual shoulder/elbow motion/velocities and full-CAD frames alongside stopped/prior walking retention and genuine running flight. Run the CPU-only audit_upper_body.py with --run CURRENT_RUN --output DATED_REPORT/upper_audit.json. A5deg reporting band is diagnostic only, no gate relaxation or posture success inference. If posture improves, keep learning; if it stalls, measure raw versus effective clipped upper targets, actual auxiliary/PPO gradient magnitudes and physical response before a narrow change. (2) If running flight still plateaus, resume the preserved phase-target/full-objective normal-limit controlled diagnosis and multiple actual PPO/prior gradient tests; do not attribute flight loss to posture merely from a behavioral tradeoff or spend the next check waiting without diagnosis. Keep earlier flight actor038 and the qualified walking teacher. (3) Verify exact full-state recovery, memory and effective samples/time including new bounded auxiliary cost, retain8192 target and reserves. All current/prior gates remain. Finish by updating this same three-hour heartbeat, journals/history and NEXT_CHECK. No5m/s or relevant motor ceiling is proven; endpoint remains save,pause,brief with computer on.'''
(OUT/'automation_focus.txt').write_text(focus,encoding='utf-8')
if args.finalize:
 assert main and all(m['maximum_pre_update_KL']==0. and m['maximum_command_difference']==0. for m in main)
 note=f'''\n\n## {local:%B %d, %Y %H:%M %Z}: R35 head/arm refinement, exact learned-state continuation\n\nUser requests natural arms/head-forward while preserving ongoing running. R34 is preserved at1473 completed main updates and exact086 full checkpoint6611cb45edce83bf91abe42e971f8413716f77001d8732b7ff07c49c22f85702. R35 retains actor/critic/Adam/exploration and continues1.5 stage; no from-scratch or actor-only reset. Training-only bounded upper_output_posture_v1 changes only eight final output rows; prior lower derivative identical, upper copying excluded. Original rewardv8/verifierv6/actions/history/commands/physics/CAD/motor limits and iteration4 unchanged.33 CPU tests; actual8-updatePPO/full-state reload;192-robot20s pre-learning replay39 arrays/cases bitwise; matched control and refinement both show small-batch speed drift, refinement narrowly worsens0.65, all survive. Posture remains unqualified and physical coupling/normalPPO can affect locomotion. Exact original seed, never smoke actor, deployed.47 source hashes include a generated metadata supplement; frozen launcher unchanged, canonical future inventory fixed. Preparation/path/metadata guard failures preserved. One actual worker, resource guards and mandatory matching gates remain. First main evidence:{len(main)} block(s), latest{main[-1]['observed_updates']} observed updates, exact full load, KL/command mismatch0; no posture/running qualification. Report:{OUT/'REPORT.md'}. Same3h heartbeat now tracks per-command upper posture/frame QA and simultaneous flight/retention.\n'''
 for file in [HERE/'ITERATION_HISTORY.md',ROOT/'docs/robot/TRAINING_JOURNAL_20261001.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
  with file.open('a',encoding='utf-8') as f:f.write(note)
 brief=f'''<!-- upper_body_refinement_20261004_current -->\n## Current upper-body continuation: R35\n\n{local:%B %d %H:%M %Z}: Read {OUT/'REPORT.md'} and fresh ACTIVE_RUN/state. User-directed head-forward/calm-arm refinement continues exact R34checkpoint086 actor/critic/Adam/exploration after1473updates at1.5m/s; no training reset. Auxiliary final-upper-output rows only, prior lower derivative preserved; runtime rewards/physics/CAD/caps/gates unchanged.33CPU tests, realPPO/full load and39-array unchanged-actor bitwise replay pass. Small-batch gait drift remains and posture is not qualified. R35 actual main PPO is verified, not inferred from capacity; inspect current matching gates and upper audit.47 hashes include documented metadata supplement; active frozen sources untouched. Same adaptive3h check now includes per-command upper telemetry/full-CAD frames alongside flight and retention. Do not confuse dated lower sections with current status.\n<!-- /upper_body_refinement_20261004_current -->\n\n'''
 for file in [Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]:
  file.write_text(brief+file.read_text(encoding='utf-8'),encoding='utf-8')
 nxt=read(HERE/'automation/NEXT_CHECK.json');nxt.update(recorded_utc=now.isoformat(),active_run=str(run),report=str(OUT/'REPORT.md'),
  latest_verified_state=snapshot,focused_tests=['Per-command independent head/arm telemetry and full-CAD frame review together with flight and all retained speed gates; preserve improving full state',
  'If flight still plateaus, controlled phase-target/objective and actual PPO/prior gradients; distinguish posture coupling from running failure',
  'Moving6144/8192 full-state recovery,memory,180healthy training ramp and actual samples/time'],
  upper_refinement=dict(version='upper_output_posture_v1',exact_full_seed=deployment['exact_continued_checkpoint'],qualified=False,diagnostic=str(OUT/'audit_upper_body.py'),main_evidence=main),
  selected_actor=dict(checkpoint=deployment['exact_continued_checkpoint'],sha256=deployment['checkpoint_sha256'],full_actor_critic_Adam_exploration=True))
 write(HERE/'automation/NEXT_CHECK.json',nxt)
 with (HERE/'automation/THREE_HOUR_CHECK.md').open('a',encoding='utf-8') as f:f.write(brief)
print(json.dumps(dict(report=str(OUT/'REPORT.md'),main_blocks=len(main),state=snapshot['stage'],operation=str(current),finalized=args.finalize)))
