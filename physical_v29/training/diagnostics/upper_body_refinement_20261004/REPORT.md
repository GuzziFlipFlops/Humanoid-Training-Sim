# Upper-body fine-tuning while preserving learned locomotion

Recorded October 04, 2026 21:12 EDT (2026-10-05T01:12:06.516268+00:00 UTC).

The user's request is implemented as training-only head/arm refinement in R35.
It continues the exact current R34 full checkpoint at the existing 1.5 m/s stage,
with all 1473 completed main updates preserved.
The actor, critic, Adam moments and trained exploration were loaded exactly.
No training from scratch, actor-only transfer, critic reset or rollback to an
older flight actor was used. The eight-update diagnostic actors are discarded.

## Reproduced diagnosis

The fully restored CAD videos show actual policy posture, not missing head/arm
parts. At the drained actor the full-population 1.5 m/s head combined RMS yaw/pitch
error is 51.71 degrees.
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

Run: E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261004_r35_upper_body_refinement

Continued checkpoint: E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261004_r34_bilateral_swing_unloading\086_run_150_train_023\candidate.pt

SHA256: 6611cb45edce83bf91abe42e971f8413716f77001d8732b7ff07c49c22f85702

Training was temporarily drained at an owned saved boundary for the physical
tests and resumed via the new frozen continuation after
461.5 seconds. Checkpoint and source lineage
are bound by upper_continuation_proof.json, full_state_smoke.json and deployment.json.
The existing1e-4 restart LR ceiling applies; the tested actual adaptive LR is1e-5.
No learned optimizer moments or exploration state were reset.

Fresh status 2026-10-05T01:12:06.516268+00:00: running, stage run_150, operation
E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261004_r35_upper_body_refinement\011_run_150_train_001, count 8192, completed new main updates
0. Startup/capacity tests are not main learning.
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


Verified actual main continuation at 2026-10-05T01:12:13.258378+00:00: 8192 environments, 26 observed running PPO updates, exact full actor/critic/Adam/exploration load, zero pre-update KL/command mismatch and20 upper-refinement minibatches per update. One actual CUDA worker and47 hashes verify. Latest free GPU485MiB/host6075MiB. The same three-hour automation is verified ACTIVE with head/arm diagnostics added and cadence/thread/endpoint preserved. This is partial main learning, not independent posture or running qualification. CPU history inventory35 physical runs,1192 operations,0 differing hashes/0 read warnings;8 historical missing manifest artifacts remain. Report: E:\Aditya\robot_sim\physical_v29\training\diagnostics\upper_body_refinement_20261004\REPORT.md.
