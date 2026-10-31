# Five-hour humanoid check — October 7, 08:44 UTC

Same R40 learner/supervisor39000 is running; latest completed main block uses6144 environments, with one actual worker and55 verified frozen files. The owned snapshot operation is145_run_350_train_046 at6144 robots; evaluation count is not PPO capacity. 2905 completed +41 hash-proven recovered = 2946 retained current-run updates, plus27507 inherited. Exact 142_run_350_train_045 matching nominal/pushed signed3.5 speeds 3.476/3.487m/s, lateral 0.337/0.310; current stage remains unqualified. Gates bind this saved actor, never newer partial learning.

No production source, reward, shoulder gain, motor cap, actor or optimizer reset was deployed. The exact original learner resumed after a 1918.3-second controlled suspension, including 32.7 seconds of worker drain. Current upper refinement and both prior auxiliaries remain at20 batches/update. Latest completed main minimum free GPU is1536MiB,host available6253MiB,private memory growth7MiB and Torch allocation growth0MiB. Highest separately certified stage remains3m/s on preserved teacher349; original267's all-prior pair is preserved. There is no qualified3.5,4,5m/s or relevant motor-ceiling endpoint.

## Current evidence

Owned snapshot: 2026-10-07T09:33:36.556612+00:00. Saved pair: [142_run_350_train_045](E:/Aditya/robot_sim/physical_v29/training/runs/locomotion_20261007_r40_goal_budget_full_state/142_run_350_train_045/candidate.pt), SHA`3261dafd001c16008ff834f9188eabc06b6eecb700c7cb7cb72c06cbf798cba8`. Each matching test has320 robots,20 seconds,32 trials per command. Current/prior failures: nominal: 0.15 (forward_net_travel), 1.5 (geometric_flight), 2.5 (lateral_speed/world_lateral_speed), 3 (world_lateral_speed), 3.5 (world_lateral_speed); pushed: 0.15 (forward_net_travel), 1.5 (geometric_flight), 2.5 (lateral_speed/world_lateral_speed), 3 (world_lateral_speed), 3.5 (world_lateral_speed). Ordinary nominal resets remain cloned; trial count alone does not establish initial-state diversity.

Retained-main throughput is 44.23k active collection+PPO / 38.21k worker-effective / 25.40k whole-chain samples/s. Startup, capacity and baseline gates enter the whole-chain denominator; the documented successful pause is excluded when before the endpoint. The first brief failed bookkeeping pause lacks exact timestamps and remains included, making this rate slightly conservative. Diagnostic, smoke and capacity actors never count as production learning. Exact full actor/critic/Adam/exploration loads,zero pre-update KL and action-command mismatch are verified. All55 frozen hashes and53 protected authored worker files match. Target8192,reserve300MiB,host guards,fallback and180 continuous healthy training seconds before upward ramps remain.

## Tested teacher-label hypothesis — rejected

The certified neutral-head349 teacher still receives head-history labels fitted to an older tilted-head walker. Four actual current-state PPO minibatches reproduce lower effective-label differences0.112–0.200 action units. Running PPO norms17.54–38.31 versus prior1.99–2.20 have cosine–0.126 to+0.296; this is not consistently opposed. The candidate removes legacy head context only for exact teacher349, with truthful student observations and unchanged0.25MSE/23-output denominator, stopped teacher, upper refinement, PPO, physics, actions, native caps and gates.

Ten CPU ownership/gradient command fixtures, actual64-env4-update PPO/full-state reload and exact4096-env64/128 controls complete. Both branches start exact139 SHA`8fa38ecdb9abc20d4cf6465b9cee254b8387477cf857e29844ce220a7d552527` with actor/critic/Adam/exploration equal. Every pre-update KL/command mismatch is zero. Fresh truthful349 nominal/pushed288-robot tests pass all earlier cases. Experimental overrides are explicitly outside original signature certification; no experimental actor is deployable under that signature.

|128-update result|Control|Candidate|
|---|---:|---:|
|3.5 lateral, nominal/pushed (m/s)|0.383/0.353|0.362/0.343|
|0.15 gait/net travel|Fails|Passes|
|1.5 case, nominal/pushed|Passes|Fails lateral|
|2.5 case, nominal/pushed|Passes|Fails slip|
|3.5 direction|Fails|Fails|

An additional pushed seed1901 reproduces the tradeoff. A separate seed2301 nominal-physics test uses32 independently generated small q/qd reset states per command, with saved arrays/seeds/hashes and unchanged classifiers. It reproduces candidate1.5 lateral and2.5 slip failures;3.5 lateral is0.379 control versus0.365 candidate. All survive. This establishes limited reset robustness of the comparison, not qualification of production or the final endpoint. Candidate deployment is rejected; useful full state and all diagnostic checkpoints are preserved.

## Actual upper-body timelines

Actual head yaw/pitch, shoulder pitch/roll and elbows were saved as angles,physical targets and velocities for initial133, both128 diagnostic actors and latest production142_run_350_train_045; every angle/velocity plot was visually inspected. Population sampling is32 robots/command at10Hz; detailed traces are16 representative robots/gate at50Hz, reset-segmented. At initial133/1m shoulders range39.37/30.78deg with hip phases–159.8/–166.2deg. At3.5 range14.20/7.49deg, phases140.9/149.5deg and mean biases–13.63/+4.16deg. Slow bias drift at2.5/3 is distinguished from periodic swing. Latest data and coordination metrics are in [final_upper_summary.json](final_upper_summary.json).

Asymmetric fast swing,roll/elbow coordination and gait-linked head oscillation remain. Both matched actors show target excursions larger than measured fast shoulder response. These diagnostics do not certify human-like arms,exact Unitree similarity or world optical gaze. Prior Isaac head/FK parity is known; original CAD face/camera optical axis remains uncalibrated. Shoulder stiffness70,uniform lateral exposure and previous failed kernels remain rejected.

## Direction objective lead

A108-case CPU audit plus gravity-alignment fixture confirms the pinned Unitree tracker is yaw-aligned,not a full-body-frame port error. A synthetic3.5m/s path/body yaw of8deg gives0.487m/s world lateral velocity while retaining4.853/5 of the tracking-plus-heading rate. Other terms are fixed; this is not a complete-return or dynamics proof. Exact139 representative50Hz states show strong tracking/heading credit despite excessive drift. Only one representative3.5 robot per test enters this objective reconstruction; population gate statistics are separate. No reward or command change is deployed. Next measure truthful normal-limit heading-channel response,actual target/heading timing and complete-objective tradeoffs before selecting a narrowly versioned64/128 continuation.

## Resume, failures and next work

The first pause bookkeeping attempt hit a str/Path hashing error after draining an evaluation; exception cleanup resumed the same supervisor. Corrected bookkeeping repeated the saved-boundary pause. Failed state and diagnostic summary are preserved; no policy update was lost. The separate stopped-hook CPU subtraction remains unresolved diagnostic accounting uncertainty,not a demonstrated production defect.

The same39000 supervisor resumed from exact139 after every diagnostic worker exited. Fresh actual main learning/full-state evidence is in [final_learning_verification.json](final_learning_verification.json); [live ownership](final_cpu/live_handoff.json),[matching gates](closing_gate_review.json),[decision](decision.json),[comparison](matched_context_review.json),[upper angles](final_upper/upper_joint_angles.png),[velocities](final_upper/upper_joint_velocities.png) and [objective audit](direction_objective_audit.json) retain exact provenance. No new actuator-ceiling evidence is claimed from this check. Videos are already fulfilled; unsaved Fusion work is untouched.

Next focus: controlled heading-response/full-objective diagnosis; mandatory per-speed upper timelines and effective targets; exact full-state/memory/sample accounting and independently seeded endpoint tests. Same ACTIVE five-hour heartbeat and notification preference remain. Stop only at qualified5m/s or qualified4m/s with causal native limits blocking5; then save,pause training+automation,brief and leave the computer on.

Primary references reviewed: [Ross et al., state-distribution-aware imitation learning](https://proceedings.mlr.press/v15/ross11a.html),[Mirelman et al., arm swing features](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043),[official Unitree workflow](https://github.com/unitreerobotics/unitree_rl_mjlab). No speed-matched Unitree encoder trajectory was obtained and no reference curves were invented.

<!-- adaptive_check_20261007_0844_completion -->
Closing confirmation 2026-10-07T09:36:28.015262+00:00: native app confirms same ACTIVEfive-hour heartbeat/thread/notification preference; durable scope/5-or-qualified4 endpoint prefix is byte-identical. No duplicate created. CPU history inventory has40 physical runs,2593 operations,zero differing hashes/read warnings;8 historical missing manifests remain documented. Eight angle/velocity images have visual QA. Exact resumed142 main64 updates and matching143/144 are verified; later145 partial learner remains separately unqualified. No production source/gain/reward/cap/state reset or diagnostic actor deployed. Reports and earlier history preserved. See E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_0844\automation_verification.json and E:\Aditya\robot_sim\physical_v29\training\diagnostics\adaptive_check_20261007_0844\inventory_verification.json.
<!-- /adaptive_check_20261007_0844_completion -->
