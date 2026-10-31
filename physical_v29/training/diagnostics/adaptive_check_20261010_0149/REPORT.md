# Humanoid training check — October 10, 2026, 01:49 UTC

Reject the tested earlier complete-state continuation for deployment. Both learning seeds retain useful 1.5 m/s flight but fail the full direction and prior-speed contract. The unchanged current-state controls also remain unqualified. Preserve both complete learned states and resume the original full learner; prioritize a measured, versioned objective/update that protects each command's retained behavior.

## Exact saved actor and independent behavior

Same R40 full learner and supervisor 39000 at 2026-10-10T03:42:41.288235+00:00, with one actual worker, 55 frozen hashes and 53 protected files. 24,615 completed + 50 recovered = 24,665 retained current-run updates, plus 27,507 inherited; 0 partial updates are separate. Exact actor 1157_run_350_train_383, SHA f0c727f41c551e08452e6956c2b330f37523956e33dff101152052440baf5224: command 3.5 gives signed speed 3.420/3.434 and lateral speed 0.451/0.422 m/s. Current nominal/pushed passes: [False, False]. Failures: {'nominal': [{'command': 0.15, 'failures': ['forward_net_travel']}, {'command': 1.5, 'failures': ['geometric_flight']}, {'command': 3.0, 'failures': ['world_lateral_speed']}, {'command': 3.5, 'failures': ['world_lateral_speed']}], 'pushed': [{'command': 0.15, 'failures': ['forward_net_travel']}, {'command': 1.5, 'failures': ['geometric_flight']}, {'command': 3.0, 'failures': ['world_lateral_speed']}, {'command': 3.5, 'failures': ['world_lateral_speed']}]}. Saved gates never qualify newer actors. No 5/4 m/s or causal ceiling endpoint is established.

|Command m/s|Signed N/P m/s|Lateral N/P m/s|True flight N/P %|Failures N/P|
|---|---|---|---|---|
|0|-0.000/-0.000|0.001/0.019|0.000/0.000|pass / pass|
|0.15|0.112/0.115|0.090/0.093|0.000/0.000|forward_net_travel / forward_net_travel|
|0.35|0.402/0.401|0.099/0.104|0.000/0.000|pass / pass|
|0.65|0.719/0.721|0.089/0.098|0.000/0.000|pass / pass|
|1|1.034/1.031|0.086/0.095|0.000/0.000|pass / pass|
|1.5|1.473/1.472|0.086/0.096|0.000/0.014|geometric_flight / geometric_flight|
|2|2.046/2.041|0.066/0.083|5.149/5.236|pass / pass|
|2.5|2.525/2.525|0.070/0.083|14.840/14.837|pass / pass|
|3|3.002/2.998|0.245/0.250|24.406/24.861|world_lateral_speed / world_lateral_speed|
|3.5|3.420/3.434|0.451/0.422|38.611/38.892|world_lateral_speed / world_lateral_speed|

Each ordinary evaluation uses 320 robots for 20 seconds, with 32 per command. Its nominal clones are distinct from the 32 individually seeded joint-position/velocity initial states per command used in the comparison's additional tests. Startup landing events and all existing survival, contact, direction, flight and retention classifiers remain unchanged.

The 3.5 m/s stage requires its current case, stopped behavior and every earlier command to pass in both matching tests. A separately passing fast case does not authorize promotion to 4, 4.5 or 5 m/s. Earlier useful actors and their exact certificates are preserved. Saved gates do not qualify a newer partial actor.

## Earlier complete learned-state continuation

The current anchor is **E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261007_r40_goal_budget_full_state\1157_run_350_train_383\candidate.pt**, SHA **f0c727f41c551e08452e6956c2b330f37523956e33dff101152052440baf5224**. The earlier neutral-head actor 267 has SHA **20a133fdd4843118d816172239201a387ca947b38952fb38f094a156afa0a320**.

Fresh CPU teacher, context and checkpoint checks pass. The current nominal replay matches every evaluation case and all **39** saved arrays bit for bit: 13 basic telemetry arrays plus the synchronized representative measurements. Fresh ordinary and individually seeded nominal/pushed tests reverify every command through 3 m/s for actor 267. Its baseline 3.5 m/s direction fails. [Replay parity](full_baseline_parity.json), [starting-state comparison](starting_motion_comparison.json).

Both branches resume their complete actor, critic, Adam state and trained exploration. They use identical original objectives, teacher actors and frozen head context, PPO, 35/1 gains, truthful 400-input/23-output histories, native motor limits and local ground. The initial complete checkpoint is the branch difference. This does not isolate an actor feature, critic, optimizer or individual reward term. No teacher controller, target override, observation substitution or new gradient hook enters inference.

|Seed|Branch|Updates|Ordinary N/P pass|Diverse N/P pass|
|---|---|---|---|---|
|317|control|64|False/False|tested at128|
|317|control|128|False/False|False/False|
|317|earlier|64|False/False|tested at128|
|317|earlier|128|False/False|False/False|
|331|control|64|False/False|tested at128|
|331|control|128|False/False|False/False|
|331|earlier|64|False/False|tested at128|
|331|earlier|128|False/False|False/False|

Two learning seeds, 317 and 331, use 4,096 environments and 64/128 updates per branch. Every saved checkpoint has ordinary nominal/pushed tests. At 128 updates, it also has nominal/pushed evaluations with 32 individually seeded initial states per command. Independent validation confirms all 32 comparison operations, 512 genuine diagnostic updates and 50,331,648 samples. Sixteen separate 64-environment smoke/reload updates add 24,576 samples. All diagnostic samples are excluded from main progress.

At 128 updates, earlier-state ordinary nominal/pushed 1.5 m/s geometric flight is 2.226/2.132% in seed 317 and 1.875/1.861% in seed 331; corresponding controls have 0.000/0.035% and 0.000/0.049%. The earlier branches nevertheless have 3.5 m/s lateral speeds 0.623/0.599 and 0.451/0.421 m/s, versus control 0.148/0.137 and 0.373/0.349. Both earlier seeds fail 2.5 m/s slip and retained 3 m/s direction at 128; seed 331 adds retained 1 m/s direction failure, and slow path efficiency is not reliably recovered. Individually seeded initial states reproduce these failures. All robots survive, so survival alone is insufficient. Fresh actor 267 passed all priors through 3 m/s before this continuation, making its subsequent retention loss reproduced evidence. The complete checkpoint, including critic/Adam/exploration, is the branch difference; this does not isolate a reward term, actor feature or mechanical arm cause. Running neck RMS increases from matched controls 2.40 to 3.92 degrees and 4.55 to 5.65 degrees in the earlier branches; smaller shoulder offsets alone do not certify a better upper controller. This bounded failure does not prove every earlier-state approach ineffective, 5 m/s impossible or a native motor ceiling. No new reward, CPO optimizer, teacher target, diagnostic actor, gain or cap is deployed.

[Exact comparison and failures](matched_full_state_review.json), [independent load, provenance and diversity validation](experiment_validation.json).

The new [force, sole-clearance and COM comparison](flight_retention_analysis.json) uses actual physical measurements and the evaluator's 2.02-second warmup boundary. Representative 50 Hz traces are distinguished from full-population gates. [Mixed training metrics](training_metric_comparison.json) save logged reward components and bounded PPO returns. These are not per-command complete-return certification. The moving exploration value 0.7 is an upper bound, not a forced floor.

[Retention-constraint research](constraint_reward_review.md) records an undeployed follow-up and its primary-source limitations. No CPO or new reward is claimed as tested.

## Upper-body motion over time

Actual head yaw/pitch, shoulder pitch/roll and elbow angles, physical targets and velocities were visually inspected for the initial current actor, freshly tested actor 267, all four 128-update comparison actors and the exact closing actor **1157_run_350_train_383**. Closing QA covers all commands in both matching tests with 14 plots. The comparison actors have focal 0.15/1.5/2.5/3.5 m/s plots and all-command population statistics.

Population measurements cover 32 robots per command at 10 Hz. High-rate traces cover 16 selected representatives per evaluation at 50 Hz. Reset/terminal rows are segmented. The 0.5 Hz slow-bias decomposition excludes three-second edges; its residual includes non-cycle motion and does not certify faster jitter.

At closing command 3.5, native neck RMS is **3.22 degrees**. Shoulder pitch p95–p05 ranges are **[10.101331615448, 5.12462818622589] degrees** and slow pitch means are **[-26.864336452380982, 11.09171507307815] degrees**. Held offsets, unequal pitch/roll/elbow motion and gait-linked neck oscillation remain. Phase is interpreted alongside amplitude. Existing upper refinement continues at 20 batches per main update. [Closing timelines and inspection receipt](final_upper/audit_set.json), [matched motion comparison](upper_matched_comparison.json).

The primary [Collins 2009 study](https://pmc.ncbi.nlm.nih.gov/articles/PMC2817299/) measures walking at 1.25 m/s. Its shoulder/elbow plots are not a mapped 3.5 m/s running reference; numerical axes, signs, rest poses and morphology remain unverified. The inspected [official Unitree material](https://github.com/unitreerobotics/unitree_rl_mjlab) did not supply usable mapped speed-matched encoder curves. No reference curve is fabricated. Human/Unitree likeness and calibrated optical gaze remain uncertified; native-zero neck angles are torso-relative. [Reference scope](reference_review.json).

## Actual learning, resources and limits

R40 retains **4,016,504,832 main samples**. Since the previous exact owned endpoint, it adds **960 retained updates / 188,743,680 samples**, using 5046.4 active collection/PPO seconds and 6179.7 worker seconds. The 27,507 inherited updates are declared separately; their sample count is not inferred from a constant environment count. Fifty hash-received R40 recovery updates count. Thirty-two interrupted unsaved R39 updates do not.

Retained active / worker-effective / whole-chain rates are **35.99k / 29.78k / 19.16k samples/s**. The denominators differ. Documented suspensions are excluded only through each measurement endpoint; capacity, smoke, evaluation and diagnostic learning are excluded from main samples.

Supervisor 39000 resumes after **5693.6 seconds**, including worker drain. A genuine 64-update main receiver loads the exact anchor actor, critic, Adam and exploration with zero pre-update KL, zero action-command mismatch and all three original auxiliaries at 20 batches per update. [Recovery and timing proof](final_learning_verification.json).

Last verified main block **E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261007_r40_goal_budget_full_state\1160_run_350_train_384** completes **64 updates at 8,192 environments**. Its minimum free GPU/host memory is **1417 / 5819 MiB**, with private growth 10 MiB and Torch allocation growth 0 MiB. Keep the 8,192 target, 300 MiB reserve, host guards/fallback and 180 continuous healthy training seconds before upward ramps. Evaluation or diagnostic memory does not establish main capacity.

Closing synchronized COM/contact/target/action data are finite with zero action-command mismatch. The 10 Hz population motor audit uses actual case labels and CAD differential Jacobians/reductions. Sparse cap crossings do not establish sustained 5 ms driving clipping or available-torque limitation. This comparison adds no native 5 ms request/application probe; earlier results remain actor/cohort-specific. Previous 4/4.5/5 m/s extrapolation failures do not establish a ceiling. [Synchronized measurements](closing_COM_contact_audit.json), [nominal motors](closing_motor_nominal.json), [pushed motors](closing_motor_pushed.json).

## Next focused tests

1. Prioritize a versioned retention-aware objective/update, rather than repeating the unchanged earlier-state continuation. First read fresh exact current/prior gates and preserve the current complete learner, fresh all-prior actor 267 and separately passing fast actors. Capture complete per-command returns, advantages, critic/bootstrap accounting, teacher gradients and actual path/sole-force/clearance events on truthful settled and transitioning histories. Derive prospective path-efficiency and geometric-flight costs from unchanged classifiers, keeping commands separate so improvements elsewhere cannot cancel a violation. Stage one narrow training-only candidate only after finite/reset/transition/clipping/ratio and stationary, diagonal/sliding, hopping and early-exit counterexamples plus normal-limit physical/full-load checks. Then run exact full-state 64/128 controls, two learning seeds and every ordinary/diverse nominal, pushed, stopped, prior, head, direction and flight gate. CPO is an undeployed research lead, not a tested optimizer or locomotion guarantee. No arbitrary weight sweep, gate relaxation, diagnostic actor deployment or from-scratch reset.

2. Inspect actual head yaw/pitch, shoulder pitch/roll and elbow angle, target and velocity timelines at every command, with bilateral amplitude, arm-leg phase and noncancelling slow bias. Compare the preserved earlier branch's flight advantage with its direction/slip losses and current measured motion; smaller held bias alone does not qualify a repair. Measure phase-conditioned effective upper/lower targets and auxiliary/PPO gradients before another upper derivative. Preserve forward head, useful legs, both teachers, original 35/1 gains and existing upper 20 batches per update. Keep rejected row, coupled-target, shoulder-mean, stiffness, exposure, teacher/context, course/filter and reward candidates rejected absent new evidence. No production hybrid, observation substitution, imposed reference trajectory or cap raise. Human/Unitree encoder likeness and calibrated optical gaze remain uncertified.

3. Verify the same owned supervisor, one actual CUDA worker, all 55 frozen hashes/53 protected authored files, local ground and exact actor/critic/Adam/exploration plus both teachers. Count 27,507 inherited updates, actual completed main updates and 50 hash-proven R40 recoveries; exclude unsaved R39, capacity, smoke and diagnostic samples. Exclude this documented pause only through each timing endpoint. Keep target 8,192, reserve 300 MiB, host guards/fallback and 180 continuous healthy training seconds; Fusion stays closed. This check adds no native 5 ms motor request/application probe. Sparse newer 10 Hz cap crossings cannot inherit older actor-specific driving/braking attribution. A high-speed plateau needs actual CAD motor speeds, requested/applied torque, available torque and sustained driving saturation before a limit claim. Only fully qualified 5 m/s, or qualified 4 m/s with causal native limits blocking 5, all unchanged gates and 32 individually seeded states per command, ends training and this same five-hour heartbeat.

The same five-hour heartbeat stays active. Only fully qualified 5 m/s, or qualified 4 m/s with causal native limits proving 5 is blocked, ends training. Each endpoint requires every unchanged gate and 32 independently seeded initial states per command. At that endpoint save, pause training and automation, prevent restart and brief with the computer on. Fusion stays closed; queued work and completed videos are preserved.
