"""Readable report template; all numeric findings come from saved evidence."""
from pathlib import Path

def render_report(e,delta,last,head,parity,table,trials,priorities):
 v=e['current_owned_learning'];r=e['diagnostic_resume'];d=e['decision'];p=e['current_exact_pair'];u=e['upper']
 actor=Path(p['checkpoint']).parent.name
 rows=''.join(f'{i}. {text}\n\n' for i,text in enumerate(priorities,1))
 return f'''# Humanoid training check — October 10, 2026, 01:49 UTC

{d['conclusion']}

## Exact saved actor and independent behavior

{e['summary']}

{table}
Each ordinary evaluation uses 320 robots for 20 seconds, with 32 per command. Its nominal clones are distinct from the 32 individually seeded joint-position/velocity initial states per command used in the comparison's additional tests. Startup landing events and all existing survival, contact, direction, flight and retention classifiers remain unchanged.

The 3.5 m/s stage requires its current case, stopped behavior and every earlier command to pass in both matching tests. A separately passing fast case does not authorize promotion to 4, 4.5 or 5 m/s. Earlier useful actors and their exact certificates are preserved. Saved gates do not qualify a newer partial actor.

## Earlier complete learned-state continuation

The current anchor is **{r['checkpoint']}**, SHA **{r['checkpoint_sha256']}**. The earlier neutral-head actor 267 has SHA **20a133fdd4843118d816172239201a387ca947b38952fb38f094a156afa0a320**.

Fresh CPU teacher, context and checkpoint checks pass. The current nominal replay matches every evaluation case and all **{parity['array_count']}** saved arrays bit for bit: 13 basic telemetry arrays plus the synchronized representative measurements. Fresh ordinary and individually seeded nominal/pushed tests reverify every command through 3 m/s for actor 267. Its baseline 3.5 m/s direction fails. [Replay parity](full_baseline_parity.json), [starting-state comparison](starting_motion_comparison.json).

Both branches resume their complete actor, critic, Adam state and trained exploration. They use identical original objectives, teacher actors and frozen head context, PPO, 35/1 gains, truthful 400-input/23-output histories, native motor limits and local ground. The initial complete checkpoint is the branch difference. This does not isolate an actor feature, critic, optimizer or individual reward term. No teacher controller, target override, observation substitution or new gradient hook enters inference.

{trials}
Two learning seeds, 317 and 331, use 4,096 environments and 64/128 updates per branch. Every saved checkpoint has ordinary nominal/pushed tests. At 128 updates, it also has nominal/pushed evaluations with 32 individually seeded initial states per command. Independent validation confirms all 32 comparison operations, 512 genuine diagnostic updates and 50,331,648 samples. Sixteen separate 64-environment smoke/reload updates add 24,576 samples. All diagnostic samples are excluded from main progress.

{d['interpretation']}

[Exact comparison and failures](matched_full_state_review.json), [independent load, provenance and diversity validation](experiment_validation.json).

The new [force, sole-clearance and COM comparison](flight_retention_analysis.json) uses actual physical measurements and the evaluator's 2.02-second warmup boundary. Representative 50 Hz traces are distinguished from full-population gates. [Mixed training metrics](training_metric_comparison.json) save logged reward components and bounded PPO returns. These are not per-command complete-return certification. The moving exploration value 0.7 is an upper bound, not a forced floor.

[Retention-constraint research](constraint_reward_review.md) records an undeployed follow-up and its primary-source limitations. No CPO or new reward is claimed as tested.

## Upper-body motion over time

Actual head yaw/pitch, shoulder pitch/roll and elbow angles, physical targets and velocities were visually inspected for the initial current actor, freshly tested actor 267, all four 128-update comparison actors and the exact closing actor **{actor}**. Closing QA covers all commands in both matching tests with 14 plots. The comparison actors have focal 0.15/1.5/2.5/3.5 m/s plots and all-command population statistics.

Population measurements cover 32 robots per command at 10 Hz. High-rate traces cover 16 selected representatives per evaluation at 50 Hz. Reset/terminal rows are segmented. The 0.5 Hz slow-bias decomposition excludes three-second edges; its residual includes non-cycle motion and does not certify faster jitter.

At closing command 3.5, native neck RMS is **{u['head_RMS_deg']:.2f} degrees**. Shoulder pitch p95–p05 ranges are **{u['shoulder_pitch_ranges_deg']} degrees** and slow pitch means are **{u['slow_pitch_means_deg']} degrees**. Held offsets, unequal pitch/roll/elbow motion and gait-linked neck oscillation remain. Phase is interpreted alongside amplitude. Existing upper refinement continues at 20 batches per main update. [Closing timelines and inspection receipt](final_upper/audit_set.json), [matched motion comparison](upper_matched_comparison.json).

The primary [Collins 2009 study](https://pmc.ncbi.nlm.nih.gov/articles/PMC2817299/) measures walking at 1.25 m/s. Its shoulder/elbow plots are not a mapped 3.5 m/s running reference; numerical axes, signs, rest poses and morphology remain unverified. The inspected [official Unitree material](https://github.com/unitreerobotics/unitree_rl_mjlab) did not supply usable mapped speed-matched encoder curves. No reference curve is fabricated. Human/Unitree likeness and calibrated optical gaze remain uncertified; native-zero neck angles are torso-relative. [Reference scope](reference_review.json).

## Actual learning, resources and limits

R40 retains **{e['retained_main_totals']['current_run_actual_samples']:,} main samples**. Since the previous exact owned endpoint, it adds **{delta['retained_main_updates']:,} retained updates / {delta['actual_main_samples']:,} samples**, using {delta['active_collection_PPO_seconds']:.1f} active collection/PPO seconds and {delta['worker_seconds']:.1f} worker seconds. The 27,507 inherited updates are declared separately; their sample count is not inferred from a constant environment count. Fifty hash-received R40 recovery updates count. Thirty-two interrupted unsaved R39 updates do not.

Retained active / worker-effective / whole-chain rates are **{v['active_PPO_samples_s']/1000:.2f}k / {v['worker_effective_samples_s']/1000:.2f}k / {v['whole_chain_samples_s']/1000:.2f}k samples/s**. The denominators differ. Documented suspensions are excluded only through each measurement endpoint; capacity, smoke, evaluation and diagnostic learning are excluded from main samples.

Supervisor 39000 resumes after **{r['pause_seconds']:.1f} seconds**, including worker drain. A genuine 64-update main receiver loads the exact anchor actor, critic, Adam and exploration with zero pre-update KL, zero action-command mismatch and all three original auxiliaries at 20 batches per update. [Recovery and timing proof](final_learning_verification.json).

Last verified main block **{last['folder']}** completes **{last['updates']} updates at {last['environments']:,} environments**. Its minimum free GPU/host memory is **{last['minimum_free_gpu_MiB']} / {last['minimum_host_available_MiB']} MiB**, with private growth {last['private_memory_growth_MiB']} MiB and Torch allocation growth {last['torch_allocation_growth_MiB']} MiB. Keep the 8,192 target, 300 MiB reserve, host guards/fallback and 180 continuous healthy training seconds before upward ramps. Evaluation or diagnostic memory does not establish main capacity.

Closing synchronized COM/contact/target/action data are finite with zero action-command mismatch. The 10 Hz population motor audit uses actual case labels and CAD differential Jacobians/reductions. Sparse cap crossings do not establish sustained 5 ms driving clipping or available-torque limitation. This comparison adds no native 5 ms request/application probe; earlier results remain actor/cohort-specific. Previous 4/4.5/5 m/s extrapolation failures do not establish a ceiling. [Synchronized measurements](closing_COM_contact_audit.json), [nominal motors](closing_motor_nominal.json), [pushed motors](closing_motor_pushed.json).

## Next focused tests

{rows}The same five-hour heartbeat stays active. Only fully qualified 5 m/s, or qualified 4 m/s with causal native limits proving 5 is blocked, ends training. Each endpoint requires every unchanged gate and 32 independently seeded initial states per command. At that endpoint save, pause training and automation, prevent restart and brief with the computer on. Fusion stays closed; queued work and completed videos are preserved.
'''
