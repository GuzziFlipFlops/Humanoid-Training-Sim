"""Readable report for the independently verified closeout; no training mutations."""
from pathlib import Path


def render(out, v, pair, matched, delta, resume, upper, gate_table, trial_table, priorities, evidence):
    actor = Path(pair['checkpoint']).parent.name
    head = next(r['head_rms_deg'] for r in upper['records']
                if r['actor_tag'] == 'latest' and r['test'] == 'nominal' and r['command_m_s'] == 3.5)
    last = v['blocks'][-1]
    fast = [next(c for c in pair[t + '_cases'] if c['command'] == 3.5)
            for t in ['nominal', 'pushed']]
    speeds = '/'.join(f"{c['signed_speed']:.3f}" for c in fast)
    lateral = '/'.join(f"{c['lateral_speed']:.3f}" for c in fast)
    current_samples = evidence['retained_main_totals']['current_run_actual_samples']
    trend_table = '| Exact actor | 3.5 signed N/P m/s | Lateral N/P m/s | Current case N/P |\n|---|---|---|---|\n'
    for record in evidence['saved_checkpoint_trends']['records']:
        cases = [next(c for c in record['tests'][t]['cases'] if c['command'] == 3.5)
                 for t in ['nominal', 'pushed']]
        trend_table += (f"| {Path(record['checkpoint']).parent.name} | "
            f"{cases[0]['signed_speed']:.3f}/{cases[1]['signed_speed']:.3f} | "
            f"{cases[0]['lateral_speed']:.3f}/{cases[1]['lateral_speed']:.3f} | "
            f"{cases[0]['passed']}/{cases[1]['passed']} |\n")
    return f'''# Five-hour humanoid training check — October 9, 2026, 20:48 UTC

The tested final-output-row teacher continuation does not pass the complete independent contract and is rejected for deployment. Its shared-feature comparison also fails. The original full learner has resumed and completed genuine main training. The recurring failures concern direction and retention of useful lower speeds; this check does not establish an actuator ceiling.

## Current saved actor and qualification

Owned observation: **{v['owned_snapshot_utc']}**. R40 supervisor **39000**, one actual CUDA worker, **55 frozen hashes and 53 protected authored files verified**. Current-run progress is **{v['completed_main_updates']:,} completed + 50 hash-proven recovered = {v['actual_retained_main_updates']:,} retained main updates**, plus **27,507 inherited updates**. The {v['observed_partial_main_updates']} observed partial updates are separately unqualified.

The exact reviewed actor is **{actor}**, SHA **{pair['sha256']}**. At command 3.5 m/s, nominal/pushed signed forward speed is **{speeds} m/s** and lateral speed is **{lateral} m/s**. Current-case passes are **{[c['passed'] for c in fast]}**. The full command-by-command contract is:

{gate_table}
These saved gates and the mandatory upper-body review bind this exact actor. Newer saved or partial actors do not inherit them. Each ordinary evaluation uses 320 robots for 20 seconds and 32 robots per command. Ordinary nominal clones do not establish the independent initial-state diversity required at the final endpoint. No complete 3.5, 4 or 5 m/s stage is inferred.

Preserve the separately useful 1106/1058/1061/1064 current-speed pairs, actor 267's reverified all-prior certificate through 3 m/s, and separately certified actor 349. Actor 267's nominal/pushed and diverse-state requalification was performed during the **15:48 check** and its byte-bound certificate is reused here; it is not reported as a new requalification in this check.

{trend_table}
Current direction and lower-speed retention are nonmonotonic. Closing actor 1115 recovers 2.5 m/s but adds a nominal 0.65 m/s bilateral-stance failure and fails both retained 3 m/s direction cases. Its 0.15 m/s signed travel falls to 0.076/0.082 m/s; 1.5 m/s genuine flight is 0.101/0.059%, below the unchanged requirement. The earlier passing current-speed actors retain their own evidence. [Exact pair trends](saved_gate_trends.json).

## Controlled learning comparison

The exact current full-state anchor is **1112_run_350_train_368**, SHA **680daf86e2b608f5bf69bd3154c45f6e1498a7bef1637c9afdaf0c14d74682b8**. Unchanged nominal replay reproduces all 51 telemetry arrays and every case bit for bit.

The new candidate routes only the eligible, settled 0.15/1.5 m/s coupled-267 teacher derivative to the **21 non-head final output rows**. Its shared-feature route and an unchanged full-state control are evaluated on the same anchor. The original 0.25/23 loss, sample eligibility, other teacher/context derivatives, PPO, critic, exploration, 35/1 gains, upper refinement, rewards, physics, native limits and truthful student inputs remain. Teacher 267 supplies no labels above its reverified 3 m/s certificate. The diagnostic composite supplies no production inference.

Twenty-one CPU fixtures pass, including objective equivalence, reset/lifecycle behavior, selected head/trunk derivative exclusion and accumulation across reused backward graphs. Four actual PPO captures contain 32–48 eligible states. An isolated 1e-4 SGD probe preserves fixed-state head means exactly on the final-row route, but its fast-command output changes are nearly as large as those of the shared route. This restriction does not isolate global gradient clipping, Adam, other teacher updates, PPO or physical feedback.

The completed experiment contains **48 operations**, **768 updates and 75,497,472 diagnostic learning samples**, with learning seeds **251/263**, 4,096 environments, and matched **64/128-update** continuations. Ordinary nominal/pushed gates and extra 32 individually generated initial states per command are reviewed. Twelve small smoke/reload updates and four passive PPO updates are separately excluded from main progress. Full actor/critic/Adam/exploration loading, zero pre-update KL and command mismatch, consumed row derivatives, the three original 20-batch auxiliaries and memory guards are independently checked.

{trial_table}
The routes are rejected on the full results, not on one favorable checkpoint or a single flight statistic. [All matched cases](matched_row_route_review.json), [execution, recovery and diversity validation](experiment_validation.json), and [matched upper-body comparison](upper_matched_comparison.json) preserve every failure.

The completed CPU weight-response review covers all twelve learned checkpoints on at most 128 sampled truthful recorded states per command from one actual PPO minibatch. The recorded MLP input transform is held fixed only for this diagnostic. Full learning changes head and fast-command outputs even when the isolated selected derivative preserves head rows. This is a whole-learning weight response, not teacher-only causation, physical isolation, deterministic gate coverage or production observation substitution. [Recorded-state analysis](fixed_state_learning_response.json).

## Physical feasibility and native motors

The new current-anchor test has **640 robots for 20 seconds**, two seeds, and 32 individually generated paired states per controller/command at 0.15/1.5 m/s. All use truthful free-base feedback, gravity and native limits. All survive without resets; command and full reward-term reconstruction checks pass.

Qualified 267 body/arms with the current head gives **2.364/2.393%** genuine 1.5 m/s flight and higher complete return than the current actor. The whole-upper composite now gives about 1.6% flight but lower complete return; the earlier blanket below-threshold observation is actor-specific and does not apply here. Physical compatibility does not guarantee that a training label transfers the same gait. Every composite remains diagnostic only.

The **4,000 native 5 ms samples** bind these 0.15/1.5 m/s cohorts. CAD motor-speed and requested/applied torque reconstruction passes, with zero driving-speed clips. This does not certify the newer 3.5 m/s actor or the 4/4.5/5 m/s extrapolations. Sparse current-speed 10 Hz cap crossings cannot inherit substep attribution, and no sustained relevant motor ceiling is proved. [Physical response](neutral_teacher_response_analysis.json), [motor attribution](teacher_compatibility_response/substep_motor_audit.json).

## Upper-body motion over time

All **14 current all-command angle/target, velocity and bias plots**, plus four plots for each of the six matched 128-update actors, were visually inspected. The data explicitly separates 32 robots per command at 10 Hz from 16 selected representatives at 50 Hz, segments resets, and reports phase alongside amplitude, bilateral coordination, cadence, smoothness, saturation and noncancelling slow bias. The 0.5 Hz slow separation is descriptive.

Current native neck RMS at 3.5 m/s is **{head:.2f} degrees**. Held shoulder bias, unequal pitch/roll/elbow motion and gait-linked neck oscillation remain. Phase on a small-amplitude shoulder is not strong coordination evidence. Upper fine-tuning continues at **20 batches per genuine main update** with original gains 35/1. No tested arm derivative, gain or diagnostic actor has been deployed. Native head zero is torso-relative, and the original face/camera optical axis remains uncalibrated.

The primary [Collins 2009 study](https://pmc.ncbi.nlm.nih.gov/articles/PMC2817299/) documents walking at 1.25 m/s and shoulder/elbow time plots in Figure 6. Numerical curve extraction, anatomical axis/sign/rest alignment and mapping to this robot's morphology remain unverified; those plots are not a 3.5 m/s running reference. [Mirelman 2015](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043) supports feature-based amplitude, symmetry and coordination comparisons, rather than mapped robot encoder curves. The inspected [official Unitree repository](https://github.com/unitreerobotics/unitree_rl_mjlab) did not provide the mapped speed-matched joint trajectories needed for a numerical curve comparison. No reference curve was fabricated. Human/Unitree likeness and optical gaze remain uncertified. [Reference scope](reference_review.json), [current plots and population data](final_upper/audit_set.json).

## Genuine progress, timing and memory

R40 retained main samples total **{current_samples:,}**. Since the previous exact 18:34 owned endpoint, **{delta['retained_main_updates']:,} retained main updates and {delta['actual_main_samples']:,} actual main samples** were added, using {delta['active_collection_PPO_seconds']:.1f} active collection/PPO seconds and {delta['worker_seconds']:.1f} worker seconds. The 27,507 inherited updates are declared separately; their sample count is not inferred by assuming a fixed environment count.

Retained active / worker-effective / whole-chain rates are **{v['active_PPO_samples_s']/1000:.2f}k / {v['worker_effective_samples_s']/1000:.2f}k / {v['whole_chain_samples_s']/1000:.2f}k samples/s**. The distinct denominators and endpoint-scoped suspension exclusions are preserved. Rates since the previous endpoint are **{delta['active_samples_s']/1000:.2f}k / {delta['worker_effective_samples_s']/1000:.2f}k / {delta['whole_chain_samples_s']/1000:.2f}k**.

The same supervisor resumed after **{resume['pause_seconds']:.1f} seconds**, including worker drain. The genuine receiver loads exact anchor 1112 actor, critic, Adam and exploration, with zero pre-update KL/command mismatch and all three auxiliaries at 20 batches. Original nominal 1113's wrapper includes this suspension; only documented intervals before each measurement endpoint are excluded from throughput.

Latest verified completed main block **{last['folder']}** has **{last['updates']} updates at {last['environments']:,} environments**, minimum free GPU/host memory **{last['minimum_free_gpu_MiB']}/{last['minimum_host_available_MiB']} MiB**, private growth **{last['private_memory_growth_MiB']} MiB**, and Torch allocation growth **{last['torch_allocation_growth_MiB']} MiB**. Capacity, smoke and diagnostic samples are excluded. Preserve the 8,192 target, 300 MiB reserve, host guards/fallback and 180 continuous healthy training seconds before ramps. Evaluation or 4,096-environment diagnostic memory is not 8,192 training capacity.

The CPU lifecycle hook-count fixture error and copied teacher-reference lookup failure are preserved and corrected; neither proves a production defect. The live progress display initially included four setup receipts in its 48-operation comparison denominator; that CPU reporting counter was corrected without interrupting the diagnostic worker or changing learned-sample accounting. The independent final validation counts the 48 comparisons explicitly. Original sources, reward v8/verifier v6, truthful 400-input/23-output inference, local ground, native CAD caps and iteration-4 execution remain. Fusion stays closed; queued work and completed videos are preserved.

## Next bounded decisions

Read-only CPU preparation confirms that actor 267 has matching actor/critic tensor shapes, finite saved Adam and trained exploration, matching reward/asset/actuator/command metadata, identical teacher actor bytes and identical frozen head-context data. This makes a complete earlier learned-state branch reviewable; it does not prove a runtime load or a learning improvement. No earlier-state continuation has been started. [CPU preflight](older_branch_CPU_preflight.json).

'''+''.join(f'{i}. {p}\n\n' for i,p in enumerate(priorities,1))+'''The same five-hour automation stays active. Only fully qualified 5 m/s, or fully qualified 4 m/s with controlled native limits proving 5 m/s blocked, ends training. Every endpoint requires all unchanged gates and 32 independently seeded initial states per command. At that endpoint save exact evidence, safely pause training and automation, prevent restart and brief the user. Leave the computer on.
'''
