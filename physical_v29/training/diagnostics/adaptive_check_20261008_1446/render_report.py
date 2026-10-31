"""Human-readable report from already validated exact evidence."""
import json
from pathlib import Path

def render(c):
    OUT, pair, decision = c['OUT'], c['pair'], c['decision']
    v, total, delta, last = (c[k] for k in ['v', 'total', 'delta', 'last'])
    rows = []
    coverage = json.loads((OUT / 'sequence_coverage_analysis.json').read_text())['records']
    for r in c['comparison']['rows']:
        bias = next(x for x in coverage if x['branch'] == r['branch'] and x['updates'] == r['updates'])
        fast = next(x for x in bias['physical'] if x['command'] == 3.5 and x['test'] == 'nominal')
        n, p = r['nominal_cases'][-1], r['pushed_cases'][-1]
        rows.append(f"| {r['branch']} {r['updates']} | {n['signed_speed']:.3f} / {p['signed_speed']:.3f} | "
            f"{n['lateral_speed']:.3f} / {p['lateral_speed']:.3f} | "
            f"{n['flight']*100:.2f}% / {p['flight']*100:.2f}% | "
            f"{fast['slow_RMS_deg'][0]:.2f} / {fast['slow_RMS_deg'][1]:.2f} | "
            f"{r['nominal_all_pass']} / {r['pushed_all_pass']} |")
    comparison_table = '\n'.join([
        '| Branch / updates | Signed speed n/p (m/s) | Lateral speed n/p (m/s) | True flight n/p | Slow shoulder RMS L/R (deg, nominal) | Every case passes n/p |',
        '|---|---:|---:|---:|---:|---|', *rows])
    stats = {r['branch']: r['actual_gradient_statistics_mean'] for r in c['comparison']['rows'] if r['updates'] == 128}
    latest_cycles = {r['branch']: r for r in coverage if r['updates'] == 128}
    next_tests = '\n'.join(f'{i+1}. {s}' for i, s in enumerate(c['next_tests']))
    return f'''# Five-hour humanoid check — October 8, 14:46 UTC

{c['summary']}

## Decision and current saved actor

{decision['explanation']}

The original learner continues with its compatible full state. Preserve actor 605's separately passing 3.5 m/s case, actor 267's all-prior pair and original 349's separate 3 m/s certification. The whole current stage and final 5/conditional 4 m/s endpoint remain unqualified. Matching tests use 320 robots for 20 seconds, with 32 trials per command. Ordinary nominal clones do not establish the independently seeded initial-state diversity needed at the final endpoint.

{chr(10).join(c['table'])}

Read initial actor 704, older actor 713, all-command audited actor 716 and latest exact actor {Path(pair['checkpoint']).parent.name} as separate evidence. Retention is nonmonotonic: 704 recovered 0.15 m/s but failed 0.35 and 3 m/s; 713 recovered 0.35 and 3 m/s but lost 0.15; resumed 719 recovers 0.15 in both tests but still fails 1.5 m/s geometric flight and 3/3.5 m/s lateral direction. A saved qualification belongs to its checkpoint/hash and cannot be attributed to newer partial learning.

## Completed arm continuation and unchanged control

{comparison_table}

`matched_sequence_review.json` contains exact full-state 4,096-environment control and candidate branches at 64 and 128 updates, all unchanged nominal/pushed gates, and additional 128-update tests with 32 independently generated small joint initial states per command. Seeds 2309 and 1911 generate identical initial realizations for both branches. All diagnostic actors remain separate from the production supervisor chain.

The candidate penalizes persistent shoulder pitch means using completed eight-bin cycles within each robot and episode. It records truthful 400-input histories, keeps at most 128 robots and 256 cycles, retains two recent cycles per robot and rejects cycles older than 200 ticks. Resets, invalid histories and command switches clear samples. Current hidden features are recomputed and detached; only two final shoulder output rows receive direct new derivatives. Ordinary PPO, global gradient clipping and physical coupling can still affect legs, so every locomotion gate remains necessary.

The weight is 0.025, the mean band is 5 degrees and the derivative cap is 10% of the incoming shoulder PPO norm. These values match the rejected ensemble candidate, but temporal and command coverage intentionally differ. This experiment changes more than an algebraic average. It imposes no swing amplitude, phase, reference trajectory or inference override. The experimental 5-degree band does not certify natural motion. Original frozen signatures do not certify the new gradient; no experimental checkpoint enters production under an old signature.

`sequence_coverage_analysis.json` compares actual latest completed training-cycle snapshots with independent physical shoulder motion. The physical slow RMS does not cancel opposite biases or slow reversals. Training states, evaluation states and exploration differ; a difference between their means is not automatically a numerical or observation fault. `upper_matched_comparison.json` binds the motion, gradient statistics, coverage, gates and plot inspection to each exact checkpoint.

At the 128-update block, the mean absolute cached training-cycle output is {stats['candidate']['cycle_mean_abs_deg']:.2f} degrees for the candidate versus {stats['control']['cycle_mean_abs_deg']:.2f} for control. Its measured bounded derivative norm averages {stats['candidate']['bounded_gradient_norm']:.3f}, versus incoming shoulder PPO norm {stats['candidate']['actual_PPO_shoulder_norm']:.3f}. The computational objective responds, but independent physical bias and locomotion do not improve enough. The latest candidate cache contains {latest_cycles['candidate']['cycles']} cycles from {latest_cycles['candidate']['robots']} robots; that is a subset of 4,096 training robots and only the latest update, not a claim of complete state/command coverage. Independent nominal and pushed initial-state tests reproduce the candidate's direction failure. There is one paired learning seed, 149; the additional seeds diversify physical initial states rather than independently replicating learning.

## Validation and preserved failures

Twenty-six CPU checks cover aggregation, reset/command handling, output-row scope and the installed RSL-RL MLP. The successful unchanged-actor replay matches all cases and {c['validation']['physical_arrays']} physical arrays bit for bit. Actual 64-environment, eight-update PPO/full-state load/reload passes. The 1,024-environment, 16-update memory smoke also passes. Every completed 4,096-environment block has exact actor/critic/Adam/exploration loading, zero pre-update KL and command mismatch, and all original three auxiliaries at 20 batches per update. Complete-cycle collection has a reported warm-up.

Attempt 01 failed in diagnostic code because the installed MLP subclass cannot be sliced like Sequential. Its smoke actor was discarded, and supervisor 39000 resumed after {c['failures']['attempts'][0]['resume']['pause_seconds']:.1f} seconds. Iterating the existing hidden layers explicitly fixes the problem; installed-MLP backward checks pass.

Attempt 02 retained GPU graphs because a hook callback captured its own mean tensor, even when the new derivative was disabled. The native 300 MiB guard stopped that diagnostic control after 12 updates. Its logs and recovery checkpoint are preserved and excluded from production/main learning. Supervisor 39000 resumed after {c['failures']['attempts'][1]['resume']['pause_seconds']:.1f} seconds. With cyclic garbage collection disabled, a CPU weak-reference test reproduces 12/12 retained outputs before repair and 0/12 afterward. Moving the constant cycle-count tensor outside the callback preserves the numerical objective and gradients. The corrected memory smoke has 1 MiB allocation variation; the final comparison also checks steady allocation and early graph growth. These are experimental-tool failures, not reproduced faults in production PPO or physics. The earlier failed phase-boundary fixture is preserved separately; midpoint samples repaired the fixture without changing runtime phase handling.

## Required upper motion analysis

Actual head yaw/pitch, shoulder pitch/roll and elbow angles, physical targets and velocities were saved and visually inspected at all commanded speeds, alongside hip motion, bilateral amplitude/phase, cadence, elbow coordination, smoothness and saturation. `final_upper` is explicitly actor 716 and includes fourteen inspected nominal/pushed plots for every stopped, slow, intermediate and fast command. `closing_upper` independently audits newer resumed actor 719: all-speed population statistics and four inspected representative angle/target, velocity and bias plots. Actor 716's full traces do not certify 719's posture. Population statistics use 32 robots per command at 10 Hz; high-rate traces use 16 selected representatives per gate at 50 Hz. Resets and terminal rows are segmented. The 10 Hz population sample cannot certify faster jitter.

PNG plots have SVG, CSV and JSON companions. The descriptive 0.5 Hz slow-bias decomposition excludes three-second filter edges and separates episode segments. RMS and absolute means prevent opposite or reversing biases from cancelling. A large total shoulder range can reflect slow drift rather than gait swing. The branch and final inspection receipts describe the measured patterns without declaring natural motion from bounds or reward.

The latest 719 plots still show opposite held shoulder offsets, unequal swing and elbow/roll coordination, slow target-driven reversals at 2.5/3 m/s and gait-linked head oscillation. A separate stopped-trace interpretation remains open: almost static positions coexist with small reported head/hip velocity residuals. Solver position corrections and sampling have not been attributed at 5 ms; this does not establish a clock defect or a motor ceiling.

The [primary human IMU study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043) supports feature comparisons of amplitude, asymmetry and smoothness; its wrist IMUs do not supply morphology-matched shoulder encoder targets. The [official Unitree workflow](https://github.com/unitreerobotics/unitree_rl_mjlab) provides motion/policy workflows, but verified, mapped, speed-matched G1/H1 encoder trajectories were not obtained. No reference curves are fabricated and no exact human/Unitree likeness is certified. Native head zero is torso-relative. Known Isaac/FK head-orientation parity does not calibrate the original CAD face/camera optical axis or world optical gaze.

## Synchronized physics and native limits

`final_com_contact_audit.json` binds finite synchronized COM, joints, targets, actions, contacts, sole forces and commands to exact actor 716. Full-population support/contact/direction classifiers stay unchanged. COM offsets and simplified collision envelopes do not certify full-body pressure distributions.

`final_motor_speed_audit.json` and `final_pushed_motor_speed_audit.json` are exact actor 716, full-population 10 Hz nominal/pushed evidence; older `motor_speed_audit.json` remains actor 704 evidence. Exact CAD differential Jacobians and reductions use actual interleaved case labels and pass directional/unit checks. At 3.5 m/s, left ankle motor 2 is at/above the declared cap in 2.031%/1.997% of sampled nominal/pushed instants; its p99 ratio is 1.639/1.652. Right ankle motor 2 is at/above cap in 1.111%/0.833%. These warrant current-actor 5 ms driving-request/application and duration attribution; they do not establish sustained driving clipping or available-torque limits. Sampled generalized joint effort is not motor-side available torque. Older different-actor substep tests cannot certify this actor. No relevant actuator ceiling or 4 m/s fallback is inferred; caps remain unchanged.

## Actual samples, time and memory

At {v['owned_snapshot_utc']}, {v['completed_main_updates']:,} completed plus {c['recovered']} verified recovered updates retain {v['actual_retained_main_updates']:,} current-run main updates, plus 27,507 inherited. The 32 interrupted unsaved R39 updates remain excluded. Diagnostic, capacity and smoke samples never count as main learning.

Retained main totals are {total['samples']:,} samples in {total['active_seconds']:,.1f} collection-plus-PPO seconds and {total['worker_seconds']:,.1f} worker seconds. Since the previous check, this adds {delta['updates']:,} updates and {delta['samples']:,} samples in {delta['active_seconds']:,.1f} active, {delta['worker_seconds']:,.1f} worker and {delta['chain_seconds_excluding_documented_pauses']:,.1f} whole-chain seconds excluding documented pauses. Interval rates are {delta['active_samples_s']:,.0f} active, {delta['worker_effective_samples_s']:,.0f} worker-effective and {delta['chain_samples_s']:,.0f} chain samples/s.

Cumulative rates are {v['active_PPO_samples_s']:,.0f} active, {v['worker_effective_samples_s']:,.0f} worker-effective and {v['whole_chain_samples_s']:,.0f} chain samples/s. Whole-chain time excludes {v['documented_diagnostic_pause_seconds_excluded']:,.1f} seconds of documented suspensions through its endpoint, and includes startup, capacity checks and gates. These denominators differ. Latest completed main: {last['environments']} environments, minimum free GPU {last['minimum_free_gpu_MiB']} MiB, host available {last['minimum_host_available_MiB']} MiB, private-memory growth {last['private_memory_growth_MiB']} MiB and Torch allocation growth {last['torch_allocation_growth_MiB']} MiB.

Keep the 8,192 target, 300 MiB reserve, host guards/fallback and 180 continuous healthy training seconds before an upward ramp. Evaluation/startup memory does not establish capacity. Fusion stays closed; queued work and completed videos are preserved. Leave the computer on.

## Next focused tests

{next_tests}

Keep the same active five-hour heartbeat, thread and notification preference. Finish only after signed forward speed of at least 4.95 m/s with every unchanged matching nominal/pushed/stopped/prior gate and 32 independently seeded states per command, or qualified 4 m/s at least 3.96 with every gate plus controlled native limits proving 5 m/s blocked. At that verified endpoint save exact evidence, pause owned training and this automation, prevent restart, brief the user and leave the computer on.
'''
