# Three-hour check: bilateral flight unloading, October 4, 2026, 20:39 UTC

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
- Final actor reconstructed 2,240 saved actions within 3.82e-6. Commands match; 37.66% of sampled raw running outputs exceed the existing +/-3 action clamp. Exact pose-dependent CAD motor speeds show no sustained locomotion-relevant ceiling; sampled ankle spikes and elbow motion are not a velocity-limit endpoint. Substep clipping remains unexcluded. effective_leg_target_audit.json separates each leg joint's raw clipping from its measured target/pose response during the physical intervention.

## Tests and deployment

Eleven focused CPU checks reproduce/fix the redistribution incentive and verify monotonic force derivatives, boundedness, symmetry and zero no-swing credit. **600 standing/walking full reward fixtures are bitwise identical.** Eight full-cycle suites through 5 m/s rank ideal alternating running above fast walking, sliding, in-place movement, backwards movement and synchronous hopping. Six negative transfer checks reject changed contract, reward signature/version, asset, motion and actuator metadata.

Real 64-environment running PPO actor-transfer and subsequent full actor/critic/Adam/exploration reload pass. All eight smoke pre-update KL values and action-command differences are zero. Smoke actors are discarded. A same-actor **192-robot, 20-second** physical nominal replay is bitwise identical in every saved array and every case result. This demonstrates unchanged execution/physics and inference; it does not qualify the new learning actor.

Exactly two authored runtime files changed: stage_rewards.py and reward_warm_start.py. The latter strictly permits the audited v7 signature and physical contract; it does not broaden acceptance to arbitrary actors. Forty-two protected authored files remain byte-identical. Selected mean and trained exploration are exact; critic/Adam are fresh because the objective changed. All nominal/pushed/stopped/prior-speed qualification must be repeated under R34.

R33 supervisor was suspended at a saved boundary, its eval worker completed and drained, and its full checkpoint was preserved. It was retired only after the diagnostics and candidate tests passed. R34 launched at 2026-10-04T21:03:40.923265+00:00 in `locomotion_20261004_r34_bilateral_swing_unloading`. Target 8192, 300 MiB GPU reserve, host guards, automatic fallback and a continuous 180-second healthy-memory upward ramp remain. The separate frozen worker and automatic curriculum continue through actual 5 m/s with unchanged limits and gates.

## Throughput and resource observations

R33 had two GPU-reserve recoveries and fell back to 4096. Across logged main attempts, active collection-plus-PPO throughput is 35.48k samples/s and worker-effective throughput 29.43k samples/s; these differ from whole-chain throughput, which includes initialization and gates. The last completed 4096 block is 26.38k active and 21.42k worker-effective, with 1437 MiB minimum free GPU, 9154 MiB minimum host available and 6 MiB private growth. Do not extrapolate earlier 8192 speed to these 4096 blocks. R34 capacity must be measured anew rather than reused across a changed objective.

## Diagnostic corrections and scope limits

Initial re-scoring used an incorrect reward-term label, then action-time phase instead of the post-step reward phase, and finally an unexpanded phase mask. Those rejected attempts are preserved in diagnostic_attempts.json; no conclusion or runtime repair used their failed results. The corrected calculation is verified against the actual saved reward term. Earlier failures and smoke evidence remain intact. The first new-run CPU status snapshot was rejected because ACTIVE_RUN initially used selected_checkpoint rather than the recorder's initial_actor_checkpoint provenance field. The pointer was completed with the exact actor path/hash; the recorder then verified46 hashes and owned processes. No frozen runtime changed.

The separate [video-body report](../video_body_review_20261004/REPORT.md) explains the missing-looking body: clips render the simplified box training asset, while the source CAD export still has its 140 meshes. A full-visual playback asset is prepared but not rendered or installed. Simplified collision envelopes also underrepresent some physical body surfaces, so present gates are simulation-contract qualification, not full-CAD collision clearance certification. Do not silently migrate current CADv32 into the frozen v29 training robot. Preserve unrelated unsaved Fusion work.

## Next focused check

1. Compare matched full v8 nominal/pushed flight and every prior-speed case against the preserved checkpoint021 baseline; preserve improving actors. Complete qualification before promoting. If simultaneous unloading still plateaus, test full objective/cost tradeoffs and effective clipped leg targets in controlled physics rather than repeating reward-identical waiting or arbitrary weights.
2. Monitor actual moving 6144/8192 memory, saved fallback/full-state recovery, 180-second ramps and active/worker/whole-chain denominators. The old reserve failure inflated marginal VRAM estimates; measure capacity before changing the guard.
3. Inspect running force distributions, cadence, geometric flight, direction and exact CAD motor speeds, plus additional measured PPO/prior gradient batches if interference is suspected. Retain the separate full-body visual and collision-fidelity audit as a documented limitation.

Evidence: initial_live_handoff.json, gate_trends.json, diagnostic_pause.json, physical_response/result.json and telemetry.npz, physical_objective_audit.json, gradient_analysis.json, running_audit.json, bilateral_counterfactuals.json, candidate_cycle_tests.json, candidate_proof.json, validation_proof.json, selected_seed.json, throughput.json and deployment.json. Fresh observation is saved separately in replacement_status.json and live_handoff.json.


![Measured forces, sole clearances and unloading quality](flight_force_evidence.png)


Fresh R34 standing and walking through1.0 m/s are requalified by matching full nominal/pushed gates009-016 on its exact transferred baseline. The moving-prior certificate is independently bound to the new v8 signature and all retained commands; see fresh_prior_certificate.json. The subsequent running actor remains unqualified.


physical_cost_tradeoff.json quantifies the remaining measured pulse tradeoff: tracking rate -0.06853, joint-acceleration cost -0.03786, height -0.01844, versus flight +0.00418. These are physical group means; do not infer an optimal learned action from the manual pulse. If v8 still plateaus, investigate phase/target timing and these full-objective contributions before another repair.


## Latest verified replacement observation

Observed 2026-10-04T21:37:25.044569+00:00: running, stage run_150, operation 019_run_150_train_001, current count 8192, selected next count 8192, 0 completed new main updates. Fresh promoted speeds: [0.0, 0.35, 0.65, 1.0]. Frozen hashes and owned processes verify in live_handoff.json. Startup/capacity/smokes are excluded from main learning progress. Read replacement_status.json for matching gate cases, memory and exact checkpoint provenance; this snapshot is not a final running/5m/s qualification.


 Fresh deployed partial main evidence: 43 observed8192 updates, exact full load, zero maximum pre-update KL/command mismatch, qualified prior20 batches/update; minGPU370MiB,host5655MiB,private growth11MiB,Torch allocation growth1MiB,about37728 active samples/s. Read new_main_evidence.json; this is not running qualification or completed-block throughput. The take-off pulse tradeoff is quantified in physical_cost_tradeoff.json: tracking-0.0685,acceleration-0.0379,height-0.0184,flight+0.00418; if plateaued test full objective and phase/target timing before another focused repair.
