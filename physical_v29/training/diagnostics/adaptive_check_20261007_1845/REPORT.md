# Five-hour check — October7,18:45 UTC

Closing owned evidence 2026-10-07T19:28:21.723650+00:00: the same R40 supervisor 39000 has one actual worker and all 55 hashes verified. It retains 7207 completed + 50 recovered = 7257 new main updates, plus 27,507 inherited. Exact actor 344_run_350_train_112 reaches 3.475/3.482 m/s signed at command3.5, but lateral 0.495/0.465 fails both direction tests. Both0.15 path-efficiency and1.5 geometric-flight tests fail; pushed3.0 lateral regresses. All survive. The earlier338 pair's3.0 recovery is preserved and is not attributed to344. The newer344 angle/target,velocity and slow-bias plots are also visually inspected. At3.5,head RMS3.87deg and shoulder pitch ranges13.09/6.76deg. Fast asymmetry,held biases and gait-linked head oscillation persist;these diagnostics do not certify human/Unitree motion or optical gaze. Fresh completed-main rates are 37.62k active/31.91k worker/20.24k whole-chain samples/s, with 6101.8s documented suspensions excluded. Resumed341 loads exact338 actor/critic/Adam/exploration;344 also loads its predecessor exactly. All KL/command checks and three20-batch auxiliaries pass. Last6144 block minimumGPU1937MiB/host9054MiB,private+12MiB,Torch allocation unchanged.

[Newer exact gates](liveness_gate_review.json),[newer upper plots/data](closing_newer_upper/upper_joint_time_audit.json) and[resumed learning](liveness_learning_verification.json). The338 review below is the earlier saved diagnostic boundary.

Same R40 learner/supervisor39000,one actual worker,55 frozen hashes. 7143 completed+50 hash-proven recovered=7193 retained new main updates,plus27507 inherited. Exact reviewed 338_run_350_train_110 reaches3.5-command signed3.423/3.436m/s with0.340/0.308m/s lateral;both current direction gates fail. Both0.15 net travel and1.5 geometric flight fail;other stopped/prior cases including recovered3m pass in this pair. Qualification never extends to newer partial actors.

Preserve and resume the exact original full-state learner. Saved-actor response and passive actual gradients narrow the next diagnosis; no new candidate has independent learning/retention evidence sufficient for deployment. Previous rejected candidates remain rejected.

## Exact saved gates and progress

Owned observation:2026-10-07T19:16:53.251445+00:00. Checkpoint SHA`5e5f7da02aba7031defca38cec1897a3df75d3f2855905b9966f5348c32696c6`. Matching320-robot20-second nominal/pushed tests contain32 trials per command. All survive. Nominal cloned resets do not establish final independently seeded initial-state diversity. Separately certified original349 at3m and267 all-prior remain preserved.

|Command m/s|Signed nominal/pushed|Lateral nominal/pushed|Failed nominal/pushed checks|
|---:|---:|---:|---|
|0|-0.000/-0.000|0.001/0.016|pass; pass|
|0.15|0.106/0.108|0.075/0.080|forward_net_travel; forward_net_travel|
|0.35|0.322/0.327|0.078/0.087|pass; pass|
|0.65|0.654/0.655|0.059/0.068|pass; pass|
|1|0.953/0.951|0.063/0.076|pass; pass|
|1.5|1.479/1.476|0.070/0.079|geometric_flight; geometric_flight|
|2|1.992/1.971|0.100/0.099|pass; pass|
|2.5|2.507/2.496|0.076/0.086|pass; pass|
|3|3.014/3.012|0.094/0.114|pass; pass|
|3.5|3.423/3.436|0.340/0.308|world_lateral_speed; world_lateral_speed|

Compared with the initial332 pair,338 recovers3m hopping/direction and retains1m support,while current3.5 drift decreases. Retention remains nonmonotonic:0.15 efficiency and1.5 flight still fail. Running1.5 geometric flight is0.2222%/0.3958%,below unchanged1.5% minimum. Neither a3.5 command nor these signed speeds qualify the whole stage. No4/5m or native-limit endpoint is established.

## Bounded saved-actor physical comparison

Exact boundary338,original254,unchanged-control128 and rejected-course128 actors are replayed in384 free-base robots for20 seconds. Each of16 cohorts uses24 independently drawn small q/qd states,identical across actor/law branches. Root,gravity,self-collision,geometry,35/1 shoulder gains and all native motor caps remain. This is a diagnostic24-state protocol,not the final32-state qualification. Actor hashes,initial states,commands and synchronized joints/targets/forces/root/actions are saved in[heading_response](heading_response/result.json). All actors load exactly,all survive without resets,observed/action command difference is0 and full reward sums reconstruct within9.5e-07. No pulses,root support,input rewrite or hybrid is deployed.

|Saved actor|3.5 baseline/course lateral m/s|Complete reward-rate change|
|---|---:|---:|
|current|0.368/0.129|+0.171|
|original254|0.386/0.129|+0.207|
|control128|0.430/0.145|+0.169|
|course128|0.634/0.185|+0.291|

Statistics above are instantaneous world velocity after5s warmup,not the verifier's sampled-pose direction statistics. Current338 still responds physically to course feedback;the learned course128 actor has more drift under the same law than original254/current338. Its0.15 travel and1.5 lateral/flight behavior also weakens. After warmup no group's yaw command saturates the+/-0.2rad/s bound. Every sampled native5ms CAD driving-speed clip statistic is zero over4000 substeps. These results reject command-bound saturation as the explanation in this experiment;they do not prove5m feasible or establish a high-speed motor ceiling. All diagnostic actors and command laws remain outside production certification.

The[response plot](saved_actor_response.png) was inspected. Desired-yaw variation is dominated near2.66Hz,with substantial within-stride oscillation. The[angular audit](angular_heading_audit.json) distinguishes world angular-z from quaternion heading differences. Its analytic gyro/heading-rate reconstruction and50Hz forward differences disagree substantially;sampling/coordinate/phase attribution remains unresolved. No production clock fault or smoothing benefit is inferred. Establish5ms parity and controlled phase-response before a feedback-timing candidate. The course term is a local command adaptation,not copied Unitree numerical guidance.

## Actual PPO,teacher and upper gradients

Five CPU hook-order fixtures pass. Four genuine4096-env diagnostic PPO updates load exact338 actor/critic/Adam/exploration and both teachers,with all KL/command checks0. Its learned actor is discarded. Four actual minibatches capture exact advantages,means,Gaussian parameters,teacher labels and incoming PPO mean derivatives before either retention hook. The new passive before-standing capture resolves subtraction uncertainty on these fresh samples:CPU standing derivative differs by at most4.5e-10;moving analytic-surrogate cosine is1. Earlier failed accounting evidence remains preserved and is not a production defect.

Current-command PPO mean-network parameter norms range17.86–29.63,prior norms2.09–2.61,upper auxiliary norms2.17–2.68. Running/prior cosine ranges-0.259–0.265;upper/PPO cosine-0.444–0.380. Alignment has both signs. Per-command advantages,coverage,effective clipped errors and contribution norms are saved in[current_gradient_analysis.json](current_gradient_analysis.json). These are before global clipping/Adam,mean-network only;four minibatches do not identify a universal conflict or justify an arbitrary weight change.

## Required upper-body time series

Both initial332 and reviewed338 head yaw/pitch,shoulder pitch/roll,elbow angles,physical targets and measured velocities were visually inspected,with[plot hashes/QA](plot_qa.json). Population32/cmd10Hz and16 representative robots/gate50Hz are separated and reset-segmented. At3.5 nominal,head RMS3.43deg;physical pitch ranges14.23/7.52deg. Slow pitch means-15.99/6.40deg and faster residual ranges13.43/6.65deg. Arm/ipsilateral-hip phases147.0/162.7deg at2.67Hz;phase and amplitude are both diagnostic. At2.5/3m,large angle span includes slow shoulder bias drift. Fast target excursions exceed actual shoulder swing;roll/elbow imbalance and gait-linked neck oscillation remain. No new gain or imposed trajectory is selected. Existing upper-only posture/symmetry refinement continues20 batches/update and does not redefine physical proportions or certify natural motion.

The[angle plot](final_upper/upper_joint_angles.png),[velocity plot](final_upper/upper_joint_velocities.png),[slow-bias plot](final_upper/upper_bias_decomposition.png) andCSV/JSON data remain nearby. The0.5Hz filter,3s edges and post-warmup2.02s origin are descriptive and never cross resets;four decomposition fixtures pass. Native neckzero is torso-relative. Isaac-head/FK parity is known;original CAD face/camera optical axis remains uncalibrated.

Primary references:[official Isaac Lab heading generator](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py),[primary human arm-swing features](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043),[official Unitree workflow](https://github.com/unitreerobotics/unitree_rl_mjlab). No mapped speed-matched G1/H1 encoder trajectories were obtained;no reference curve,human likeness or optical world gaze is fabricated or certified.

## Resume,resources and failed attempts

Same supervisor39000 resumed after two bounded suspensions:370.7s first attempt and310.7s successful diagnostic,including respective worker drains108.9/28.9s. Total681.4s. The first diagnostic found a stale initial command/history cache after its own perturbed-state write,before rollout. Its source,traceback,receipts and owned failed-Kit cleanup are preserved in[attempt01](attempt01_initial_cache_failure/physical_probe_failure.json). Production resumed automatically. Diagnostic-only reset-history/cache rebuilding fixed the setup;live sources never changed. A first CPU fixture incorrectly demanded bitwise equality to reordered algebra;the corrected comparison uses actual unhooked autograd and passes. A premature CPU analysis invocation before result creation is also preserved. None is a reproduced training fault.

Completed-main rates37.68k active collection+PPO/31.97k worker-effective/20.28k whole-chain samples/s use distinct denominators and exclude documented pauses only when before their measurement endpoint. Main capacity/smoke/diagnostic samples are excluded. Latest completed6144-env block has minimumGPU1940MiB,host8455MiB,private growth12MiB,Torch allocation growth0MiB. Resumed full-state load/KL/teacher/upper evidence is in[final learning verification](final_learning_verification.json). Original local ground and55 source hashes verify;no optimizer/actor reset,reward/physics/limit change or new production parameter was deployed.

Fusion is already closed;queued work stays preserved and it should not be reopened for routine checks. Stable6144 training now has about1.9GiB free VRAM and9GiB host RAM. Current empirical ramp asks2531MiB free for8192;1942MiB does not meet it. Preserve8192 target,300MiB reserve,host guards/fallback and180 continuous healthy TRAINING seconds. Evaluation/startup free memory is not steady capacity.32 unsaved R39 updates remain excluded;50 recovered R40 updates are hash-proven.

The exact saved nominal/pushed gates also retain actual mass-weighted COM, contacts, joint targets and actions at50Hz for16 representatives. Both saved traces are finite with no resets or command mismatch; per-speed statistics and hashes are in[gate_com_contact_audit.json](gate_com_contact_audit.json). Whole-population contact/support proxies stay separate. The prior transformed-sole diagnostic remains context only; neither COM offsets nor geometry certify pressure or replace any gate.

## Next focused tests

1. Preserve current neutral-head full learner and compare fresh exact3.5 direction,0.15 path efficiency and1.5 geometric flight plus every prior gate. The paired384-robot response reproduces useful course feedback but course-trained128 worsens residual drift even under its own law,without yaw-command saturation or5ms driving clips. Actual desired-yaw variation is gait-linked near2.66Hz. Before another learning candidate, verify5ms quaternion-heading/world-gyro/action-command timing and phase-resolved response;50Hz heading differences and gyro kinematics disagree and do not establish a production clock defect. Then test bounded cycle-averaged course/heading feedback or another evidence-based timing candidate in truthful normal-limit physics,complete return and separate seeds. Require fresh versioned provenance,exact-current-full-state64/128 controls and all gates before deploying. Do not merely raise gains,caps or blindly repeat course1.
2. Mandatory actual head/shoulder pitch+roll/elbow angle-target-velocity timelines,arm-leg phase and slow-bias decomposition. Current fast arm swing remains unequal and2.5/3m slow bias drifts. Four actual minibatches show upper versus PPO alignment of both signs; no universal opposition. Measure per-command slow target bias versus measured joint bias and phase-conditioned effective targets,then test a narrow training-only bias refinement using the current full state against64/128 control/all gates. Preserve original35/1 gains,forward head,legs,teachers and useful motion;no imposed reference curves,spoofs,hybrids or from-scratch restart. Human/Unitree encoder likeness and optical gaze remain unverified.
3. Verify same supervisor/oneworker55 hashes,local ground,full actor/critic/Adam/exploration and both teachers. Account27507 inherited plus completed and50 recovered;32 unsaved R39 remain excluded. Exclude both current diagnostic pauses370.7/310.7s once before the timing endpoint,plus prior documented pauses. Fusion is already closed;do not reopen it. Actual6144 steady freeGPU about1942MiB,host about9GiB;current scheduler needs2531MiB free for8192,so retain300MiB reserve,host guards and180 continuous healthy TRAINING seconds rather than forcing the ramp. Endpoint still needs every gate and32 independently seeded initial states/cmd;only qualified5 or qualified4 with causal native limits blocking5 ends training/automation.

Keep the same ACTIVE five-hour heartbeat,thread,notification preference and verified5/conditional4 endpoint. Final success requires>=4.95m/s or>=3.96m/s with causal native limits proving5 blocked,every unchanged gate and32 independently seeded initial states per command. Save,pause training and automation,and brief with computer on only then. Completed videos remain fulfilled.
