# Five-hour check — October7 23:46 UTC

## Closing matching pair — exact431

Closing evidence 2026-10-08T00:08:38.898500+00:00:same R40/39000,one actual worker and55 hashes. 9063 completed+50 hash-proven recovered=9113 retained current-run,+27507 inherited;25 partial434 updates separate. Newest exact431 nominal432/pushed433 has3.5 signed3.443/3.452m/s,lateral0.270/0.242,improved from4280.351/0.326 but still fails world-lateral. Retained1.5 flight improves0.444/0.542% to1.187/1.208%,still below1.5%;0.15 signed travel worsens to0.058/0.062m/s and fails nettravel. All survive;every other saved case including3m passes. Preserve this useful nonmonotonic progress,no new qualification. New431 headRMS3.66deg,shoulder pitch ranges12.46/11.27deg and phases173/-145deg;angle/target/velocity/slow-bias plots inspected. Opposite held3.5 biases and2.5/3 slow reversals persist,no human/Unitree/optical-gaze certification. Same full learner retains all state,upper and both priors20batches/update. Resumed43164 real6144 updates has exact428 full load/zeroKL/command mismatch,minGPU1922MiB/host7191MiB,private+6MiB,Torch allocation unchanged. Completed-main rates36.40k active/30.67k worker/19.35k chain,6358.9s documented pauses excluded. No source/reward/gain/cap/optimizer/actor reset;same ACTIVE5h scope.5ms parity/filter experiments remain428-specific. Preserve improving gait while testing focused upper mean-bias/control and retention;do not restart merely from dated plateau.

See `newer_matching_pair.json`,`newer_upper` and `post_resume_learning_verification.json`. Earlier428 analysis below is preserved and is not attributed to431.


Evidence recorded 2026-10-08T00:04:39.022323+00:00. Training remains on the same R40 full-state learner and supervisor39000. No runtime/reward/gain/geometry/CAD cap/optimizer/actor reset or diagnostic actor was deployed. All55 frozen hashes and one actual worker verify. Independently qualified original3m actor349 and all-prior actor267 remain preserved separately.

## Exact saved policy and status

Reviewed checkpoint `E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261007_r40_goal_budget_full_state\428_run_350_train_140\candidate.pt`, SHA `b6a9e087f45f46cb3f931c95b773294196041b03c3398701bf774397d55668ee`; matching nominal429/pushed430 have320 robots,20s,32 trials/command. All survive. Gates belong to428,not newer partial learning. Current3.5 is unqualified;stopped/walking0.35–1.0 and2/2.5/3 cases pass,but0.15 net travel and1.5 geometric flight do not.

|Command|Nominal signed/lateral m/s|Pushed signed/lateral m/s|Matching outcome|
|---|---|---|---|
|0|-0.000/0.001|-0.000/0.017|pass both|
|0.15|0.080/0.084|0.085/0.090|forward_net_travel|
|0.35|0.324/0.088|0.327/0.094|pass both|
|0.65|0.677/0.078|0.678/0.087|pass both|
|1|1.002/0.081|0.998/0.092|pass both|
|1.5|1.453/0.069|1.451/0.081|geometric_flight|
|2|2.001/0.058|1.995/0.074|pass both|
|2.5|2.484/0.098|2.488/0.101|pass both|
|3|3.017/0.089|3.014/0.114|pass both|
|3.5|3.395/0.351|3.401/0.326|world_lateral_speed|

Fresh owned snapshot 2026-10-08T00:04:22.417150+00:00 has 9063 completed+50 hash-proven received recoveries=9113 retained current-run updates,plus27507 inherited;0 partial updates are separate.32 interrupted unsaved R39 updates stay excluded. Operation `E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261007_r40_goal_budget_full_state\433_run_350_gate_142_push` is distinct from the reviewed actor. `liveness_learning_verification.json` checks actor/critic/Adam/exploration,all KL/observed-action-command values and both teachers plus upper20 batches/update.

## Timing and controlled physics

At exact428,saved supervisor boundary then worker drain.384 free-base robots,20s,two seeds4191/4211,32 independently drawn paired q/qd states per cohort;baseline0.15/1.5 and3.5 baseline/instantaneous course/120ms velocity low-pass/380ms trailing-cycle velocity mean. Truthful student feedback,normal gravity/self-collision/CAD/native limits and complete reward remain. No action offset,head override,root support,observation rewrite,hybrid or new learned actor. External command laws are diagnostic;the old signature does not certify them. Small-reset diversity is verified but does not complete the final all-command nominal/pushed/retention protocol.

4000 actual5ms post-physics copies compare cached root link pose/world COM velocity with direct PhysX,final substep with20ms control state,and action command held overfour substeps:all maximum differences zero. Installed Isaac source and [official documentation](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html) describe mixed link pose/COM velocity in root_state_w;angular velocity is common to a rigid body. Five independent quaternion fixtures pass. At3.5 baseline,20ms heading finite difference versus final gyro has0.642rad/s RMS disagreement;averaging allfour actual post-substeps reduces it to0.058.5ms endpoint comparison is0.126. This is sampling/alignment evidence,not a proven production clock fault or policy-delay identification. Endpoint gyro fits this integrator better than a trapezoid;no timing patch deployed.

Both seeds:baseline instantaneous lateral0.368/0.366m/s;instantaneous course0.1166/0.1167;120ms0.1276/0.1277;380ms0.1333/0.1333. Both velocity-smoothing variants have lower complete mean reward than instant. Reject for immediate learning/deployment;no64/128 smoothing comparison was run or claimed. This does not permanently refute all averaging methods. Course1's earlier learned128 remains rejected for retention/drift. Diagnostic instantaneous velocities after5s differ from the gate's pose-derived fixed-frame statistics.

All384 survive without resets;commands match exactly and reward term reconstruction max9.54e-7. Every leg and3.5 cohort has zero native5ms driving clips.1.5 right elbows have isolated5ms clips(0.1594%/0.1563% samples,max1.00128cap);no sustained or locomotion-relevant ceiling established. The first CPU validator's blanketzero-clip assumption failed;source/log are preserved and the corrected validator reports actual elbow events. No gate was relaxed and no production fault inferred. See `physical_validation.json`,`timing_analysis.json`,`timing_alignment.png` and original NPZ/logs.

## Actual upper motion

Required plots for425 and428,head yaw/pitch,shoulder pitch/roll,elbow actual/target/velocity and hip coordination,were visually inspected. Population32/cmd10Hz versus16 total representatives/gate50Hz,first episodes/reset segmentation,remain distinct.4283.5 headRMS3.32deg,shoulder pitch ranges12.33/10.06deg,ipsilateral phases174/-157deg at2.67Hz. At2.5 total ranges51/56deg largely include slow bias reversal;these are not certified cycle amplitudes.0.5Hz decomposition,3s edge exclusion and actual2.02s population time origin retained.

New `upper_target_bias_audit.py` compares physical target slow components with actual joints. At4283.5 the representative left actual/target means are-23.53/-30.71deg(RMSerror7.19deg),right16.92/16.10(error0.87);left raw outputs exceedclamp3.66% of this representative window. Opposite held target biases and commanded2.5/3m slow reversals are reproduced;this is not only a plot of mechanical lag. One representative/cmd is not a population target statistic.

Three actual frozen-objective fixtures reproduce equalzero comfortable-band/symmetry loss for quiet arms,held opposite21.5deg,and zero-mean opposite sine. This is intended pointwise band behavior,not a numerical defect or a human-motion reference. Existing upper refinement cannot by itself guarantee low cycle-mean bias or natural swing. Next test must measure phase-balanced per-command actual targets/gradients and a bounded shoulder-final-row continuation against unchanged full-state64/128 controls plus independent all-gate replication. No new weight/reference/gain was deployed;existing head/arm and both priors20 batches continue.

Human comparison remains feature-based and morphology-limited: [primary arm-swing study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136043). [Official Unitree workflow](https://github.com/unitreerobotics/unitree_rl_mjlab) is not a mapped speed-matched encoder reference. No such G1/H1 curve was obtained or fabricated. Native headzero is torso-relative;Isaac/FK parity is known but original CAD optical face/camera axis remains uncalibrated. No human/Unitree/optical-gaze certification.

`gate_com_contact_audit.json` separately verifies actual mass-weighted COM/contact/joint/target/action synchronization in exact428 gates:finite arrays,zero command mismatch,no resets in16 representatives. Full-population support proxy remains distinct;transformed-sole audit is diagnostic context only,no contact-pressure or changed-margin claim.

## Resources and safe continuation

Pause257.1s including82.9s drain;same39000 resumed,no prolonged outstanding pause. Throughput denominators through saved production endpoint:36.40k active collection+PPO,30.67k worker-effective,19.35k whole-chain. Documented suspensions excluded only through that endpoint:6358.9s. Last428 block32.57k active/26.92k worker,minimumGPU1952MiB/host8534MiB,private+10MiB,Torch allocation unchanged. Capacity/smoke/diagnostic samples excluded. Original five-foot55.5k collection-only rate is a different model/contract/denominator;the current400-input23-output CAD/self-collision learner is not the old9k pipeline.

8192target,300MiBreserve,host guards,fallback and180continuoushealthyTRAINING-second ramps stay.6144 steadyfreeVRAM~1.95GiB;current8192 ramp requires2531MiB,so do not force it. Fusion stays closed and its queued work preserved. Existing full-body recordings are complete. Goal-budgetmax_hours=0 and all stall/memory guards remain. No5m/s or conditional4m/s endpoint;keep same activefive-hour automation and computer on.

## Next focused tests

1. Prioritize a narrowly scoped upper shoulder-bias continuation on the CURRENT full learner. Actual428 physical targets carry opposite held pitch,while2.5/3m bias reverses slowly. Three frozen-objective fixtures give equal zero loss to quiet,held opposite21.5deg and zero-mean swing. This is intended band/symmetry behavior,not a numerical fault. Measure phase-balanced per-command target means and actual auxiliary/PPO gradients;test bounded shoulder-final-row low-frequency mean refinement against exact full-state64/128 unchanged control and independently seeded nominal/pushed/current/all-prior/head/direction/flight gates before deployment. Preserve forward head,legs,teachers,35/1 gains;no imposed reference curve,spoof,hybrid or reset.
2. Compare fresh exact3.5 direction and0.15 net travel/1.5 geometric flight/all prior gates against reviewed428 and preserved267/349. The5ms root/cache/PhysX and command alignment test passes exactly;20ms endpoint gyro error0.642 drops to0.058 with all four actual substeps. No production clock defect or timing shift is justified.120ms velocity low-pass and380ms trailing-cycle course are rejected for immediate learning:both seeds have more drift/lower full mean reward than instantaneous feedback. Original course1 learned128 remains rejected. Use measured per-command advantages and full-state retention to select a different justified direction continuation;do not blindly repeat rejected filters,kernels,gains or teacher changes.
3. Verify one owned worker55 hashes,local ground,full actor/critic/Adam/exploration and both teachers,27507 inherited plus completed and50 proven recovered R40 updates. Exclude257.1s current diagnostic suspension and all prior documented pauses only through each timing endpoint. Keep8192target,300MiBreserve,host guards/fallback,180continuoushealthyTRAINING-second ramps;Fusion stays closed. All3.5 and leg cohorts have zero5ms driving clips;1.5 has isolated right-elbow5ms clips~0.16%,not a relevant sustained ceiling. Final5 or conditional4 still requires all unchanged gates and32 independently seeded states/cmd plus causal native limits for4.

## Final continuation verification

2026-10-08T00:07:36.659720+00:00: same39000/one actual worker/no diagnostic process;55 source hashes verified. Resumed431 completes64 genuine6144 updates,exact full428 state/zeroKL/command mismatch. MinGPU1922MiB,host7191MiB,private+6MiB,Torch allocation unchanged. Retained current-run snapshot9113 plus27507 inherited. Current ownedoperation`E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261007_r40_goal_budget_full_state\434_run_350_train_142` may be newer than reviewed428. Same nativeheartbeat remains ACTIVE5h/same thread/notifications/durable endpoint;confirmation`automation_verification.json`. Inventory40 runs/2881 operations,zero differing hashes/read warnings;8 historical missing manifests remain documented. See `closeout_verification.json`;no new qualification.

Diagnostic accounting correction:early closeout counted theWindowsvenv launcher and its actual Python child as two workers. Canonical leaf-PID accounting verifies one actualIsaac/CUDA worker;both failed assertion logs/source remain. No extra simulator was started or production fault inferred. Inventory summary is read from inventory.log rather than the unrelatedartifact list namedchecks.

Final metadata receipt 2026-10-08T00:13:05.568327+00:00: native automation reverified ACTIVE on the same thread at five hours with the newest431 handoff and preserved durable scope/notifications/endpoint. CPU inventory now contains 2882 operations across40 physical runs, zero differing hashes/read warnings, and eight documented historical missing manifests. Compact closing receipt replaces the mistakenly copied artifact list with this integrity summary. Same supervisor39000 has one actual worker and no diagnostic process; current learning operation is `E:\Aditya\robot_sim\physical_v29\training\runs\locomotion_20261007_r40_goal_budget_full_state\435_run_350_gate_143_nominal`. Reviewed431 qualification and posture remain separate from this newer actor.
