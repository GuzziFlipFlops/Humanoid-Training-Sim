"""Save evidence-backed check, journals and same native automation arguments."""
import json,hashlib,tomllib,shutil
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1];ROOT=TRAIN.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
live=read(OUT/'liveness_cpu/live_handoff.json');v=read(OUT/'liveness_learning_verification.json')
r=read(OUT/'final_gate_review.json');pair=r['latest'];upper=read(OUT/'final_upper/upper_joint_time_audit.json')
physical=read(OUT/'physical_validation.json');timing=read(OUT/'timing_analysis.json');resume=read(OUT/'diagnostic_resume.json')
assert physical['passed'] and live['state']['status']=='running' and len(live['actual_worker_pids'])==1
assert len(live['frozen_source_integrity'])==55 and all(live['frozen_source_integrity'].values())
now=datetime.now(timezone.utc).isoformat();run=Path(v['run'])
u=next(x for x in upper['records'] if x['actor_tag']=='latest' and x['test']=='nominal' and x['command_m_s']==3.5)
cases={k:[{j:c[j] for j in ['command','passed','failures','signed_speed','lateral_speed','flight','survival']} for c in pair[k]] for k in ['nominal_cases','pushed_cases']}
next_tests=[
 'Prioritize a narrowly scoped upper shoulder-bias continuation on the CURRENT full learner. Actual428 physical targets carry opposite held pitch,while2.5/3m bias reverses slowly. Three frozen-objective fixtures give equal zero loss to quiet,held opposite21.5deg and zero-mean swing. This is intended band/symmetry behavior,not a numerical fault. Measure phase-balanced per-command target means and actual auxiliary/PPO gradients;test bounded shoulder-final-row low-frequency mean refinement against exact full-state64/128 unchanged control and independently seeded nominal/pushed/current/all-prior/head/direction/flight gates before deployment. Preserve forward head,legs,teachers,35/1 gains;no imposed reference curve,spoof,hybrid or reset.',
 'Compare fresh exact3.5 direction and0.15 net travel/1.5 geometric flight/all prior gates against reviewed428 and preserved267/349. The5ms root/cache/PhysX and command alignment test passes exactly;20ms endpoint gyro error0.642 drops to0.058 with all four actual substeps. No production clock defect or timing shift is justified.120ms velocity low-pass and380ms trailing-cycle course are rejected for immediate learning:both seeds have more drift/lower full mean reward than instantaneous feedback. Original course1 learned128 remains rejected. Use measured per-command advantages and full-state retention to select a different justified direction continuation;do not blindly repeat rejected filters,kernels,gains or teacher changes.',
 'Verify one owned worker55 hashes,local ground,full actor/critic/Adam/exploration and both teachers,27507 inherited plus completed and50 proven recovered R40 updates. Exclude257.1s current diagnostic suspension and all prior documented pauses only through each timing endpoint. Keep8192target,300MiBreserve,host guards/fallback,180continuoushealthyTRAINING-second ramps;Fusion stays closed. All3.5 and leg cohorts have zero5ms driving clips;1.5 has isolated right-elbow5ms clips~0.16%,not a relevant sustained ceiling. Final5 or conditional4 still requires all unchanged gates and32 independently seeded states/cmd plus causal native limits for4.'
]
summary=(f"Same R40 learner/supervisor39000,one actual worker and55 hashes at{v['owned_snapshot_utc']}. "
 f"{v['completed_main_updates']} completed+{v['hash_proven_recovered_main_updates']} recovered={v['actual_retained_main_updates']} retained new main,+27507 inherited;{v['observed_partial_main_updates']} observed partial separate. "
 "Reviewed exact428 nominal429/pushed430:3.5 signed3.395/3.401,lateral0.351/0.326m/s;current direction fails. Both0.15 net travel and1.5 geometric flight fail;other cases including3.0 pass. All survive. Saved gates never qualify newer actors. "
 f"No production source/reward/gain/cap/optimizer/actor change. Same supervisor resumes after{resume['pause_seconds']:.1f}s. "
 "384-robot20s truthful5ms timing/filtered-course physical test,two seeds32 paired states per cohort,all survive,exact root/cache/directPhysX/control-command parity. Velocity-only smoothing is rejected versus instantaneous feedback. Mandatory upper timelines inspected:headRMS3.32deg,fast pitch ranges12.33/10.06deg,slow opposite biases and2.5/3m reversals remain;no human/Unitree/optical certification. "
 f"Rates{v['active_PPO_samples_s']/1000:.2f}k active/{v['worker_effective_samples_s']/1000:.2f}k worker/{v['whole_chain_samples_s']/1000:.2f}k chain,distinct denominators. "
 "Keep existing upper/both prior20-batch refinement,35/1 gains,full learner and all gates. Same ACTIVE5h endpoint;no5/4 or ceiling proof.")
decision=dict(recorded_utc=now,decision='Preserve and continue the same original full learner. Reject the two velocity-smoothing variants for immediate learning/deployment;no new actor or runtime patch.',
 evidence_summary=summary,reviewed_actor=pair,owned_liveness=str(OUT/'liveness_cpu/live_handoff.json'),
 physical_validation=str(OUT/'physical_validation.json'),upper_objective_counterexamples=str(OUT/'upper_objective_counterexamples.json'),
 production_changed=False,training_resumed=True,endpoint_proven=False,next_focused_tests=next_tests)
write(OUT/'decision.json',decision)
write(OUT/'evidence.json',dict(recorded_utc=now,reviewed_matching_gates=cases,checkpoint=pair['checkpoint'],sha256=pair['sha256'],
 owned_snapshot_utc=v['owned_snapshot_utc'],retained_updates=v['actual_retained_main_updates'],inherited=27507,
 source_hashes=live['frozen_source_integrity'],throughput={k:v[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s','documented_diagnostic_pause_seconds_excluded']},
 physical_validation=str(OUT/'physical_validation.json'),timing_analysis=str(OUT/'timing_analysis.json'),upper=str(OUT/'final_upper/upper_joint_time_audit.json')))
tables=['|Command|Nominal signed/lateral m/s|Pushed signed/lateral m/s|Matching outcome|','|---|---|---|---|']
for a,b in zip(pair['nominal_cases'],pair['pushed_cases']):
    reason='pass both' if a['passed'] and b['passed'] else '; '.join(sorted(set(a['failures']+b['failures'])))
    tables.append(f"|{a['command']:g}|{a['signed_speed']:.3f}/{a['lateral_speed']:.3f}|{b['signed_speed']:.3f}/{b['lateral_speed']:.3f}|{reason}|")
report=f'''# Five-hour check — October7 23:46 UTC

Evidence recorded {now}. Training remains on the same R40 full-state learner and supervisor39000. No runtime/reward/gain/geometry/CAD cap/optimizer/actor reset or diagnostic actor was deployed. All55 frozen hashes and one actual worker verify. Independently qualified original3m actor349 and all-prior actor267 remain preserved separately.

## Exact saved policy and status

Reviewed checkpoint `{pair['checkpoint']}`, SHA `{pair['sha256']}`; matching nominal429/pushed430 have320 robots,20s,32 trials/command. All survive. Gates belong to428,not newer partial learning. Current3.5 is unqualified;stopped/walking0.35–1.0 and2/2.5/3 cases pass,but0.15 net travel and1.5 geometric flight do not.

{chr(10).join(tables)}

Fresh owned snapshot {v['owned_snapshot_utc']} has {v['completed_main_updates']} completed+{v['hash_proven_recovered_main_updates']} hash-proven received recoveries={v['actual_retained_main_updates']} retained current-run updates,plus27507 inherited;{v['observed_partial_main_updates']} partial updates are separate.32 interrupted unsaved R39 updates stay excluded. Operation `{v['operation_path']}` is distinct from the reviewed actor. `liveness_learning_verification.json` checks actor/critic/Adam/exploration,all KL/observed-action-command values and both teachers plus upper20 batches/update.

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

Pause{resume['pause_seconds']:.1f}s including{resume['worker_drain_seconds']:.1f}s drain;same39000 resumed,no prolonged outstanding pause. Throughput denominators through saved production endpoint:{v['active_PPO_samples_s']/1000:.2f}k active collection+PPO,{v['worker_effective_samples_s']/1000:.2f}k worker-effective,{v['whole_chain_samples_s']/1000:.2f}k whole-chain. Documented suspensions excluded only through that endpoint:{v['documented_diagnostic_pause_seconds_excluded']:.1f}s. Last428 block32.57k active/26.92k worker,minimumGPU1952MiB/host8534MiB,private+10MiB,Torch allocation unchanged. Capacity/smoke/diagnostic samples excluded. Original five-foot55.5k collection-only rate is a different model/contract/denominator;the current400-input23-output CAD/self-collision learner is not the old9k pipeline.

8192target,300MiBreserve,host guards,fallback and180continuoushealthyTRAINING-second ramps stay.6144 steadyfreeVRAM~1.95GiB;current8192 ramp requires2531MiB,so do not force it. Fusion stays closed and its queued work preserved. Existing full-body recordings are complete. Goal-budgetmax_hours=0 and all stall/memory guards remain. No5m/s or conditional4m/s endpoint;keep same activefive-hour automation and computer on.

## Next focused tests

{chr(10).join(str(i+1)+'. '+s for i,s in enumerate(next_tests))}
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
plots=[p for folder in ['initial_upper','final_upper'] for p in (OUT/folder).glob('*.png')]+[OUT/'timing_alignment.png']
write(OUT/'plot_qa.json',dict(recorded_utc=now,inspected_by_assistant=True,files=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in plots],findings='425 held opposite biases;428 slow2.5/3 shoulder reversal and unequal fast swing/head oscillation;50Hz actual velocity spikes;5ms heading and gyro follow the same waveform. No visual natural-motion certification.'))
entry=f"\n\n## Five-hour check October7 23:46 UTC — timing parity and upper-target bias\n\n{summary}\n\nReport: {OUT/'REPORT.md'}. Physical384/two independent seeds and5 quaternion fixtures pass;velocity120ms/cycle380ms variants rejected before learning. Three frozen upper-objective fixtures document the band/symmetry gap;no new posture candidate deployed. First zero-clip validator assumption failed and is preserved;isolated1.5 elbow clips are reported,no ceiling. Updated actual joint/target/velocity/bias plots and COM/contact audit;current full learner resumed. Next: bounded phase-balanced shoulder-bias continuation/control/all gates,retention-preserving direction diagnosis,and fresh resource/full-state audit.\n"
for f in [TRAIN/'ITERATION_HISTORY.md',ROOT/'docs/robot/TRAINING_JOURNAL_20261001.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    with f.open('a',encoding='utf-8') as h:h.write(entry)
marker='adaptive_check_20261007_2346_record'
for f in [Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]:
    old=f.read_text(encoding='utf-8');assert marker not in old
    f.write_text(f'<!-- {marker} -->\n{summary} Read {OUT/"REPORT.md"},FIVE_HOUR_CHECK and freshACTIVE_RUN/state. Next focused shoulder-bias/control and retained-direction checks;earlier entries remain historical.\n<!-- /{marker} -->\n\n'+old,encoding='utf-8')
next_path=TRAIN/'automation/NEXT_CHECK.json';shutil.copy2(next_path,OUT/'NEXT_CHECK_before.json');n=read(next_path)
for key in ['focused_tests','next_tests','focus']:n[key]=next_tests
n.update(recorded_utc=now,report=str(OUT/'REPORT.md'),decision=str(OUT/'decision.json'),latest_repair_decision=decision['decision'],active_run=str(run),run=str(run),
 checkpoint=pair['checkpoint'],checkpoint_sha256=pair['sha256'],selected_actor=pair['checkpoint'],latest_gate_trends=cases,
 latest_closing_gate_pair=pair,newer_matching_pair=pair,current_stage='run_350',current_stage_qualified=False,running_qualified=False,
 current_owned_operation=v['operation_path'],new_main_updates=v['completed_main_updates'],recovered_main_updates=v['hash_proven_recovered_main_updates'],actual_retained_current_run_main_updates=v['actual_retained_main_updates'],
 inherited_main_updates=27507,inherited_main_updates_preserved=27507,observed_partial_main_updates=v['observed_partial_main_updates'],
 latest_observation=v['owned_snapshot_utc'],latest_liveness=str(OUT/'liveness_cpu/live_handoff.json'),latest_evidence=str(OUT/'evidence.json'),production_change_this_check=False,
 latest_verified_state=str(OUT/'liveness_cpu/live_handoff.json'),last_main_resume=str(OUT/'diagnostic_resume.json'),last_diagnostic_pause_seconds=resume['pause_seconds'],
 throughput={k:v[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s','documented_diagnostic_pause_seconds_excluded']},throughput_evidence=str(OUT/'liveness_learning_verification.json'),
 mandatory_upper_audit=str(OUT/'final_upper/upper_joint_time_audit.json'),mandatory_upper_bias_audit=str(OUT/'final_upper/upper_bias_audit.json'),
 upper_joint_time_audit=str(OUT/'final_upper/upper_joint_time_audit.json'),upper_target_bias_audit=str(OUT/'final_upper/upper_target_bias_audit.json'),
 upper_objective_counterexamples=str(OUT/'upper_objective_counterexamples.json'),timing_parity=str(OUT/'timing_analysis.json'),
 diagnostic_pauses_this_check=[str(OUT/'diagnostic_resume.json')],synchronized_com_contact_audit=str(OUT/'gate_com_contact_audit.json'),velocity_ceiling_proven=False,
 cadence_hours=5,target_environments=8192,reserve_MiB=300,newer_observation_scope='Reviewed matching/upper actor428;owned live operation may be newer. Never infer newer gates or natural motion from liveness.',
 closing_learning_verification=str(OUT/'liveness_learning_verification.json'),closing_owned_observation=str(OUT/'liveness_cpu/live_handoff.json'))
write(next_path,n)
automation=Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml')
shutil.copy2(automation,OUT/'automation_before.toml');old=tomllib.loads(automation.read_text(encoding='utf-8'))
prefix=old['prompt'].split('Latest dated five-hour check',1)[0]
dated='Latest dated five-hour check,October7 23:46 UTC;owned evidence '+v['owned_snapshot_utc']+': '+summary+'\nRead diagnostics/adaptive_check_20261007_2346/REPORT.md,decision.json,physical_validation.json,timing_analysis.json,timing_alignment.png,upper_objective_counterexamples.json,final_upper actual angle/target/velocity/bias plots and liveness_learning_verification.json,then freshACTIVE_RUN/state. No diagnostic learned actor or runtime change.\n\nNext focused tests:\n'+ '\n'.join(str(i+1)+'. '+s for i,s in enumerate(next_tests))+'\n\nKeep same ACTIVEfive-hour heartbeat/thread/notifications/durable verified5 or qualified4+causal native-limit endpoint. Preserve all learned state and independently certified actors;no clock patch,cap raise,spoof,hybrid or from-scratch restart. Videos fulfilled,Fusion closed/queuedwork preserved,computeron.'
args=dict(id=old['id'],mode='update',kind='heartbeat',name=old['name'],prompt=prefix+dated,rrule=old['rrule'],status='ACTIVE',targetThreadId=old['target_thread_id'])
if 'notification_policy' in old:args['notificationPolicy']=old['notification_policy']
write(OUT/'automation_update_args.json',args)
print(json.dumps(dict(report=str(OUT/'REPORT.md'),retained=v['actual_retained_main_updates'],reviewed_checkpoint=pair['checkpoint'],pending_native_automation_update=True),indent=2))
