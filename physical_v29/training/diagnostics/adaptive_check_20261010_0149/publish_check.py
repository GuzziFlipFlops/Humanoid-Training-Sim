"""Publish completed evidence only; no training or automation mutation here."""
import hashlib,json,tomllib
from datetime import datetime,timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1];ROBOT=TRAIN.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,d:Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')
MARKER=OUT.name+'_record';STAMP=datetime.now(timezone.utc).isoformat()
v=read(OUT/'final_learning_verification.json');pair=read(OUT/'closing_gate_review.json')['latest']
m=read(OUT/'matched_full_state_review.json');check=read(OUT/'experiment_validation.json');resume=read(OUT/'diagnostic_resume.json')
parity=read(OUT/'full_baseline_parity.json');assert parity['passed']
decision=read(OUT/'decision.json');priorities=read(OUT/'next_priorities.json')
u=read(OUT/'final_upper/upper_joint_time_audit.json');bias=read(OUT/'final_upper/upper_bias_audit.json')
um=read(OUT/'upper_matched_comparison.json');old=read(OUT/'previous_end_verification.json')
assert m['all_expected_components_complete'] and check['passed'] and um['all_four128_actors_complete']
assert len(priorities) in [1,2,3] and decision['diagnostic_actor_deployed'] is False
assert v['supervisor_pid']==39000 and len(v['actual_worker_pids'])==1 and v['frozen_files_verified']==55 and v['protected_authored_worker_files_equal']==53
assert read(OUT/'final_upper/audit_set.json')['all_plots_visually_inspected']
assert read(OUT/'earlier_baseline_upper/audit_set.json')['all_plots_visually_inspected']
assert all(read(OUT/f'seed{s}_{b}_upper/visual_inspection.json')['visually_inspected'] for s in [317,331] for b in ['control','earlier'])
assert any(b['updates']>=64 and b['full_load_proof']['source_sha256']==resume['checkpoint_sha256'] for b in v['blocks'])
assert v['hash_proven_recovered_main_updates']==50 and v['inherited_actual_main_updates']==27507
known={b['folder'] for b in old['blocks']};added=[b for b in v['blocks'] if b['folder'] not in known]
delta=dict(retained_main_updates=sum(b['updates'] for b in added),actual_main_samples=sum(b['samples'] for b in added),
 active_collection_PPO_seconds=sum(b['active_seconds'] for b in added),worker_seconds=sum(b['worker_seconds'] for b in added),
 elapsed_endpoint_seconds=v['whole_chain_measurement_endpoint_unix']-old['whole_chain_measurement_endpoint_unix'],
 documented_pause_seconds=v['documented_diagnostic_pause_seconds_excluded']-old['documented_diagnostic_pause_seconds_excluded'])
delta['chain_seconds_excluding_documented_pauses']=delta['elapsed_endpoint_seconds']-delta['documented_pause_seconds']
for label,seconds in [('active',delta['active_collection_PPO_seconds']),('worker_effective',delta['worker_seconds']),('whole_chain',delta['chain_seconds_excluding_documented_pauses'])]:
 assert seconds>0;delta[label+'_samples_s']=delta['actual_main_samples']/seconds
case=lambda t,s:next(c for c in pair[t+'_cases'] if c['command']==s)
fast=[case(t,3.5) for t in ['nominal','pushed']]
speeds='/'.join(f"{c['signed_speed']:.3f}" for c in fast);lat='/'.join(f"{c['lateral_speed']:.3f}" for c in fast)
failures={t:[dict(command=c['command'],failures=c['failures']) for c in pair[t+'_cases'] if not c['passed']] for t in ['nominal','pushed']}
actor=Path(pair['checkpoint']).parent.name;head=next(r for r in u['records'] if r['actor_tag']=='latest' and r['test']=='nominal' and r['command_m_s']==3.5)
slow=next(r for r in bias['records'] if r['test']=='nominal' and r['speed_m_s']==3.5)
last=v['blocks'][-1];current_samples=sum(b['samples'] for b in v['blocks'])
summary=(f"Same R40 full learner and supervisor 39000 at {v['owned_snapshot_utc']}, with one actual worker, 55 frozen hashes and 53 protected files. "
 f"{v['completed_main_updates']:,} completed + 50 recovered = {v['actual_retained_main_updates']:,} retained current-run updates, plus 27,507 inherited; "
 f"{v['observed_partial_main_updates']} partial updates are separate. Exact actor {actor}, SHA {pair['sha256']}: command 3.5 gives signed speed {speeds} and lateral speed {lat} m/s. "
 f"Current nominal/pushed passes: {[c['passed'] for c in fast]}. Failures: {failures}. Saved gates never qualify newer actors. No 5/4 m/s or causal ceiling endpoint is established.")
evidence=dict(recorded_utc=STAMP,summary=summary,current_exact_pair=pair,current_owned_learning=v,measured_since_previous=delta,
 matched_full_state=m,experiment_validation=check,decision=decision,diagnostic_resume=resume,
 retained_main_totals=dict(current_run_actual_samples=current_samples,current_run_updates=v['actual_retained_main_updates'],inherited_updates=27507),
 upper=dict(exact_checkpoint=pair['checkpoint'],all_command_timelines_visually_inspected=True,matched128_actors=4,
  head_RMS_deg=head['head_rms_deg'],shoulder_pitch_ranges_deg=[head['joints'][n]['median_per_robot_p95_p05_range_deg'] for n in ['left_shoulder_pitch','right_shoulder_pitch']],
  slow_pitch_means_deg=[slow['joints'][n]['slow_bias_mean_deg'] for n in ['left_shoulder_pitch','right_shoulder_pitch']],
  human_Unitree_encoder_or_optical_gaze_certified=False),
 native_caps_unchanged=True,production_change=False,endpoint_established=False,computer_on=True)
write(OUT/'evidence.json',evidence)
table='|Command m/s|Signed N/P m/s|Lateral N/P m/s|True flight N/P %|Failures N/P|\n|---|---|---|---|---|\n'
for s in [0,.15,.35,.65,1,1.5,2,2.5,3,3.5]:
 a,b=case('nominal',s),case('pushed',s)
 table+=f"|{s:g}|{a['signed_speed']:.3f}/{b['signed_speed']:.3f}|{a['lateral_speed']:.3f}/{b['lateral_speed']:.3f}|{100*a['flight']:.3f}/{100*b['flight']:.3f}|{','.join(a['failures']) or 'pass'} / {','.join(b['failures']) or 'pass'}|\n"
trials='|Seed|Branch|Updates|Ordinary N/P pass|Diverse N/P pass|\n|---|---|---|---|---|\n'
for r in m['records']:
 tests=r['tests'];ordinary='/'.join(str(tests[t]['passed']) for t in ['nominal','push'])
 diverse='/'.join(str(tests[t]['passed']) for t in ['diverse_nominal','diverse_push']) if r['updates']==128 else 'tested at128'
 trials+=f"|{r['seed']}|{r['branch']}|{r['updates']}|{ordinary}|{diverse}|\n"
report=f'''# Five-hour humanoid training check — October10,2026,01:49UTC

{decision['conclusion']}

## Exact current saved actor

{summary}

{table}
Every ordinary gate uses320 robots for20seconds,32 per command. Ordinary nominal clones are distinct from the32 independently generated q/qd states per command in this comparison. Startup falls and all existing landing,flight,direction,contact and prior classifiers remain unchanged. The full3.5 stage requires every current/stopped/prior case; no gate is skipped to progress to4/4.5/5. Current-case evidence from separately passing actors605,908,917,1007,1061/1064 is preserved,not attributed to newer learning.

## Earlier complete learned-state comparison

Anchor current checkpoint: **{resume['checkpoint']}**, SHA **{resume['checkpoint_sha256']}**. Earlier neutral-head 267 SHA **20a133fdd4843118d816172239201a387ca947b38952fb38f094a156afa0a320**. Fresh CPU metadata, teacher and context checks pass. Unchanged current nominal replay matches **{parity['array_count']} arrays** and every evaluation case bit for bit. The driver verifies 13 basic arrays; a separate CPU check verifies all synchronized representative fields. [Replay parity](full_baseline_parity.json). Fresh earlier 267 nominal/pushed and independently seeded nominal/pushed tests reverify every prior through 3 m/s; its 3.5 direction remains failed at baseline. [Starting-state comparison](starting_motion_comparison.json).

Both branches preserve their complete actor,critic,Adam and trained exploration. Original reward v8,verifier v6,teacher actor bytes and frozen head context,35/1 gains,PPO,commands,truthful400-input/23-output histories,local ground and native caps are identical. The initial complete checkpoint is the branch difference; this does not isolate critic,optimizer or actor causality. No diagnostic target,hybrid controller,observation rewrite or gradient hook is used.

{trials}
Two learning seeds317/331,4096 environments,64/128 updates per branch. Each learned checkpoint gets unchanged nominal/pushed gates;128 also gets nominal/pushed32 individually seeded initial states per command. Independent validation confirms32 comparison operations,512 genuine diagnostic learning updates and50,331,648 samples. Sixteen64-environment smoke/reload updates add24,576 separate samples. None enters main progress. [Exact results and failures](matched_full_state_review.json),[independent provenance/load/diversity validation](experiment_validation.json).

{decision['interpretation']}

[Measured force, sole-clearance and COM comparison](flight_retention_analysis.json) separates selected 50 Hz traces from full-population gates. [Mixed training metrics](training_metric_comparison.json) record logged reward components and bounded PPO returns; they do not certify per-command complete return or physical qualification. The moving exploration value 0.7 is an upper bound, not a forced floor. [Retention-constraint research](constraint_reward_review.md) is an undeployed follow-up, with primary-source scope and required counterexamples recorded. No CPO or new reward is claimed as tested.

## Upper-body motion

Actual head yaw/pitch,shoulder pitch/roll and elbow angle,target and velocity timelines are saved and visually inspected for the initial current actor,fresh earlier267,all four128-update comparison actors and exact closing current actor. Closing all-command14plots bind**{actor}**. Matched actors have focal0.15/1.5/2.5/3.5 representative plots plus all-command population statistics. Population32/cmd at10Hz is distinguished from16 representatives/gate at50Hz; resets are segmented. The0.5Hz descriptive slow decomposition excludes3second edges and does not certify fast jitter or natural cycle motion.

Closing3.5 native neck RMS**{head['head_rms_deg']:.2f}deg**,shoulder pitch p95-p05 ranges**{evidence['upper']['shoulder_pitch_ranges_deg']}deg**,slow pitch means**{evidence['upper']['slow_pitch_means_deg']}deg**. Held bias,unequal pitch/roll/elbows and gait-linked neck motion remain. Phase is evaluated alongside amplitude; near-static arms have weak phase evidence. Existing upper fine-tuning remains20 auxiliary batches per main update. [Actual closing timelines](final_upper/audit_set.json),[matched physical motion](upper_matched_comparison.json).

The primary [Collins2009 study](https://pmc.ncbi.nlm.nih.gov/articles/PMC2817299/) studies walking at1.25m/s; its earlier inspected shoulder/elbow time plots do not provide a mapped3.5m/s running reference. Numerical axes/signs/rest poses and morphology mapping are unverified. The freshly inspected [official Unitree material](https://github.com/unitreerobotics/unitree_rl_mjlab) provides training/imitation instructions but no usable mapped speed-matched encoder curves in that inspected material. No curve is fabricated; human/Unitree likeness and calibrated optical gaze remain uncertified. Native-zero neck is torso-relative. [Reference scope](reference_review.json).

## Actual learning,resources and limits

Retained R40 main samples**{current_samples:,}**. Since the previous exact owned endpoint:**{delta['retained_main_updates']:,}retained updates/{delta['actual_main_samples']:,}actual samples**,using{delta['active_collection_PPO_seconds']:.1f}active collection/PPO seconds and{delta['worker_seconds']:.1f}worker seconds. The27507 inherited updates are declared separately; their samples are not inferred from a fixed environment count. Fifty hash-received R40 recovery updates count;32 interrupted unsaved R39 updates do not. All evaluation,smoke,capacity and diagnostic samples are excluded.

Active/worker-effective/whole-chain retained rates:**{v['active_PPO_samples_s']/1000:.2f}k/{v['worker_effective_samples_s']/1000:.2f}k/{v['whole_chain_samples_s']/1000:.2f}k samples/s**. Documented suspensions are excluded only through each measurement endpoint. Same39000 resumes after**{resume['pause_seconds']:.1f}seconds**,including worker drain. A genuine64-update main receiver loads exact anchor actor,critic,Adam and exploration,with zero pre-update KL/command mismatch and all three auxiliaries20/update. [Recovery and timing proof](final_learning_verification.json).

Last verified main block**{last['folder']}**,**{last['updates']}updates/{last['environments']}environments**,minimum free GPU/host**{last['minimum_free_gpu_MiB']}/{last['minimum_host_available_MiB']}MiB**,private growth{last['private_memory_growth_MiB']}MiB,Torch allocation growth{last['torch_allocation_growth_MiB']}MiB. Keep8192 target,reserve300MiB,host guards/fallback and180 continuous healthy TRAINING seconds before upward ramps. Evaluation/diagnostic memory is not main capacity.

Closing synchronous50Hz COM/contact/target/action arrays are finite,with zero action-command mismatch. Full-population10Hz motor speeds use actual case labels and exact CAD differential Jacobians/reductions. Sparse cap crossings do not establish5ms driving clipping or sustained available-torque limitation. This comparison adds no new native5ms request/application probe; earlier probes stay actor/cohort-specific. Earlier4/4.5/5 extrapolation failures do not prove a ceiling. All native caps remain; no4/5 endpoint is claimed. [Synchronous data](closing_COM_contact_audit.json),[nominal motors](closing_motor_nominal.json),[pushed motors](closing_motor_pushed.json).

## Next focused tests

'''+''.join(f'{i}. {p}\n\n' for i,p in enumerate(priorities,1))+'''The same five-hour heartbeat stays active. Only fully qualified5m/s,or qualified4m/s with causal native limits proving5blocked,ends training. Every endpoint needs all unchanged gates and32 independently seeded states per command. At that endpoint save,pause both,prevent restart and brief with the computer on. Fusion stays closed;queued work and completed videos are preserved.
'''
from render_check_report import render_report
report=render_report(evidence,delta,last,head,parity,table,trials,priorities)
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
handoff=('Latest dated five-hour check October 10, 01:49 UTC. '+summary+'\n\n'+decision['conclusion']+' '+decision['interpretation']+
 f" Two learning seeds, 317/331, exact full-state 64/128 controls and independently seeded states were tested; no diagnostic actor was deployed. The same supervisor 39000 resumes after {resume['pause_seconds']:.1f} seconds. "
 f"Exact main recovery, zero KL/command mismatch and all three 20-batch auxiliaries pass. Active/worker-effective/whole-chain rates are {v['active_PPO_samples_s']/1000:.2f}k/{v['worker_effective_samples_s']/1000:.2f}k/{v['whole_chain_samples_s']/1000:.2f}k samples/s; documented pauses are excluded only through each measurement endpoint. "
 f"Required upper timelines were inspected at every command. Closing native neck RMS is {head['head_rms_deg']:.2f} degrees; held bias and asymmetry persist. Human/Unitree encoder likeness and optical gaze remain unverified. "
 f"Read diagnostics/{OUT.name}/REPORT.md, decision.json, matched_full_state_review.json, starting_motion_comparison.json, older_branch_CPU_preflight.json, experiment_validation.json, upper_matched_comparison.json, final_upper, closing_gate_review.json and final_learning_verification.json; then fresh ACTIVE_RUN/state.\n\nNext focused tests:\n"+
 '\n'.join(f'{i}. {p}' for i,p in enumerate(priorities,1))+'\n\nKeep the same ACTIVE five-hour heartbeat, thread, notification preference and verified 5 m/s or qualified 4 m/s plus causal native-limit endpoint. At that endpoint save, pause both, prevent restart and brief with the computer on. Preserve the complete learned state, forward head, both teachers, existing upper 20 batches, 35/1 gains, native caps, truthful 400/23 inference and iteration 4. Fusion stays closed; queued work and videos are preserved.')
AUTO=Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml')
a=tomllib.loads(AUTO.read_text());prefix=a['prompt'].split('Latest dated five-hour check',1)[0]
PREFIX_SHA='46f1711b1bce88727bc847e7985a95992e127272ea59cc70e4174507927817c3'
assert hashlib.sha256(prefix.encode()).hexdigest()==PREFIX_SHA and a['status']=='ACTIVE' and a['rrule']=='FREQ=HOURLY;INTERVAL=5'
assert a['target_thread_id']=='01a08b12-e85a-7f73-8677-315c708cd9f7'
args=dict(mode='update',id=a['id'],kind=a['kind'],name=a['name'],prompt=prefix+handoff,rrule=a['rrule'],status=a['status'],targetThreadId=a['target_thread_id'])
if 'notification_policy' in a:args['notificationPolicy']=a['notification_policy']
write(OUT/'automation_update_args.json',args)
# Canonical handoff helper must be reviewed for this check before publication.
from prepare_next_handoff import refresh_canonical
d=read(TRAIN/'automation/NEXT_CHECK.json')
d.update(updated_utc=STAMP,current_run=v['run'],diagnostic_report=str(OUT/'REPORT.md'),current_check=str(OUT),latest_evidence=summary,
 latest_gate_pair=pair,measured_since_previous=delta,current_retained_updates=v['actual_retained_main_updates'],inherited_updates=27507,
 throughput={k:v[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s']},next_priorities=priorities,latest_focus=handoff)
refresh_canonical(d,OUT,TRAIN,v,pair,summary,priorities,STAMP);write(TRAIN/'automation/NEXT_CHECK.json',d)
entry=f"\n\n<!-- {MARKER} -->\n## October 10, 01:49 UTC five-hour check\n\n{summary}\n\n{decision['conclusion']} {decision['interpretation']}\n\nTwo complete-state branches, two learning seeds and 64/128-update controls with all gates and diverse resets were tested; 512 diagnostic updates are excluded from main progress. Required initial, earlier, matched and closing upper timelines were inspected. The exact original main learner resumes after {resume['pause_seconds']:.1f} seconds. Since the previous snapshot it adds {delta['actual_main_samples']:,} actual samples / {delta['retained_main_updates']} retained updates. No production reward, gain, cap or optimizer reset. [Report]({OUT.as_posix()}/REPORT.md).\n\n"+'\n'.join(f'{i}. {p}' for i,p in enumerate(priorities,1))+'\n'
paths=[TRAIN/'ITERATION_HISTORY.md',ROBOT/'docs/robot/TRAINING_JOURNAL_20261001.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]
assert all(p.is_file() for p in paths)
for i,p in enumerate(paths):
 text=p.read_text(encoding='utf-8-sig')
 if MARKER not in text:p.write_text(text+entry if i<3 else entry+'\n'+text,encoding='utf-8')
write(OUT/'records_receipt.json',dict(recorded_utc=STAMP,records=[dict(path=str(p),sha256=sha(p)) for p in paths],report_sha256=sha(OUT/'REPORT.md'),preserved_durable_prefix_sha256=PREFIX_SHA,same_automation_id=a['id'],same_target_thread=a['target_thread_id'],same_five_hour_cadence=True,native_update_pending=True))
print(summary)
