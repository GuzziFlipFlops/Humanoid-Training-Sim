"""Validate preserved evidence and choose a safe current-full-state handoff."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
out=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
checks=[]
def check(name,value):
 assert value,name
 checks.append(name)
physical=read(out/'teacher_compatibility/result.json')
factor=read(out/'head_arm_factorial/teacher_compatibility/result.json')
v=read(out/'final_learning_verification.json');live=read(out/'final_cpu/live_handoff.json')
pair=read(out/'closing_gate_review.json')['latest']
check('one actual owned worker',len(live['actual_worker_pids'])==1)
check('same supervisor39000',v['supervisor_pid']==39000)
check('55 frozen hashes and53 protected authored worker files',v['frozen_files_verified']==55 and v['protected_authored_worker_files_equal']==53)
check('inherited27507 and50 received recoveries',v['inherited_actual_main_updates']==27507 and v['hash_proven_recovered_main_updates']==50)
check('all completed main load/KL/command/auxiliary checks',all(b['all_preupdate_KL_and_command_mismatch_zero'] and b['three_auxiliaries_batches_per_update']==20 and all(b['full_load_proof'][k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal']) for b in v['blocks']))
check('physical actor917 preserved across both experiments',physical['checkpoint_sha256']==factor['checkpoint_sha256']=='58ec1bf8f971433a3bd21e20cc82b5ad0f8d0b74aa27355a0ce76e4b3de0e746')
check('physical checkpoint bytes unchanged',hashlib.sha256(Path(physical['checkpoint']).read_bytes()).hexdigest()==physical['checkpoint_sha256'])
for label,d,expected in [('main',physical,1152),('head/arms split',factor,704)]:
 check(label+' completed',d['completed'] and not d['smoke_only'])
 check(label+' actual expected robot population',sum(g['robots'] for g in d['groups'])==expected)
 check(label+' all robots survive/no resets',all(g['survival']==1 and g['resets']==0 for g in d['groups']))
 check(label+' two seeds/32 individually generated states',len(d['seeds'])==2 and d['independent_initial_states_per_cohort']==32)
 check(label+' exact actor and native normal-limit physics',d['actor_loaded_exact'] and all(d[k] for k in ['original_gains','native_caps','gravity','free_base','self_collision','student_observations_truthful']))
 check(label+' zero observed/action command mismatch',d['maximum_action_command_difference']==0)
 check(label+' reconstructed complete reward terms',d['maximum_reward_term_sum_error']<1e-6)
 check(label+' exact effective physical target reconstruction',d['maximum_physical_target_reconstruction_error_rad']==0)
 check(label+' teacher not used outside fresh certificate',d['teacher_calls_outside_certificate']==0)
 check(label+' no production composite/support/replay/pose override',not any(d[k] for k in ['production_controller_deployed','hybrid_qualification_claimed','runtime_pose_writes','root_support','prescribed_body_action_replay']))
for folder in ['', 'head_arm_factorial/']:
 analysis=read(out/(folder+'teacher_compatibility_analysis.json'))
 check(folder+'physical population/paired-seed analysis',analysis['passed'])
 meta=read(out/(folder+'teacher_compatibility/teacher_metadata_proof.json'))
 check(folder+'strict original teacher SHA/current-runtime certificate',meta['complete_matching_nominal_pushed_all_prior_gates_verified'] and meta['exact_asset_and_observation_contract'] and not meta['generic_signature_bypass'] and meta['sha256']==physical['teacher_sha256'])
 check(folder+'teacher SHA bytes',hashlib.sha256(Path(meta['checkpoint']).read_bytes()).hexdigest()==meta['sha256'])
 motor=read(out/(folder+'current_substep_motor_attribution.json'))
 check(folder+'4000 actual native5ms motor calls',motor['passed'] and motor['actual_native_calls']==4000 and motor['sample_rate_Hz']==200)
 check(folder+'zero recorded DRIVING clips',all(g['driving_clip_fraction']==0 and g['longest_driving_clip_seconds']==0 for g in motor['groups']))
for folder,expected in [('teacher_smoke',72),('head_arm_factorial/teacher_smoke',44)]:
 d=read(out/folder/'result.json')
 check(folder+' completed small physical check',d['completed'] and d['smoke_only'] and sum(g['robots'] for g in d['groups'])==expected)
for name,count in [('controller_cpu_tests.json',13),('head_arm_factorial/CPU_tests.json',9)]:
 d=read(out/name);check(name,d['passed'] and d['count']==count)
d=read(out/'bounded_diagnosis/validation.json')
check('four genuine diagnostic PPO/full-load checks',d['passed'] and d['genuine_diagnostic_PPO_updates']==4 and d['actor_critic_Adam_exploration_exact'] and d['maximum_pre_update_KL']==0 and d['maximum_action_command_difference']==0 and d['auxiliaries_all_20_batches'] and d['diagnostic_actor_discarded'])
gradient=read(out/'current_gradient_analysis.json')
check('four actual captured PPO minibatches',gradient['completed'] and len(gradient['batches'])==4)
check('standing hook CPU subtraction reproduced',all(g['CPU_standing_subtraction_max_error']<1e-8 for g in gradient['batches']))
check('moving PPO mean derivative independently reproduced',all(g['moving_surrogate_derivative_cosine']>.9999 for g in gradient['batches']))
resumes=[]
for name in ['attempt01_signature_metadata/diagnostic_resume.json','diagnostic_resume.json','head_arm_factorial/diagnostic_resume.json']:
 d=read(out/name);resumes.append(d)
 check(name+' safely resumed same original supervisor',d['same_supervisor'] and d['supervisor_pid']==39000 and not d['training_source_changed'] and not d['diagnostic_actors_deployed'])
 for b in v['blocks']:
  if b['full_load_proof']['source_sha256']==d['checkpoint_sha256']:
   check(name+' exact retained production receiver',b['all_preupdate_KL_and_command_mismatch_zero'] and b['updates']>=64)
   break
 else:raise AssertionError('No exact retained receiver: '+name)
check('newer owned920 checkpoint preserved rather than physical917 rollback',factor['owned_resume_checkpoint_sha256']==resumes[-1]['checkpoint_sha256']==pair['sha256'])
for folder in ['initial_upper','final_upper']:
 d=read(out/folder/'audit_set.json')
 check(folder+' every plot explicitly visually inspected',d['all_plots_visually_inspected'])
 receipts=[read(p) for p in d['visual_QA_receipts']]
 check(folder+'14 image hashes match QA receipts',sum(len(r['plots']) for r in receipts)==14 and all(p['visually_inspected'] and hashlib.sha256(Path(p['path']).read_bytes()).hexdigest()==p['sha256'] for r in receipts for p in r['plots']))
check('diagnostic plots actually inspected',all(p['visually_inspected'] and hashlib.sha256(Path(p['path']).read_bytes()).hexdigest()==p['sha256'] for p in read(out/'diagnostic_plot_visual_QA.json')['plots']))
for name in ['closing_motor_nominal.json','closing_motor_pushed.json']:
 d=read(out/name);check(name+'correct interleaved case-label mapping',d['passed'] and d['case_label_fixture_passed'] and d['source_checkpoint_sha256']==pair['sha256'])
d=read(out/'closing_com_contact_audit.json')
check('closing920 synchronous COM/contact/action/joint telemetry',d['completed'] and d['sha256']==pair['sha256'] and all(t['finite_all_arrays'] and t['reset_ticks']==0 and t['observed_action_command_difference']==0 for t in d['tests']))
validation=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),passed=True,count=len(checks),checks=checks,
 physical_analysis_checks={'main':153,'head_arm_split':102},physical_actor=physical['checkpoint'],physical_sha256=physical['checkpoint_sha256'],
 owned_resume_actor=factor['owned_resume_checkpoint'],owned_resume_sha256=factor['owned_resume_checkpoint_sha256'],
 all_full_physical_robots_survived=True,new_64_or_128_candidate_learning_comparison=False,
 production_change=False,computer_on=True)
(out/'physical_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
tests=[
 'Preserve the exact current neutral-head full learner and passing current-speed pairs908/917,605,267 all-prior and separately certified349. Compare newer3.5 direction plus0.15 net travel,1.5 genuine/geometric flight and all earlier gates; reviewed920 also regresses2/2.5/3 lateral and nominal0.35 stance. New917 two-seed component tests show teacher legs with current arms overspeed0.15 and lose1.5 flight,while head changes also affect flight. This is closed-loop coordination,not mechanical isolation. Next physically test student body with qualified349 arms-only and CURRENT head under own truthful feedback,with phase-resolved effective leg targets/sole unloading/torso response on settled and transitioning histories. Whole-upper inverse only reached~0.9% flight; arms-only inverse is untested. No349 label above its3m certificate. Only select a command-local state-conditioned arm-row or coupled transfer after physical proof,exact-current-full-state64/128 controls,two learning seeds and all independently seeded prior/current/head/direction/flight gates. No blind repeat of focused-prior reweighting or rejected mean/gain/context/course candidates.',
 'Every check inspect actual head/shoulder pitch+roll/elbow angle,target,velocity,bilateral amplitude,arm-leg phase and noncancelling slow bias at every speed. Reviewed920 headRMS4.38deg and shoulder pitch ranges11.74/10.25deg; opposite held fast biases and2.5/3 slow reversals remain. The head/arms split shows arms explain most slow overspeed and larger1.5 flight loss,but head-only substitution yields1.478/1.502% and is not qualified. Do not deploy composites or infer pure mechanical versus learned arm-feedback cause. Preserve current upper20 batches/update,forward head,35/1 gains,legs,both teachers and full state. Primary Collins2009 Figure6 has published human shoulder/elbow time plots,but exact speed/axis/rest-pose/morphology mapping and numerical curves are unverified; mapped Unitree encoders and optical gaze remain unavailable. No fabricated reference curve,spoof,cap raise or from-scratch reset.',
 'Verify same39000 supervisor,one actual worker55 frozen hashes53 protected authored files,local ground,exact actor/critic/Adam/exploration and both teachers. Count27507 inherited plus fresh completed and50 received R40 recoveries;exclude32 unsavedR39 and all smoke/capacity/diagnostic samples. Exclude126.955s failed-metadata,476.254s compatibility and371.378s component-test suspensions only through each measurement endpoint;the canceled preboundary lookup attempt never paused training. Resumed920/923 load exact917/920 respectively with zeroKL/command mismatch and all3 auxiliaries20. Keep8192 target,reserve300,host guards/fallback and180continuous healthy TRAINING seconds;Fusion stays closed.917 actual4000 native5ms samples have zero recorded driving clips and braking overspeed;92010Hz ankle cap crossings cannot inherit that attribution. Only qualified5 or qualified4 with causal native limits blocking5 ends training and automation,with every unchanged gate and32 independently seeded states per command.'
]
decision=dict(recorded_utc=validation['recorded_utc'],deploy=False,production_change=False,continue_original_full_learner=True,
 endpoint_achieved=False,velocity_ceiling_proven=False,new_learning_comparison_claimed=False,
 explanation='Keep the same full R40 learner and existing upper fine-tuning. Current3.5 cases passed at908/917 but reviewed920 regresses direction and prior retention. Controlled teacher/current upper transfers do not supply a safe repair; no new production source,reward,gain,cap,actor or optimizer change is justified. Preserve useful learned state and exact passing actor evidence.',
 diagnostic_summary='Exact917:1152 robots20s compatibility plus704 robots20s head/arms split,each two seeds32 individually seeded paired states/cohort,all survive. Qualified349 gives~1.81% genuine1.5 flight; teacher body/current whole upper gives~0.94%,current arms-only~1.075%,current head-only1.478/1.502%. At0.15 current arms on teacher body/head reach~0.285m/s versus teacher~0.184;head-only stays~0.184. Context-conditioned whole-upper transfer remains unsafe. These are closed-loop component effects,not pure mechanical isolation. All diagnostic composites remain outside production. Four actual PPO captures show mixed prior/upper alignment. No new64/128 candidate learning test or qualification is claimed.',next_tests=tests)
(out/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
print(json.dumps(dict(passed=True,checks=len(checks),deploy=False,reviewed_actor=Path(pair['checkpoint']).parent.name)))
