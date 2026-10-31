"""Reconcile the newest exact gates; never infer passes from earlier actors."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,v:Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
pair=read(OUT/'closing_matching_gates.json');live=read(OUT/'closing_owned_observation.json');s=live['state'];run=Path(live['active_run']['run'])
failed=[c for c in pair['cases'][:-1] if not(c['nominal_pass'] and c['pushed_pass'])]
passing=[c['command'] for c in pair['cases'][:-1] if c['nominal_pass'] and c['pushed_pass']]
assert failed and {c['command'] for c in failed}=={.35,.65}
last_good=run/'056_run_150_train_016/candidate.pt'
for folder in ['057_run_150_gate_017_nominal','058_run_150_gate_017_push']:
    ev=read(run/folder/'evaluation.json');cfg=read(run/folder/'run_config.json')
    assert Path(cfg['arguments']['checkpoint']).resolve()==last_good.resolve()
    assert all(c['passed'] for c in ev['cases'][:-1])
good=dict(checkpoint=str(last_good),sha256=hashlib.sha256(last_good.read_bytes()).hexdigest(),
          nominal=str(run/'057_run_150_gate_017_nominal'),pushed=str(run/'058_run_150_gate_017_push'),
          all_prior_cases_pass=True,running_pass=False)
write(OUT/'last_all_prior_passing_actor.json',good)
focused=[
 'Prioritize reproduced slow-command regression: checkpoint059 nominal0.35 and pushed0.65 fail speed, after checkpoint056 passed every prior case. Audit per-command conditioned-teacher labels and sample selection, effective clipped target error, coverage, teacher versus PPO/shared-trunk gradient magnitudes and tracking-kernel sensitivity. Compare a narrow training-only continuation on the current useful full state against unchanged control at64/128 updates and all matched prior/head/running gates before deploying. Do not roll back posture, spoof production observations or reset the whole learner.',
 'Resolve running target dead zones and pre-action/reward phase versus measured joint/sole response delay. Test feasible normal-limit timing and full objective return. All nine pulse variants remained grounded and lowered reward; local ankle derivatives do not establish the cause or a motor ceiling. Only deploy a constraint-aware exploration/target candidate after a bounded matched-control learning comparison retains prior skills and improves independent flight.',
 'Compare fresh matching stopped/prior/current speed, direction, head/arm and genuine-flight gates. Preserve checkpoint056 and the qualified walking/earlier flight actors. Verify8192 memory,300MiB reserve, exact full-state fallback and180healthy-second ramps. Keep the rejected running-row transfer and44% exposure variants rejected absent new evidence; do not spend the next check only waiting.'
]
decision=read(OUT/'decision.json');decision.update(closing_recorded_utc=live['recorded_utc'],
    closing_retention_regression=True,closing_failed_prior_commands=[.35,.65],
    last_all_prior_passing_actor=good,next_tests=focused,
    closing_reason='Newest exact1088-update matching pair regresses0.35 nominal and0.65 pushed speed; running remains grounded. Preserve the independently passing1024-update actor and current neutral-head state. This new failure makes per-command teacher/objective diagnosis the next priority; no untested repair is deployed.')
write(OUT/'decision.json',decision)
e=read(OUT/'evidence.json');e.update(closing_owned_observation=str(OUT/'closing_owned_observation.json'),
    closing_matching_gates=pair,closing_retention_regression=True,last_all_prior_passing_actor=good,decision=decision,
    closing_diagnostic_errata='Initial closing prose wrongly generalized prior passes to a newer actor. Raw closing_matching_gates always records the actual failures. This note corrects that conclusion. A live filesystem selection caught an unmatched nominal/pushed race; selecting completed checkpoint-bound pairs fixes it. Missing PPO progress during startup/evaluation uses explicitly labeled last-completed progress, not fabricated current PPO updates.')
write(OUT/'evidence.json',e)
rc=pair['cases'][-1];head=pair['running_head_RMS_deg']
text=f'''\n\n## Closing exact-gate correction: {live['recorded_utc']}\n\nThe latest matching nominal060/pushed061 pair on the1088-update checkpoint059 does **not** retain every earlier speed. Stopped,0.15 and1.0 pass both. Nominal0.35 fails speed at{pair['cases'][2]['nominal_speed']:.3f}m/s, and pushed0.65 fails speed at{pair['cases'][3]['pushed_speed']:.3f}m/s. Their counterpart tests pass. All robots survive. The1.5 command gives{rc['nominal_speed']:.3f}/{rc['pushed_speed']:.3f}m/s with0%/0% true flight; head RMS{head[0]:.2f}/{head[1]:.2f}degrees. The previous1024-update checkpoint056 passed all earlier commands in matching057/058 and is preserved with its exact SHA/gates. These qualifications belong to056, not059. [Latest exact failures](closing_matching_gates.json), [last prior-passing actor](last_all_prior_passing_actor.json).\n\nThe same R36 learner remains active with one actual CUDA worker and49 hashes verified. The owned snapshot caught062 startup with no PPO progress yet; the63 iteration counter quoted in the first closing metadata was from completed059, not062 learning. Snapshot free GPU5591MiB is startup capacity, not steady8192 training memory. Completed-main memory/throughput and exact resumed learning are recorded separately above. No sources, weights, physical caps, actor/critic/Adam/exploration or inference changed. The earlier closing prose assumed earlier-speed passes and is explicitly corrected here; raw gate results were preserved throughout. Failed diagnostic closing selection/progress reads are preserved in tool evidence and fixed by checkpoint-bound completed-pair selection and labeled fallback progress.\n\nNext priority is per-command conditioned-target/gradient and tracking diagnosis, alongside feasible take-off timing. A new candidate requires an unchanged full-state control,64/128-update comparison and all matching gates before deployment. Preserve current posture,056, the qualified walking teacher and038 flight actor. Keep the same adaptive three-hour check active, target8192/reserve300/180healthy ramp and computer-on endpoint. Running,5m/s and an actuator ceiling remain unqualified.\n'''
# Human-readable spacing in generated numeric prose; explicit substitutions
# preserve identifiers, decimals, hashes and referenced paths.
for a,b in [('nominal060/pushed061','nominal 060/pushed 061'),('the1088','the 1088'),('checkpoint059','checkpoint 059'),
 ('Stopped,0.15','Stopped, 0.15'),('and1.0','and 1.0'),('Nominal0.35','Nominal 0.35'),('pushed0.65','pushed 0.65'),
 ('The1.5','The 1.5'),('with0%','with 0%'),('degrees.',' degrees.'),('previous1024','previous 1024'),
 ('checkpoint056','checkpoint 056'),('matching057/058','matching 057/058'),('to056, not059','to 056, not 059'),
 ('and49','and 49'),('GPU5591MiB','GPU 5,591 MiB'),('steady8192','steady 8,192'),('control,64/128','control, 64/128'),
 ('posture,056','posture, 056'),('and038','and 038'),('Running,5m/s','Running, 5 m/s')]:text=text.replace(a,b)
report=(OUT/'REPORT.md').read_text(encoding='utf-8').split('\n\n## Closing owned observation',1)[0]
(OUT/'REPORT.md').write_text(report+text,encoding='utf-8')
for p in [HERE/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    with p.open('a',encoding='utf-8') as stream:stream.write(text)
n=read(HERE/'automation/NEXT_CHECK.json');n.update(focused_tests=focused,latest_repair_decision=decision,
    last_all_prior_passing_actor=good,closing_failed_prior_commands=[.35,.65],running_qualified=False)
n['selected_actor']['standing_and_walking_cases_through_1_m_s_pass']=False
n['latest_verified_state']['progress_scope']='Owned062 startup has no saved PPO progress yet; earlier iteration63 is last completed059 progress only.'
write(HERE/'automation/NEXT_CHECK.json',n)
brief=f'''<!-- adaptive_check_20261005_1142_closing -->\n## Newer R36 retention regression: October 5, 08:17 EDT\n\nRead {str(OUT/'REPORT.md').replace(chr(92),'/')} and fresh ACTIVE_RUN/state. Exact 1088-update checkpoint 059 matching 060/061 fails nominal 0.35 and pushed 0.65 speed retention; stopped, 0.15 and 1.0 still pass both. Running 1.5 travels 1.355/1.359 m/s with zero true flight. Head RMS 4.02/4.04 degrees. The 1024-update checkpoint 056 with passing prior gates 057/058 is preserved; its passes do not qualify 059. Same full-state learner remains active; no untested runtime repair deployed. Next per-command teacher-target/gradient and effective target diagnosis against unchanged control, plus phase/take-off objective analysis. Latest startup has no saved PPO progress yet; use labeled completed-block rates/memory rather than startup free memory. One worker/49 hashes, same adaptive three-hour cadence, 8,192 target/300 MiB reserve/180-second healthy ramp and computer-on endpoint.\n\n'''
for p in [HERE/'automation/THREE_HOUR_CHECK.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md'),Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md')]:
    old=p.read_text(encoding='utf-8-sig')
    if '<!-- adaptive_check_20261005_1142_closing -->' not in old:p.write_text(brief+old,encoding='utf-8')
args=read(OUT/'automation_update_args.json')
prefix=args['prompt'].split('Latest dated focus',1)[0].rstrip()
focus=f'''Latest dated focus, October 5, {live['recorded_utc']}: same R36_posture_conditioned_priors full-state learner and supervisor PID 19116. Read diagnostics/adaptive_check_20261005_1142/REPORT.md, decision.json, evidence.json, closing_matching_gates.json, last_all_prior_passing_actor.json, physical_objective_analysis.json, reward_phase_audit.json, motor_speed_audit.json, phase_objective.png and resumed_learning_verification.json. Read fresh ACTIVE_RUN/state; dated observations are not current qualification. No production source, reward, weight, physics, cap or optimizer reset was deployed this check.

Latest 1,088-update checkpoint 059 matching full nominal 060/pushed 061 tests regress prior speed retention: nominal 0.35 fails speed at 0.448 m/s and pushed 0.65 fails at 0.745 m/s. The counterpart tests pass; stopped, 0.15 and 1.0 pass both. All 192 robots survive, 32 trials per command for 20 seconds. Running at 1.5 gives 1.355/1.359 signed m/s and zero true flight; head RMS is 4.02/4.04 degrees, natural arms uncertified. The independently passing 1,024-update checkpoint 056 is preserved with matching 057/058 gates. Its earlier-speed qualifications must not be attributed to 059. Prior tracking is nonmonotonic and has now regressed; do not describe the newest actor as retaining every speed.

Controlled diagnosis: supervisor suspended at the exact 896-update full checkpoint 050, worker drained, twelve cohorts of 24 robots for 20 seconds. Nine smooth bilateral retraction/timing/preceding extension variants all survive but none achieves real flight; every running intervention lowers full mean reward. Grounded force gaps are not flight. Truthful observations, free base/gravity/self-collision and native motor constraints remain. No pulses, support, observation rewrite, hybrid or diagnostic actor is deployed. Failure of these particular pulses is not proof that physical running is impossible or every natural run has lower reward.

Exact CPU target replay on 14,400 states differs by 2.98e-8 rad; no sampled coupled-CAD target rejection. Local ankle-action target derivatives are zero in 79–88% of running-control samples from existing action/joint/slew constraints; walking also has dead zones. These hold previous targets fixed and are not PPO gradients or a motor ceiling. Sixteen CPU counterexamples reproduce v8 summed-load credit, term sums, monotone fixed-state unloading and load-redistribution invariance. Use term_running_unload, not the legacy averaging helper mean_unload_score. Pre-action and reward phases are separate; resolve target/joint/sole timing before a repair. Exact 50 Hz CAD Jacobian/reduction/unit checks pass, no sustained relevant motor ceiling proven, 5 ms substeps untested. Primary research lead: https://arxiv.org/html/2011.01387v2, smoothed periodic force/velocity composition; no new copied weights or convergence guarantee.

Same supervisor resumed after 431.7 seconds. Resumed block 053 completed 64 genuine updates at 8,192 with exact actor, critic, Adam and exploration equality, zero pre-update KL/command mismatch and stable memory. At the 960-update saved rate measurement, active/worker-effective/whole-chain throughput was about 46.68k/40.39k/25.88k samples/s excluding pause. Minimum main GPU 479 MiB and host 5,457 MiB; resumed block minimum GPU 540 MiB, host 5,584 MiB, private growth 9 MiB, Torch allocation unchanged. Eval 051 wrapper includes the pause. One worker and 49 hashes verify. Latest 062 startup has no PPO progress yet: the iteration 63 fallback is from completed 059, and 5,591 MiB free GPU at startup is not steady 8,192 memory. Target 8,192, reserve 300 MiB, host fallback and 180 healthy-training-second ramps remain. Existing full-body recordings are fulfilled.

Next priorities: (1) {focused[0]} (2) {focused[1]} (3) {focused[2]} Do not spend the next check merely waiting or claim a repair from local derivatives alone. Preserve the useful current forward-head state and last independently passing actors. Keep this same adaptive three-hour heartbeat active until verified actual 5 m/s or a causally established relevant actuator ceiling, then save, pause and brief with the computer on.'''
focus=focus.replace('checkpoint059','checkpoint 059').replace('checkpoint056','checkpoint 056').replace('nominal0.35','nominal 0.35').replace('pushed0.65','pushed 0.65').replace('at64/128','at 64/128').replace('Verify8192','Verify 8,192').replace('300MiB','300 MiB').replace('and180healthy','and 180 healthy').replace('and44%','and 44%')
args['prompt']=prefix+'\n\n'+focus
write(OUT/'automation_update_args.json',args)
print(json.dumps(dict(corrected=True,latest_main=s['total_training_iterations'],failed_prior=[c['command'] for c in failed],last_good=good,running_qualified=False),indent=2))
