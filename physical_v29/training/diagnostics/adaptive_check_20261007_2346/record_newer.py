"""Bind closing gates/plots to the newly saved431,keeping428 diagnosis."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1];ROOT=TRAIN.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,d:Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
r=read(OUT/'post_resume_gate_review.json');pair=r['latest'];v=read(OUT/'post_resume_learning_verification.json')
assert Path(pair['checkpoint']).parent.name=='431_run_350_train_141'
assert all(c['survival']==1 and c['trials']==32 for k in ['nominal_cases','pushed_cases'] for c in pair[k])
upper=read(OUT/'newer_upper/upper_joint_time_audit.json');u=next(x for x in upper['records'] if x['actor_tag']=='latest' and x['test']=='nominal' and x['command_m_s']==3.5)
write(OUT/'newer_matching_pair.json',pair)
qa=read(OUT/'plot_qa.json');qa['files'] += [dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (OUT/'newer_upper').glob('*.png')]
qa['findings']+=' New431:opposite held3.5 pitch,slower2.5/3 reversal,fast asymmetry/roll/elbow coordination and gait-linked head oscillation persist;four newer plots inspected.'
write(OUT/'plot_qa.json',qa)
summary=(f"Closing evidence {v['owned_snapshot_utc']}:same R40/39000,one actual worker and55 hashes. "
 f"{v['completed_main_updates']} completed+50 hash-proven recovered={v['actual_retained_main_updates']} retained current-run,+27507 inherited;{v['observed_partial_main_updates']} partial434 updates separate. "
 "Newest exact431 nominal432/pushed433 has3.5 signed3.443/3.452m/s,lateral0.270/0.242,improved from4280.351/0.326 but still fails world-lateral. Retained1.5 flight improves0.444/0.542% to1.187/1.208%,still below1.5%;0.15 signed travel worsens to0.058/0.062m/s and fails nettravel. All survive;every other saved case including3m passes. Preserve this useful nonmonotonic progress,no new qualification. "
 f"New431 headRMS{u['head_rms_deg']:.2f}deg,shoulder pitch ranges12.46/11.27deg and phases173/-145deg;angle/target/velocity/slow-bias plots inspected. Opposite held3.5 biases and2.5/3 slow reversals persist,no human/Unitree/optical-gaze certification. "
 "Same full learner retains all state,upper and both priors20batches/update. Resumed43164 real6144 updates has exact428 full load/zeroKL/command mismatch,minGPU1922MiB/host7191MiB,private+6MiB,Torch allocation unchanged. "
 f"Completed-main rates{v['active_PPO_samples_s']/1000:.2f}k active/{v['worker_effective_samples_s']/1000:.2f}k worker/{v['whole_chain_samples_s']/1000:.2f}k chain,6358.9s documented pauses excluded. "
 "No source/reward/gain/cap/optimizer/actor reset;same ACTIVE5h scope.5ms parity/filter experiments remain428-specific. Preserve improving gait while testing focused upper mean-bias/control and retention;do not restart merely from dated plateau.")
header=f"## Closing matching pair — exact431\n\n{summary}\n\nSee `newer_matching_pair.json`,`newer_upper` and `post_resume_learning_verification.json`. Earlier428 analysis below is preserved and is not attributed to431.\n\n"
report=OUT/'REPORT.md';old=report.read_text();first,rest=old.split('\n',1);report.write_text(first+'\n\n'+header+rest,encoding='utf-8')
with report.open('a',encoding='utf-8') as h:h.write('\nDiagnostic accounting correction:early closeout counted theWindowsvenv launcher and its actual Python child as two workers. Canonical leaf-PID accounting verifies one actualIsaac/CUDA worker;both failed assertion logs/source remain. No extra simulator was started or production fault inferred. Inventory summary is read from inventory.log rather than the unrelatedartifact list namedchecks.\n')
for f in [TRAIN/'ITERATION_HISTORY.md',ROOT/'docs/robot/TRAINING_JOURNAL_20261001.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    with f.open('a',encoding='utf-8') as h:h.write('\n\n### Closing431 pair — October7 23:46 check\n\n'+summary+'\nReport: '+str(report)+'\n')
for f in [Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]:
    text=f.read_text(encoding='utf-8');marker='adaptive_check_20261007_2346_closing';assert marker not in text
    f.write_text(f'<!-- {marker} -->\n{summary} Read {report},then freshACTIVE_RUN/state. Earlier entries preserved below.\n<!-- /{marker} -->\n\n'+text,encoding='utf-8')
n=read(TRAIN/'automation/NEXT_CHECK.json');n.update(recorded_utc=datetime.now(timezone.utc).isoformat(),checkpoint=pair['checkpoint'],checkpoint_sha256=pair['sha256'],selected_actor=pair['checkpoint'],
    newer_matching_pair=pair,latest_closing_gate_pair=pair,latest_gate_trends={k:pair[k] for k in ['nominal_cases','pushed_cases']},
    latest_liveness=str(OUT/'post_resume_cpu/live_handoff.json'),latest_observation=v['owned_snapshot_utc'],resumed_learning_evidence=str(OUT/'post_resume_learning_verification.json'),
    mandatory_upper_audit=str(OUT/'newer_upper/upper_joint_time_audit.json'),mandatory_upper_bias_audit=str(OUT/'newer_upper/upper_bias_audit.json'),
    upper_joint_time_audit=str(OUT/'newer_upper/upper_joint_time_audit.json'),upper_joint_time_series=str(OUT/'newer_upper/upper_joint_time_audit.json'),upper_body_analysis=str(OUT/'newer_upper/upper_joint_time_audit.json'),
    upper_target_bias_audit=str(OUT/'newer_upper/upper_target_bias_audit.json'),observed_partial_main_updates=v['observed_partial_main_updates'],
    closing_owned_observation=str(OUT/'post_resume_cpu/live_handoff.json'),closing_learning_verification=str(OUT/'post_resume_learning_verification.json'),
    current_owned_operation=v['operation_path'],newer_observation_scope='Newest reviewed matching and upper actor431;physics/timing/COM audits428-specific;current434 partial updates separately unqualified.',
    throughput={k:v[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s','documented_diagnostic_pause_seconds_excluded']},throughput_evidence=str(OUT/'post_resume_learning_verification.json'))
for key in ['focused_tests','next_tests','focus']:
    n[key][1]='Preserve newest431 directional/flight improvement before another repair.3.5 lateral0.270/0.242 improves from428,but still fails;1.5 flight1.187/1.208% improves yet remainsbelow1.5%;0.15 travel0.058/0.062 worsens. All other cases pass. '+n[key][1]
write(TRAIN/'automation/NEXT_CHECK.json',n)
d=read(OUT/'decision.json');d.update(closing_evidence=summary,newest_reviewed_actor=pair,closing_upper=str(OUT/'newer_upper/upper_joint_time_audit.json'))
write(OUT/'decision.json',d)
args=read(OUT/'automation_update_args.json');args['prompt']+='\n\n'+summary+' Read newer_matching_pair.json,newer_upper and post_resume_learning_verification.json. Earlier428 diagnosis stays separate;keep the full learner and compare newer complete pairs before redesigning.'
write(OUT/'automation_update_args.json',args)
write(OUT/'newer_closeout.json',dict(recorded_utc=datetime.now(timezone.utc).isoformat(),summary=summary,checkpoint=pair['checkpoint'],sha256=pair['sha256'],plots_inspected=True,full_state_preserved=True,qualification=False))
print(summary)
