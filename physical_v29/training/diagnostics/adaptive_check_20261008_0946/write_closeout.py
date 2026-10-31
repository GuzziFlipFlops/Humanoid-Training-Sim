"""Add newer saved-gate evidence while preserving the 605 diagnosis and source lineage."""
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
TRAIN = OUT.parents[1]
ROOT = TRAIN.parents[1]
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p, d):
    Path(p).write_text(json.dumps(d, indent=2) + '\n', encoding='utf-8')

v = read(OUT/'closeout_learning_verification.json')
live = read(OUT/'closeout_cpu/live_handoff.json')
pair = read(OUT/'closeout_gate_review.json')['latest']
upper = read(OUT/'closing_newer_upper/upper_joint_time_audit.json')
u = next(r for r in upper['records'] if r['actor_tag']=='latest' and r['test']=='nominal' and r['command_m_s']==3.5)
assert Path(pair['checkpoint']).parent.name == '608_run_350_train_200'
assert Path(u['checkpoint']) == Path(pair['checkpoint']) and u['sha256'] == pair['sha256']
assert live['state']['status']=='running' and live['state']['supervisor_pid']==39000
assert len(live['actual_worker_pids'])==1 and len(live['frozen_source_integrity'])==55 and all(live['frozen_source_integrity'].values())
assert (OUT/'closing_newer_upper/plot_inspection.json').exists() or (OUT/'closing_newer_upper/plot_qa.json').exists() or any((OUT/'closing_newer_upper').glob('*inspection*.json'))
a, b = pair['nominal_cases'][-1], pair['pushed_cases'][-1]
assert not a['passed'] and not b['passed']
marker = 'Closing saved-gate evidence — actor 608'
report_path = OUT/'REPORT.md'
report = report_path.read_text(encoding='utf-8')
assert marker not in report
summary = (
    f"Closing owned evidence {v['owned_snapshot_utc']}: same R40 supervisor 39000, one actual worker and 55 hashes. "
    f"It retains {v['completed_main_updates']:,} completed + 50 recovered = {v['actual_retained_main_updates']:,} current-run updates, plus 27,507 inherited. "
    f"Newer exact actor 608, matching nominal 609/pushed 610, reaches {a['signed_speed']:.3f}/{b['signed_speed']:.3f} m/s at command 3.5, "
    f"but lateral {a['lateral_speed']:.3f}/{b['lateral_speed']:.3f} fails both direction tests. Preserve actor 605's separately passing current-speed pair; improvement is nonmonotonic. "
    "Actor 608 recovers 2.5 slip in both tests; 0.15 net travel, 1.5 flight and 3.0 lateral still fail. All survive. "
    f"Actor-608 upper angle/target/velocity/bias plots were also inspected: head RMS {u['head_rms_deg']:.2f} degrees, "
    "shoulder pitch ranges 9.77/6.87 degrees, arm/hip phases 161/155 degrees. Held biases, slow 2.5/3 m/s reversals and fast asymmetry persist; natural motion remains uncertified. "
    f"Latest completed block 611 has 64 genuine 6144-environment updates, exact full-state load, zero KL/command mismatch and all three auxiliaries at 20 batches/update. "
    f"Minimum free GPU 1923 MiB and host 8391 MiB, private growth 9 MiB, Torch allocation unchanged. "
    f"Retained rates {v['active_PPO_samples_s']/1000:.2f}k active / {v['worker_effective_samples_s']/1000:.2f}k worker-effective / {v['whole_chain_samples_s']/1000:.2f}k whole-chain samples/s. "
    "The 605 passive state/gradient and motor diagnoses remain actor-specific. No production change or diagnostic actor deployed; same active five-hour goal, computer on."
)
report_path.write_text(report + f'\n## {marker}\n\n{summary}\n\nRead `closeout_gate_review.json`, `closeout_learning_verification.json` and `closing_newer_upper` for the newer pair. The owned operation at this snapshot is `{Path(v["operation_path"]).name}`; its newer actor has an incomplete matching pair and receives no qualification here. The main report\'s physical replay, state substitutions, individual cycles and motor audit stay explicitly tied to actor 605.\n', encoding='utf-8')
for p in [TRAIN/'ITERATION_HISTORY.md', ROOT/'docs/robot/TRAINING_JOURNAL_20261001.md', Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    with p.open('a', encoding='utf-8') as f:
        f.write(f'\nClosing evidence for the October 8 09:46 UTC check:\n\n{summary}\n\nEvidence: {OUT/"closeout_gate_review.json"}; {OUT/"closing_newer_upper"}.\n')
for p in [Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'), Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]:
    s=p.read_text(encoding='utf-8'); end='<!-- /adaptive_check_20261008_0946_record -->'
    assert end in s
    p.write_text(s.replace(end, '\n'+summary+'\n'+end, 1), encoding='utf-8')
e = read(OUT/'evidence.json'); e['closing_summary']=summary; e['closing_exact_pair']=str(OUT/'closeout_gate_review.json'); e['closing_upper']=str(OUT/'closing_newer_upper'); write(OUT/'evidence.json', e)
npath=TRAIN/'automation/NEXT_CHECK.json'; n=read(npath)
n['preserved_improving_actor']=n['selected_actor']; n['preserved_improving_actor_sha256']=n['selected_actor_sha256']
n.update(latest_observation=v['owned_snapshot_utc'], latest_liveness_utc=v['owned_snapshot_utc'], latest_liveness=str(OUT/'closeout_cpu/live_handoff.json'),
    resumed_learning_evidence=str(OUT/'closeout_learning_verification.json'), completed_main_updates=v['completed_main_updates'], new_main_updates=v['completed_main_updates'],
    actual_retained_current_run_main_updates=v['actual_retained_main_updates'], partial_observed_main_updates=v['observed_partial_main_updates'],
    current_owned_operation=v['operation_path'], selected_actor=pair['checkpoint'], selected_actor_sha256=pair['sha256'], latest_matching_pair=pair,
    reviewed_speed_case_passes_both=False, closing_summary=summary, upper_joint_time_series=str(OUT/'closing_newer_upper/upper_joint_time_audit.json'),
    mandatory_upper_audit=str(OUT/'closing_newer_upper/upper_joint_time_audit.json'), mandatory_upper_bias_audit=str(OUT/'closing_newer_upper/upper_bias_audit.json'),
    upper_target_bias_audit=str(OUT/'closing_newer_upper/upper_target_bias_audit.json'),
    retained_main_throughput={k:v[k] for k in ('active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s','documented_diagnostic_pause_seconds_excluded')})
n['retained_main_totals']={k:sum(b[k] for b in v['blocks']) for k in ('updates','samples','active_seconds','worker_seconds')}
n['diagnostic_actor_605_upper']=str(OUT/'final_upper')
n['focus'][0] += ' Newer actor 608 recovers 2.5 but loses 3.0/3.5 direction; preserve both exact pairs and do not claim monotonic progress.'
n['next_tests']=n['focused_tests']=n['focus']
write(npath,n)
aargs=read(OUT/'automation_update_args.json')
aargs['prompt'] += '\n\n'+summary+'\nRead closeout_gate_review.json, closing_newer_upper, closeout_learning_verification.json and fresh ACTIVE_RUN/state before assuming the dated stage. Preserve both actor 605\'s useful passing current-speed pair and actor 608\'s recovered 2.5 case. All durable priorities and endpoint rules remain.'
write(OUT/'automation_update_args.json',aargs)
write(OUT/'closeout_verification.json',dict(recorded_utc=datetime.now(timezone.utc).isoformat(),owned_snapshot=v['owned_snapshot_utc'],one_actual_worker=True,frozen_hashes_verified=55,
    latest_saved_pair=pair['checkpoint'],diagnostic_actor_605_preserved=True,upper_plots_inspected=True,production_change=False,summary=summary))
print(summary)
