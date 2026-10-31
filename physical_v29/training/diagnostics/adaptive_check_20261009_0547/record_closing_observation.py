"""Add the newer saved pair once; retain the earlier full upper/physical scope."""
import json
from pathlib import Path
out=Path(__file__).resolve().parent;train=out.parents[1];root=train.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,d:Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
v=read(out/'liveness_learning_verification.json');pair=read(out/'liveness_gate_review.json')['latest'];last=v['blocks'][-1]
assert Path(pair['checkpoint']).parent.name=='923_run_350_train_305'
summary=(f"Closing owned observation {v['owned_snapshot_utc']}: the same R40 supervisor 39000 has one actual worker and 55 verified hashes. "
 f"It retains {v['completed_main_updates']:,} completed + 50 recovered = {v['actual_retained_main_updates']:,} current-run main updates, plus 27,507 inherited. "
 "Newer exact actor 923, matching nominal 924/pushed 925, reaches 3.527/3.532 m/s signed at command 3.5 with 0.180/0.165 m/s lateral drift; both direction gates still fail. "
 "Both 2/2.5/3 m/s cases recover all checks; preserve that improvement separately from 920. Both 0.15 net-travel, 0.35 bilateral-stance, 0.65 speed and 1.5 flight/geometric-flight tests fail. All robots survive. "
 "Full all-command upper timeline QA remains bound to 920, while native 5 ms teacher/component diagnostics remain bound to 917; no newer posture or motor qualification is inferred. "
 "Completed block 926 has 64 genuine updates at 8192, loads exact 923 actor/critic/Adam/exploration, and has zero pre-update KL/command mismatch with all three auxiliaries at 20 batches/update. "
 f"Its minimum free GPU/host memory is {last['minimum_free_gpu_MiB']}/{last['minimum_host_available_MiB']} MiB, private growth {last['private_memory_growth_MiB']} MiB and Torch allocation growth {last['torch_allocation_growth_MiB']} MiB. "
 f"Retained rates are {v['active_PPO_samples_s']/1000:.2f}k active / {v['worker_effective_samples_s']/1000:.2f}k worker-effective / {v['whole_chain_samples_s']/1000:.2f}k whole-chain samples/s. "
 "No production source, reward, gain, cap, actor or optimizer reset and no diagnostic controller deployed. Same active five-hour endpoint; computer on. Read closing_newer_pair.json, closeout_verification.json and fresh ACTIVE_RUN/state.")
marker='<!-- /adaptive_check_20261009_0547_record -->'
files=[train/'ITERATION_HISTORY.md',root/'docs/robot/TRAINING_JOURNAL_20261001.md',Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]
for p in files:
 text=p.read_text(encoding='utf-8');assert text.count(marker)==1 and summary not in text
 p.write_text(text.replace(marker,summary+'\n'+marker),encoding='utf-8')
p=out/'REPORT.md';text=p.read_text(encoding='utf-8');assert '## Closing newer observation' not in text
p.write_text(text+'\n## Closing newer observation\n\n'+summary+'\n\n[Newer exact gate pair](closing_newer_pair.json), [fresh recovery evidence](liveness_learning_verification.json), [automation verification](automation_verification.json) and [CPU inventory](inventory_verification.json).\n',encoding='utf-8')
e=read(out/'evidence.json');e['closing_summary']=summary;write(out/'evidence.json',e)
h=read(train/'automation/NEXT_CHECK.json');d=read(out/'decision.json');args=read(out/'automation_update_args.json')
old=d['next_tests'][0]
new=old.replace('Reviewed 920 also fails 2/2.5/3 m/s lateral direction and nominal 0.35 m/s bilateral stance.',
 'Reviewed 920 failed 2/2.5/3 m/s lateral direction; newer 923 recovers all three in both tests. Both 0.35 m/s stance and 0.65 m/s speed now fail, alongside the persistent 0.15 path and 1.5 flight failures. Preserve that actor-specific recovery and audit these fresh slow cases too.')
assert new!=old
d['next_tests'][0]=new;write(out/'decision.json',d)
for k in ['next_tests','focused_tests','focus']:h[k][0]=new
h.update(closing_summary=summary,summary=h['summary']+' '+summary,reviewed_matching_pair=h['latest_matching_pair'],latest_matching_pair=pair,
 retained_main_throughput={k:v[k] for k in ['active_PPO_samples_s','worker_effective_samples_s','whole_chain_samples_s','documented_diagnostic_pause_seconds_excluded']},
 resources=dict(last_completed_main=last,target_environments=8192,reserve_MiB=300,healthy_training_ramp_seconds=180,evaluation_memory_is_not_capacity=True))
write(train/'automation/NEXT_CHECK.json',h)
args['prompt']=args['prompt'].replace(old,new)+'\n\n'+summary
write(out/'automation_update_args.json',args)
write(out/'closing_summary.json',dict(summary=summary,physical_actor='917',mandatory_upper_actor='920',newer_gate_actor='923'))
print('Recorded newer923 recovery and remaining failures; updated same heartbeat arguments.')
