"""Append final owned liveness without rewriting earlier exact-pair evidence."""
import json,hashlib,tomllib,re
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
live=read(OUT/'final_cpu/live_handoff.json');proof=read(OUT/'final_learning_verification.json');state=live['state']
assert proof['completed_main_updates']==128 and proof['hash_proven_recovered_main_updates']==41
assert all(live['frozen_source_integrity'].values()) and len(live['actual_worker_pids'])==1
qa=read(OUT/'plot_qa.json');assert len(qa['files'])==8
for p in qa['files']:assert p['visually_inspected'] and hashlib.sha256(Path(p['path']).read_bytes()).hexdigest()==p['sha256']
inventory=read(OUT/'final_inventory_stdout.json');assert inventory['differing_manifest_hashes']==0 and not inventory['read_warnings']
tag='closing_adaptive_check_20261007_0344'
last=proof['blocks'][-1]
note=f"""<!-- {tag} -->
Closing owned evidence {live['recorded_utc']}: R40 supervisor {state['supervisor_pid']}, one actual worker and all 55 frozen hashes verify. It has 128 completed-operation updates plus 41 hash-proven recovered updates = 169 retained new main updates, with 27,507 prior updates preserved. The second completed block loads actor/critic/Adam/exploration exactly; every retained pre-update KL and command mismatch is zero, with standing/moving/head-arm auxiliaries active at 20 batches/update. Its 6,144 environments have minimum free GPU {last['minimum_free_gpu_MiB']} MiB and host available {last['minimum_host_available_MiB']} MiB. Retained rates are {proof['active_PPO_samples_s']/1000:.2f}k active / {proof['worker_effective_samples_s']/1000:.2f}k worker-effective / {proof['whole_chain_samples_s']/1000:.2f}k whole-chain samples/s; the last rate includes new startup, capacity and baseline gates through the recorded last PPO tick.

The closing snapshot is evaluating newer actor013 in nominal014; no new qualification is inferred while its matching pair is incomplete in that snapshot. The completed actor010 nominal011/pushed012 remains the reported independent pair: 3.517/3.534 m/s signed, 0.433/0.405 m/s lateral; current direction fails. Retained0.15 net travel and2.5 hopping/lateral fail both, with pushed3.0 lateral also failed. Eight actual joint-angle/target/velocity plots have been inspected; slow shoulder-bias reversal at2.5/3 and fast asymmetry remain. Stronger shoulder gain is rejected, original35/1 and full learned state preserved. Native automation update is confirmed ACTIVE on the same five-hour heartbeat/thread/notification preference and durable5 or qualified4+causal native-limit endpoint. No pending recording request, no shutdown, no unsaved Fusion work touched. Inventory40 physical runs, zero differing hashes/read warnings; eight historical missing manifests remain documented.

Fresh liveness: {OUT/'final_cpu/live_handoff.json'}. Full-state evidence: {OUT/'final_learning_verification.json'}. Exact completed pair and plots: {OUT/'closing_gate_review.json'} and {OUT/'final_upper/upper_joint_time_audit.json'}. Native confirmation: {OUT/'automation_verification.json'}. Earlier observation counts above are retained with their timestamps.
<!-- /{tag} -->
"""
report=OUT/'REPORT.md';text=report.read_text(encoding='utf-8');assert tag not in text
report.write_text(text+'\n## Closing liveness and automation confirmation\n\n'+note,encoding='utf-8')
for p in [TRAIN/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    text=p.read_text(encoding='utf-8-sig');assert tag not in text;p.write_text(text+'\n\n'+note,encoding='utf-8')
for p in [Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md')]:
    text=p.read_text(encoding='utf-8-sig');assert tag not in text;p.write_text(note+'\n'+text,encoding='utf-8')
next_check=read(TRAIN/'automation/NEXT_CHECK.json')
next_check.update(recorded_utc=live['recorded_utc'],latest_observation=str(OUT/'final_cpu/live_handoff.json'),
    new_main_updates=128,recovered_main_updates=41,actual_retained_current_run_main_updates=169,
    throughput=proof,throughput_evidence=str(OUT/'final_learning_verification.json'),latest_verified_state=state,
    latest_saved_actor_unqualified=dict(checkpoint=state['checkpoint'],scope='Newer169-update actor; matching tests incomplete in final snapshot. Reported exact pair remains105-update010.'),
    automation_verification=str(OUT/'automation_verification.json'))
(TRAIN/'automation/NEXT_CHECK.json').write_text(json.dumps(next_check,indent=2)+'\n',encoding='utf-8')
evidence=read(OUT/'evidence.json');evidence.update(final_owned=str(OUT/'final_cpu/live_handoff.json'),final_learning_verification=str(OUT/'final_learning_verification.json'),actual_retained_current_run_main_updates=169,automation_verification=str(OUT/'automation_verification.json'),history_inventory=inventory)
(OUT/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
args=read(OUT/'automation_update_args.json')
args['prompt']+=f"""\nClosing owned snapshot {live['recorded_utc']}: same R40 supervisor {state['supervisor_pid']}, one actual worker, 55 hashes. It has 128 completed + 41 hash-proven recovered = 169 retained new main updates, preserving 27,507 inherited. Both teacher and upper auxiliaries stay at 20 batches/update; all completed full-state loads/KL/command checks pass. Second block at 6,144 has minimum free GPU {last['minimum_free_gpu_MiB']} MiB, host {last['minimum_host_available_MiB']} MiB. Retained rates {proof['active_PPO_samples_s']/1000:.2f}k active / {proof['worker_effective_samples_s']/1000:.2f}k worker / {proof['whole_chain_samples_s']/1000:.2f}k chain, including new startup/capacity/gates. That snapshot is evaluating newer actor013; its matching pair is incomplete there and no qualification is inferred. Read final_cpu/live_handoff.json and final_learning_verification.json; the independently reviewed/upper-plotted pair remains actor010/011/012. Read fresher actual pairs at the next check rather than assuming either is current. Inventory40 physical runs, zero differing hashes/read warnings, eight historical missing manifests documented. Durable focus/cadence/endpoint unchanged.\n"""
(OUT/'automation_update_args.json').write_text(json.dumps(args,indent=2)+'\n',encoding='utf-8')
native=Path('C:/Users/rshan/.codex/automations/humanoid-training-research-and-repair-every-four-hours/automation.toml')
(OUT/'automation_after_first_update.toml').write_bytes(native.read_bytes())
receipt=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),owned_snapshot_utc=live['recorded_utc'],status=state['status'],
    retained_new_updates=169,hashes=55,one_actual_worker=True,plots_inspected=8,
    inherited27507_preserved=True,diagnostic_actor_gain_deployed=False,history_inventory=inventory,
    native_final_focus_update_pending=True)
(OUT/'final_verification.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))
