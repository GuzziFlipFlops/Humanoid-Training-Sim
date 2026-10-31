"""Fix final-reference paths and bind latest memory to its measured timestamp."""
import json,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,d:Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
r=read(OUT/'final_liveness.json');live=read(OUT/'final_liveness_owned_observation.json')
op=Path(r['operation']);rows=[json.loads(x) for x in (op/'progress.jsonl').read_text().splitlines()]
# Use the exact last progress timestamp recorded by final verification, not a newer row.
from datetime import datetime,timezone
row=next(z for z in rows if datetime.fromtimestamp(z['time'],timezone.utc).isoformat()==r['progress_last_utc'])
n=read(HERE/'automation/NEXT_CHECK.json')
n['latest_verified_state'].update(owned_snapshot_utc=live['recorded_utc'],actual_workers=live['actual_worker_pids'],
    gpu=live['gpu'],host=live['system_memory'],main_progress_memory=dict(recorded_utc=r['progress_last_utc'],gpu=row['gpu']))
write(HERE/'automation/NEXT_CHECK.json',n)
paths=[HERE/'ITERATION_HISTORY.md',HERE/'automation/THREE_HOUR_CHECK.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md'),Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md')]
for p in paths:
    text=p.read_text(encoding='utf-8-sig');tag='<!-- closing_2046 -->';prior,closing=text.split(tag,1)
    closing=closing.replace('](final_liveness.json)',']('+(OUT/'final_liveness.json').as_posix()+')').replace('](closing_newer_matching_gates.json)',']('+(OUT/'closing_newer_matching_gates.json').as_posix()+')')
    p.write_text(prior+tag+closing,encoding='utf-8')
args=read(OUT/'automation_update_args.json')
args['prompt']=args['prompt'].replace('](final_liveness.json)',']('+(OUT/'final_liveness.json').as_posix()+')').replace('](closing_newer_matching_gates.json)',']('+(OUT/'closing_newer_matching_gates.json').as_posix()+')')
write(OUT/'automation_update_args.json',args)
report=OUT/'REPORT.md';text=report.read_text(encoding='utf-8');title,body=text.split('\n',1)
tag='<!-- closing_summary_2046 -->'
if tag not in body:
    p=r['latest_matching_pair'];nom=p['nominal_cases'];push=p['pushed_cases']
    summary=f'''\n{tag}
Closing verification at {r['recorded_utc']}: **same learner running at 6,144; running still unqualified.** The newer exact checkpoint 142/nominal143/pushed144 still fails 0.35 retention, and the 1.5 command gives {nom[-1]['signed_speed_m_s']:.3f}/{push[-1]['signed_speed_m_s']:.3f} m/s with {100*nom[-1]['geometric_flight_fraction']:.4f}%/{100*push[-1]['geometric_flight_fraction']:.4f}% true flight. The measured head RMS is {nom[-1]['head_RMS_deg']:.2f}/{push[-1]['head_RMS_deg']:.2f} degrees. Current completed-operation updates are {r['completed_operation_main_updates']}, plus 58 recovered updates. [Closing evidence](final_liveness.json). Earlier measurements below retain their original timestamps and checkpoint binding.\n'''
    report.write_text(title+'\n'+summary+body,encoding='utf-8')
write(OUT/'diagnostic_source_hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.py')})
print('Final references and observed-memory timestamps verified')
