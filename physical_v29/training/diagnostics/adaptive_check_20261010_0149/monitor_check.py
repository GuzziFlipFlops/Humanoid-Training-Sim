"""Compact owned diagnostics progress; never count it as main learning."""
import json
from pathlib import Path
from datetime import datetime,timezone
import psutil
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_full_state'
names={'worker_unitree.py','worker_entry.py','diverse_eval.py'}
workers=[]
for p in psutil.process_iter(['pid','cmdline']):
    args=p.info['cmdline'] or []
    if not any(Path(x).name in names for x in args):continue
    row=dict(pid=p.pid,create_time=p.create_time(),script=next(Path(x).name for x in args if Path(x).name in names),
        matching_child_pids=[c.pid for c in p.children(recursive=True) if any(Path(x).name in names for x in c.cmdline())])
    if '--output' in args:row['output']=args[args.index('--output')+1]
    elif row['script']=='neutral_teacher_probe.py':row['output']=str(OUT/'teacher_compatibility_response')
    folder=Path(row['output'])
    if (folder/'progress.jsonl').exists():
        lines=(folder/'progress.jsonl').read_text().splitlines();last=json.loads(lines[-1])
        row.update(observed_updates=len(lines),KL=last['ppo']['pre_update_kl'],command_mismatch=last['behavior']['action_command_difference'],
            free_GPU_MiB=last['gpu']['free_mb'],available_host_MiB=last['gpu']['ram_available_mb'],
            three_auxiliary_batches=[last[k]['batches'] for k in ['retention','locomotion_retention','upper_refinement']],
            target_samples=last['locomotion_retention'].get('command_local_coupled_samples'))
    if (folder/'stdout.log').exists() and row['script']=='neutral_teacher_probe.py':
        ticks=[line for line in (folder/'stdout.log').read_text(errors='replace').splitlines() if line.startswith('{"tick"')]
        if ticks:row['physical_progress']=json.loads(ticks[-1])
    workers.append(row)
receipts=[]
for f in sorted(ROOT.glob('*/operation_receipt.json'),key=lambda x:x.stat().st_mtime):
    receipt=json.loads(f.read_text());receipts.append(receipt)
latest=receipts[-3:]
actual=[w for w in workers if not w['matching_child_pids']]
snapshot=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),actual_workers=actual,launcher_and_worker_processes=workers,
    last_completed_operations=latest,diagnostic_main_samples=0,actual_worker_count=len(actual),
    Windows_virtualenv_launchers_excluded_by_verified_matching_child_tree=True)
assert len(actual)<=1
(OUT/'monitor_snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n');print(json.dumps(snapshot,indent=2))
