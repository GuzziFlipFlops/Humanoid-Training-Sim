import json,time
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
active=read(HERE/'ACTIVE_RUN.json');run=Path(active['run']);state=read(run/'state.json')
print(json.dumps({k:state.get(k) for k in ['status','stage','environment_count','operation','operation_path','child_pid','supervisor_pid','checkpoint']}))
for proc in psutil.process_iter(['pid','cmdline','status','memory_info']):
    if any(Path(x).name in ['prepare_and_pause.py','sequence_candidate.py','worker_unitree.py'] for x in proc.info['cmdline'] or []):
        print(json.dumps(dict(owned_kind=next(Path(x).name for x in proc.info['cmdline'] if Path(x).name in ['prepare_and_pause.py','sequence_candidate.py','worker_unitree.py']),pid=proc.pid,status=proc.info['status'],RSS_MiB=proc.info['memory_info'].rss/2**20)))
root=OUT/'matched_sequence'
if root.exists():
    folders=sorted(p for p in root.iterdir() if p.is_dir())
    for folder in folders[-3:]:
        completed=(folder/'result.json').exists() and read(folder/'result.json').get('completed')
        rows=(folder/'progress.jsonl').read_text(encoding='utf-8').splitlines() if (folder/'progress.jsonl').exists() else []
        print(json.dumps(dict(folder=folder.name,completed=completed,updates=len(rows),last_ppo=(json.loads(rows[-1]).get('ppo',{}) if rows else {}))))
    if (root/'failure.json').exists():print((root/'failure.json').read_text())
if (OUT/'pause.log').exists():print('\n'.join((OUT/'pause.log').read_text().splitlines()[-2:]))
print(json.dumps(dict(host_available_MiB=psutil.virtual_memory().available/2**20,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))))
