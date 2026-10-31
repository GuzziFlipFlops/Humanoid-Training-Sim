import json, hashlib, time, urllib.request, sys
from pathlib import Path
from datetime import datetime, timezone
import psutil
OUT=Path(__file__).resolve().parent; HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
active=read(HERE/'ACTIVE_RUN.json');run=Path(active['run']);s=read(run/'state.json')
processes=[]
for p in psutil.process_iter(['pid','cmdline','create_time']):
 a=p.info['cmdline'] or []
 if any(Path(x).name in ['worker_unitree.py','supervisor_unitree.py'] for x in a):processes.append(p.info)
assert not processes,processes
cfg=read(run/'chain_config.json');mismatch=[]
for rel,h in cfg['sources_sha256'].items():
 if rel in ('supervisor_unitree.py','vram_ramp.py'):p=run/rel
 elif rel.startswith('physical_assets/'):p=run/'source/physical_v29/assets'/rel.removeprefix('physical_assets/')
 elif rel.startswith('physical_v29/'):p=run/'source/physical_v29'/rel.removeprefix('physical_v29/')
 else:p=run/'source/physical_v29/training'/rel
 if sha(p)!=h:mismatch.append(rel)
gates=[]
for a in s['attempts']:
 if a['operation']=='eval' and a.get('completed') and 'gate_' in Path(a['folder']).name:
  p=Path(a['folder']);r=read(p/'result.json')
  gates.append(dict(folder=str(p),result=r))
record=dict(observed_utc=datetime.now(timezone.utc).isoformat(),active=active,status=s['status'],
 error=s.get('error'),checkpoint=s['checkpoint'],checkpoint_sha256=sha(s['checkpoint']),
 total_main_updates=s['total_training_iterations'],inherited_updates=s.get('prior_training_iterations'),
 processes=processes,hash_count=len(cfg['sources_sha256']),hash_mismatches=mismatch,
 failure=read(Path(s['operation_path'])/'failure.json'),last_gates=gates[-6:],
 host_available_mb=psutil.virtual_memory().available/2**20)
(OUT/'initial_evidence.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ['last_gates','active','error']},indent=2))
url='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Grid/default_environment.usd'
cache=OUT/'ground_asset';cache.mkdir(exist_ok=True)
try:
 with urllib.request.urlopen(url,timeout=30) as response:
  data=response.read(); headers=dict(response.headers)
 (cache/'default_environment.usd').write_bytes(data)
 fetched=dict(url=url,bytes=len(data),sha256=sha(cache/'default_environment.usd'),headers=headers,completed=True)
except Exception as e:fetched=dict(url=url,error=repr(e),completed=False)
(OUT/'ground_download.json').write_text(json.dumps(fetched,indent=2)+'\n');print(json.dumps(fetched,indent=2))
