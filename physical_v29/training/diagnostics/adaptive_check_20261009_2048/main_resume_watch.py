"""Compact original-main recovery witness; excludes diagnostic learning."""
import json
import subprocess
import sys
from pathlib import Path
OUT = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
pause = read(OUT / 'diagnostic_pause.json')
run = Path(pause['run'])
r = subprocess.run([sys.executable, str(OUT / 'monitor_check.py')], check=True, capture_output=True, text=True)
m = json.loads(r.stdout)
receivers = []
for p in run.glob('*_train_*/full_load_proof.json'):
    proof = read(p)
    if proof['source_sha256'] != pause['checkpoint_sha256']:
        continue
    result = p.parent / 'result.json'
    receivers.append(dict(folder=str(p.parent), full_load_proof=proof,
        result=read(result) if result.exists() else None,
        completed=result.exists() and read(result).get('completed', False)))
workers = [{k: v for k, v in w.items() if k in ['pid', 'script', 'output', 'observed_updates',
    'KL', 'command_mismatch', 'free_GPU_MiB', 'available_host_MiB', 'three_auxiliary_batches']}
    for w in m['actual_workers']]
d = dict(recorded_utc=m['recorded_utc'], workers=workers, receivers=receivers,
    original_receiver_completed=any(r['completed'] for r in receivers),
    diagnostic_updates_in_main_progress=0, same_supervisor_resume=read(OUT / 'diagnostic_resume.json')['same_supervisor'])
(OUT / 'main_resume_watch.json').write_text(json.dumps(d, indent=2) + '\n')
print(json.dumps(d))
