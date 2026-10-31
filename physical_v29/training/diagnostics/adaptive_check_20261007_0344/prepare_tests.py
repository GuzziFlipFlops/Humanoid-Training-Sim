import json,shutil,hashlib
from pathlib import Path
out=Path(__file__).resolve().parent
live=json.loads((out/'initial_owned_observation.json').read_text())
run=Path(live['active_run']['run']);state=live['state'];cp=Path(state['checkpoint'])
assert state['status']=='failed' and state['attempts'][-1]['failure']['error']=='CHAIN_TIME_LIMIT'
assert not live['actual_worker_pids'] and all(live['frozen_source_integrity'].values())
src=run/'source/physical_v29';dst=out/'control_source/physical_v29'
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
files={p.relative_to(src).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in src.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and 'usd' not in p.parts}
assert all(hashlib.sha256((dst/n).read_bytes()).hexdigest()==h for n,h in files.items())
(out/'protected_source_copy.json').write_text(json.dumps(dict(source=str(src),protected_hashes=files,protected_files=len(files)),indent=2))
(out/'stopped_boundary.json').write_text(json.dumps(dict(run=str(run),checkpoint=str(cp),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),state_status=state['status'],failure='CHAIN_TIME_LIMIT',worker_drained=True,source_hashes_verified=len(live['frozen_source_integrity']),discarded_unsaved_updates=live['operations'][-1]['updates'],scope='Already failed supervisor; last full checkpoint retained, no new training suspension'),indent=2))
print(json.dumps(dict(checkpoint=str(cp),protected_files=len(files),unsaved_updates=live['operations'][-1]['updates'])))
