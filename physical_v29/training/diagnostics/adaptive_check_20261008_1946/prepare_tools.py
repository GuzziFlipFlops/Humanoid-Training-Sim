"""Prepare diagnostics without pausing or changing the production learner."""
import hashlib
import json
import shutil
from pathlib import Path

OUT = Path(__file__).resolve().parent
TRAIN = OUT.parents[1]
active = json.loads((TRAIN / 'ACTIVE_RUN.json').read_text(encoding='utf-8'))
run = Path(active['run'])
source = run / 'source/physical_v29'
copy = OUT / 'control_source/physical_v29'
shutil.copytree(source, copy, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'), dirs_exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
inventory = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc' and 'usd' not in p.parts}
assert all(sha(copy / p) == digest for p, digest in inventory.items())
(OUT / 'protected_source_copy.json').write_text(json.dumps(dict(source=str(source), protected_hashes=inventory, files=len(inventory)), indent=2)+'\n', encoding='utf-8')
prev = OUT.parent / 'adaptive_check_20261008_1446'
for name in ['prepare_and_pause.py', 'resume_after_diagnosis.py', 'diverse_eval.py']:
    shutil.copy2(prev / name, OUT / name)
path = OUT / 'prepare_and_pause.py'
text = path.read_text(encoding='utf-8').replace('Bounded truthful per-robot cycle mean exact-full-state 64/128 control comparison; no diagnostic actor deployed', 'Bounded truthful retention exposure and native motor attribution, exact-full-state controls; no diagnostic actor deployed')
path.write_text(text, encoding='utf-8')
path = OUT / 'resume_after_diagnosis.py'
text = path.read_text(encoding='utf-8').replace("'sequence_candidate.py'", "'sequence_candidate.py','focused_prior_candidate.py','motor_retention_probe.py'")
path.write_text(text, encoding='utf-8')
shutil.copy2(OUT.parent / 'adaptive_check_20261007_1845/capture_current_gradients.py', OUT / 'capture_current_gradients.py')
print('Protected frozen copy and owned pause/resume diagnostic tools prepared.')
