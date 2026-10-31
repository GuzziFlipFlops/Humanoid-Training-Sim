"""Cancel an external helper while it is waiting, preserving the real worker."""
import json, time, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone
import psutil
out = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
original = psutil.Process(39000)
assert original.create_time() == 1791346353.9543097
parent = psutil.Process(47044)
assert any(Path(x).name == 'run_bounded_diagnosis.py' for x in parent.cmdline())
children = parent.children(recursive=True)
assert all(any(Path(x).name == 'prepare_and_pause.py' for x in p.cmdline()) for p in children)
leaf = psutil.Process(36928)
assert leaf in children
leaf.suspend()
assert not (out / 'diagnostic_pause.json').exists()
assert not (out / 'teacher_smoke').exists()
for p in reversed(children):
    try: p.kill()
    except psutil.NoSuchProcess: pass
parent.kill()
psutil.wait_procs(children + [parent], timeout=15)
resumed = original.status() == psutil.STATUS_STOPPED
if resumed: original.resume()
state = read(out.parents[1] / 'runs/locomotion_20261007_r40_goal_budget_full_state/state.json')
assert state['status'] == 'running'
archive = out / 'attempt02_preboundary_lookup'
archive.mkdir(exist_ok=False)
for name in ['bounded_diagnosis', 'bounded_diagnosis.log', 'pause.log']:
    path = (out / name).resolve()
    assert path.parent == out.resolve()
    if path.exists(): path.rename(archive / name)
record = dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
    external_harness_cancelled_before_diagnostic=True, original_supervisor=39000,
    original_supervisor_resumed=resumed, actual_training_worker_preserved=True,
    reason='An outdated external smoke-result lookup was caught before the saved-boundary pause. Corrected the path; no physical diagnostic or learner update ran in this attempt.')
(archive / 'RECORD.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record))
