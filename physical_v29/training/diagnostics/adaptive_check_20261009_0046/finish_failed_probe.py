"""Close only a recorded failed probe, allowing its harness to resume training."""
import json
from pathlib import Path
import psutil
OUT = Path(__file__).resolve().parent
assert (OUT / 'arm_probe_failure.json').exists()
r = json.loads((OUT / 'arm_target_response/process.json').read_text())
try:
    p = psutil.Process(r['pid'])
    assert p.create_time() == r['create_time']
    assert any(Path(x).name == 'arm_target_probe.py' for x in p.cmdline())
    tree = p.children(recursive=True) + [p]
    identities = {q.pid: q.create_time() for q in tree}
    for q in tree:
        if q.is_running() and q.create_time() == identities[q.pid]: q.terminate()
    _, alive = psutil.wait_procs(tree, timeout=15)
    for q in alive:
        if q.is_running() and q.create_time() == identities[q.pid]: q.kill()
    psutil.wait_procs(alive, timeout=15)
    (OUT / 'failed_probe_shutdown_cleanup.json').write_text(json.dumps(dict(
        failed_probe_only=True, supervisor_touched=False, owned_process_identities=identities,
        reason='Recorded KeyError during configuration loading; no physical data collected. Close stuck Kit shutdown to let guaranteed-resume harness finish.'), indent=2))
except psutil.NoSuchProcess: pass
print('Recorded failed diagnostic closed; harness owns resume.')
