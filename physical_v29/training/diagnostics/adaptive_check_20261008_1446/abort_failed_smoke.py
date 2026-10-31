"""Stop only the failed diagnostic PID tree; the harness owns safe resume."""
import json,time,sys
from pathlib import Path
import psutil
OUT=Path(__file__).resolve().parent;folder=OUT/'matched_sequence'/('control_064_train' if '--memory-guard' in sys.argv else 'candidate_smoke')
receipt=json.loads((folder/'process.json').read_text());p=psutil.Process(receipt['pid'])
assert p.create_time()==receipt['create_time']
assert any(Path(x).name=='sequence_candidate.py' for x in p.cmdline())
reason='Reproduced diagnostic MLP subclass slicing incompatibility'
if '--memory-guard' in sys.argv:
    assert json.loads((folder/'failure.json').read_text())['error']=="MemoryError('VRAM_RESERVE_BELOW_300MB')"
    reason='Diagnostic hook retains its own mean tensor; native300MiB reserve guard stopped test'
else:assert 'TypeError: MLP.__init__()' in (folder/'stderr.log').read_text()
tree=p.children(recursive=True)+[p]
identities={q.pid:q.create_time() for q in tree}
for q in tree:
    if q.is_running() and q.create_time()==identities[q.pid]:q.terminate()
_,alive=psutil.wait_procs(tree,timeout=15)
for q in alive:
    if q.create_time()==identities[q.pid]:q.kill()
psutil.wait_procs(alive,timeout=15)
(folder/'controlled_abort.json').write_text(json.dumps(dict(reason=reason,
    owned_PID_identities=identities,diagnostic_actor_discarded=True,production_checkpoint_untouched=True,unix=time.time()),indent=2))
print('Failed diagnostic tree stopped; harness will resume its original supervisor.')
