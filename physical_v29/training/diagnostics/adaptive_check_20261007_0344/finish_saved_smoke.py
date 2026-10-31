import psutil,json,time
from pathlib import Path
folder=Path(__file__).resolve().parent/'matched_gain/candidate_smoke'
r=json.loads((folder/'result.json').read_text());identity=json.loads((folder/'process.json').read_text())
assert r['completed'] and Path(r['checkpoint']).is_file()
p=psutil.Process(identity['pid']);assert p.create_time()==identity['create_time']
assert '--output' in p.cmdline() and Path(p.cmdline()[p.cmdline().index('--output')+1]).resolve()==folder.resolve()
owned=p.children(recursive=True)+[p]
for x in owned:
 if x.is_running():x.terminate()
psutil.wait_procs(owned,timeout=20)
(folder/'exit_cleanup.json').write_text(json.dumps(dict(reason='Saved completed smoke hung in Kit exit, recovered bounded harness',owned_pid=p.pid,only_after_completed_result=True),indent=2))
