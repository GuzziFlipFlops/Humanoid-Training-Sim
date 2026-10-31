import json,time
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
p=HERE/'ACTIVE_RUN.json';active=json.loads(p.read_text());s=json.loads((Path(active['run'])/'state.json').read_text())
assert s['status']=='failed' and s['child_pid'] is None
backup=OUT/'active_before_repair.json';assert not backup.exists();backup.write_text(json.dumps(active,indent=2)+'\n')
active.update(status='failed',review_report=str(OUT/'REPORT.md'),review_started_unix=time.time(),
 review_scope='Owned supervisor absent; original full checkpoint preserved. Sequential diagnostic workers only until tested startup repair is deployed.')
t=p.with_suffix('.tmp');t.write_text(json.dumps(active,indent=2)+'\n');t.replace(p)
print('ACTIVE_RUN status now matches preserved failed supervisor state')
