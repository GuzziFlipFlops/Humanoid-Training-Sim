"""Receipt after explicit inspection of the final six-column physical plot."""
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
OUT = Path(__file__).resolve().parent
p = OUT/'flight_retention_traces.png'
d = json.loads((OUT/'flight_retention_analysis.json').read_text())
assert len(d['records']) == 6
r = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), path=str(p),
    sha256=hashlib.sha256(p.read_bytes()).hexdigest(), visually_inspected=True,
    observations='All six exact nominal force, sole-clearance and mass-weighted COM traces were inspected. Alternating loads and swing clearance remain in every policy; the earlier baseline and both earlier-state continuations retain brief genuine flight in full-population gates, while both current controls lose it. Force peaks and COM ranges differ between policies. These selected 4-6 second representatives do not establish a reward-clock fault, mechanical cause, complete-return advantage or actuator ceiling.',
    scope='One 50Hz representative per nominal test; full32-robot population gates and actual2.02s post-warmup first-episode scope are saved separately.')
(OUT/'flight_visual_inspection.json').write_text(json.dumps(r,indent=2)+'\n')
print('Final six-column measured force/clearance/COM plot inspected and hashed.')
