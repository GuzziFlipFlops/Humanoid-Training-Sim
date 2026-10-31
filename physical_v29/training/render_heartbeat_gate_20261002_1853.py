"""CPU-only pose reconstruction of the actual 150-update standing gate."""
from pathlib import Path
import json
from build_unitree_review import reconstruction

here = Path(__file__).parent
active = json.loads((here / 'ACTIVE_RUN.json').read_text(encoding='utf-8-sig'))
run = Path(active['run'])
assert run.name == 'locomotion_20261002_r22_unitree_nominal'
out = here / 'diagnostics/heartbeat_20261002_1853'
out.mkdir(exist_ok=True)
frames, summary = reconstruction(run, out, subdir='standing_gate_150_frames',
    evaluation_folder='017_stand_gate_002_nominal', cases=[('r22_stand_150', 0)],
    policy_scope='actual 150-update unqualified standing evaluation')
assert len(frames) == 201 # 10 Hz samples plus the explicit final control tick
assert summary[0]['first_failure_s'] is None
(out / 'standing_gate_150_frames.json').write_text(json.dumps(dict(frames=frames, summary=summary), indent=2), encoding='utf-8')
sheets = sorted((out / 'standing_gate_150_frames').glob('*_sheet_*.jpg'))
html = '<!doctype html><html><meta charset="utf-8"><title>R22 150-update standing evaluation</title><body style="font:16px sans-serif;max-width:1250px;margin:24px auto">'
html += '<h1>R22: actual 150-update standing evaluation</h1><p>CPU reconstructions from synchronized joint/root telemetry; these are not Isaac screenshots. All 201 frames cover 20 seconds at 10 Hz plus the final control tick for robot0. It remains upright but drifts; the independent standing gate fails its anchor limit. No walking or completion claim.</p>'
html += ''.join(f'<img style="max-width:100%" src="standing_gate_150_frames/{path.name}" alt="Consecutive standing frames, sheet {i+1}">' for i, path in enumerate(sheets))
html += '</body></html>'
(out / 'standing-gate-review.html').write_text(html, encoding='utf-8')
print(json.dumps(dict(frames=len(frames), sheets=len(sheets), summary=summary, review=str(out / 'standing-gate-review.html')), indent=2))
