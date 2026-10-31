"""Freeze diagnostic validation sources and select the best retained actor."""
import hashlib,json,shutil,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
d=json.loads((OUT/'drain.json').read_text());run=Path(d['run'])
assert json.loads((OUT/'cpu_tests.json').read_text())['passed']
physical=json.loads((OUT/'physical_response/result.json').read_text())
assert physical['completed'] and all(g['survival']==1. for g in physical['groups'])
assert next(g for g in physical['groups'] if g['name']=='retract_030')['unloaded_geometric_flight_fraction']>0.
seed=run/'101_run_150_train_021_part_02/candidate.pt'
gates=[run/n/'result.json' for n in ['102_run_150_gate_022_nominal','103_run_150_gate_022_push']]
for p in gates:
    report=json.loads(p.read_text());assert all(c['passed'] for c in report['cases'] if c['statistics']['command_m_s']<1.5)
    config=json.loads((p.parent/'run_config.json').read_text());assert Path(config['arguments']['checkpoint']).resolve()==seed.resolve()
proof=dict(checkpoint=str(seed),sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),
    retained_gate_paths=[str(p) for p in gates],latest_preserved=d['checkpoint'],
    reason='Checkpoint101 passes all stopped/prior-speed nominal+pushed cases; checkpoint107 regresses0.35/0.65. Running is unqualified in both. Retain the better prior-skill mean/exploration; reset values/moments for changed objective; no qualification reuse.',
    teacher=json.loads((run/'state.json').read_text())['promotions'][0]['checkpoint'])
proof['teacher_sha256']=hashlib.sha256(Path(proof['teacher']).read_bytes()).hexdigest()
(OUT/'selected_seed.json').write_text(json.dumps(proof,indent=2)+'\n')
target=OUT/'validation_source/physical_v29'
assert not target.exists(),'Preserve earlier validation source'
shutil.copytree(run/'source/physical_v29',target,ignore=shutil.ignore_patterns('__pycache__','usd','*.pyc'))
for name in ['stage_rewards.py','evaluation.py','reward_warm_start.py']:
    shutil.copy2(OUT/'candidate_source/training'/name,target/'training'/name)
print(json.dumps(proof,indent=2))
