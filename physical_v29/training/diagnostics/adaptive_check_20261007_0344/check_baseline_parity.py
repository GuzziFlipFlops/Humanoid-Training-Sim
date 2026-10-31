"""Compare saved same-actor evaluations before resumed optimization."""
import json, numpy as np
from pathlib import Path
from datetime import datetime, timezone
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
run=Path(read(OUT/'deployment.json')['run'])
old=read(OUT/'initial_gate_review.json')['latest']
pair=[]
for label,suffix in [('nominal','nominal'),('pushed','push')]:
    new=next(run.glob('*_run_350_gate_001_'+suffix))
    assert read(new/'result.json')['completed']
    assert Path(read(new/'run_config.json')['arguments']['checkpoint']).resolve()==Path(old['checkpoint']).resolve()
    before=Path(old[label])
    assert read(new/'evaluation.json')==read(before/'evaluation.json')
    compared=[]
    assert sorted(p.name for p in before.glob('*.npz'))==sorted(p.name for p in new.glob('*.npz'))
    for file in sorted(before.glob('*.npz')):
        a=np.load(file);b=np.load(new/file.name)
        assert a.files==b.files
        for k in a.files:
            assert a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and a[k].tobytes()==b[k].tobytes(), (file.name,k)
            compared.append(dict(file=file.name,array=k,shape=list(a[k].shape),bitwise_equal=True))
        a.close();b.close()
    pair.append(dict(label=label,original=str(before),resumed=str(new),cases_equal=True,arrays=compared))
(OUT/'baseline_physical_parity.json').write_text(json.dumps(dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
    checkpoint=old['checkpoint'],sha256=old['sha256'],pairs=pair,all_equal=True,
    scope='Full320-robot20-second nominal and pushed frozen-policy replay before new R40 main learning; original rewards, physics and gate classifiers. Does not qualify3.5.'),indent=2)+'\n',encoding='utf-8')
print('Same-actor baseline cases and all saved arrays match both prior tests.')
