"""Every same-actor physical array and unchanged classifier case must match."""
import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def validate():
    pause=read(OUT/'diagnostic_pause.json');old=Path(pause['completed_eval']);new=OUT/'matched_focused_prior/unchanged_actor_nominal'
    if read(old/'run_config.json')['arguments']['perturb']:
        matches=[p for p in Path(pause['run']).glob('*_nominal') if (p/'result.json').exists()
            and read(p/'result.json').get('completed') and Path(read(p/'run_config.json')['arguments']['checkpoint']).resolve()==Path(pause['checkpoint']).resolve()]
        assert len(matches)==1;old=matches[0]
    assert read(old/'evaluation.json')['cases']==read(new/'evaluation.json')['cases']
    details=[]
    for filename in ['telemetry.npz','gait_diagnostics_50hz.npz','startup_diagnostics.npz']:
        with np.load(old/filename) as a,np.load(new/filename) as b:
            assert set(a.files)==set(b.files)
            for key in a.files:
                assert np.array_equal(a[key],b[key],equal_nan=True) if np.issubdtype(a[key].dtype,np.number) else np.array_equal(a[key],b[key]),(filename,key)
                details.append(dict(file=filename,key=key))
    record=dict(passed=True,all_cases_bitwise_equal=True,arrays_bitwise_equal=len(details),array_details=details,
        source=str(old),replay=str(new),checkpoint=pause['checkpoint'],sha256=pause['checkpoint_sha256'])
    (OUT/'physical_parity.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');return record
if __name__=='__main__':
    result=validate();print('Bitwise physical replay:',result['arrays_bitwise_equal'],'arrays and all cases.')
