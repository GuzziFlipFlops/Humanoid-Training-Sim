import json,hashlib
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
pause=read(OUT/'diagnostic_pause.json');cp=Path(pause['checkpoint']);nominal=Path(read(OUT/'state_before_diagnosis.json')['operation_path'])
assert nominal.name.endswith('push')
nominal=nominal.with_name(nominal.name.replace('607_run_350_gate_200_push','606_run_350_gate_200_nominal'))
assert read(nominal/'run_config.json')['arguments']['checkpoint']==str(cp)
candidate=OUT/'truthful_nominal'
assert read(candidate/'full_load_proof.json')['source_sha256']==pause['checkpoint_sha256']
assert read(candidate/'evaluation.json')['cases']==read(nominal/'evaluation.json')['cases']
arrays=0;files=[]
for p in sorted(nominal.glob('*.npz')):
    other=candidate/p.name;assert other.exists()
    with np.load(p) as a,np.load(other) as b:
        assert set(a.files)==set(b.files)
        for key in a.files:
            assert np.array_equal(a[key],b[key],equal_nan=True) if a[key].dtype.kind not in 'US' else np.array_equal(a[key],b[key]),(p.name,key)
            arrays+=1
    files.append(dict(file=p.name,original_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),capture_sha256=hashlib.sha256(other.read_bytes()).hexdigest(),all_values_bitwise_equal=True))
source=OUT/'control_source/physical_v29';protected=read(OUT/'protected_source_copy.json')
assert all(hashlib.sha256((source/name).read_bytes()).hexdigest()==value for name,value in protected['protected_hashes'].items())
assert read(OUT/'diagnostic_completion.json')['completed']
result=dict(passed=True,checkpoint=str(cp),sha256=pause['checkpoint_sha256'],passive_unchanged_replay_arrays=arrays,
    all_nominal_cases_identical=True,files=files,protected_source_copy_equal=True,full_state_PPO_load_and_four_updates_pass=True,
    production_change=False,diagnostic_actors_discarded=True,diagnostic_initial_import_failure_preserved=True,
    failed_attempt_safely_resumed=read(OUT/'attempt01_initial_import_failure/diagnostic_resume.json'))
(OUT/'diagnostic_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,arrays=arrays,case_equality=True)))
