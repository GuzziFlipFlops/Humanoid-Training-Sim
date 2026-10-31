"""Record implementation-only differences and verify every experimental hash."""
import hashlib
import json
from pathlib import Path

here=Path(__file__).resolve().parent
baseline=json.loads((here/'baseline/manifest.json').read_text())
allowed={
 '01':{'cad_differential.py'},
 '02':{'cad_differential.py','owned_cuda_graph.py'},
 '03':{'cad_differential.py','owned_cuda_graph.py','hardware_actuators.py'},
 '04':{'cad_differential.py','owned_cuda_graph.py','hardware_actuators.py'},
 '05':{'cad_differential.py','owned_cuda_graph.py','hardware_actuators.py','training/environment.py'},
}
rows=[]
for iteration,permitted in allowed.items():
    variant=here/('iteration_'+iteration)
    manifest=json.loads((variant/'manifest.json').read_text())
    source=variant/'source/physical_v29'
    for relative,digest in manifest.items():
        assert hashlib.sha256((source/relative).read_bytes()).hexdigest()==digest,(iteration,relative)
    differences={k for k in set(baseline)|set(manifest) if baseline.get(k)!=manifest.get(k)}
    assert differences==permitted,(iteration,differences,permitted)
    rows.append(dict(iteration=iteration,verified_files=len(manifest),differences=sorted(differences),
                     all_other_geometry_physics_reward_policy_gate_files_byte_identical=True))
experiment=json.loads((here/'experiment.json').read_text())
assert hashlib.sha256(Path(experiment['checkpoint']).read_bytes()).hexdigest()==experiment['checkpoint_sha256']
state=json.loads((Path(experiment['run'])/'state.json').read_text())
assert state['status']=='paused_for_performance_review'
result=dict(passed=True,variants=rows,original_checkpoint_untouched=True,
            original_run_still_paused=True,scope='Authored experiment sources; generated USD files are copied unchanged for these benchmarks')
(here/'source_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
