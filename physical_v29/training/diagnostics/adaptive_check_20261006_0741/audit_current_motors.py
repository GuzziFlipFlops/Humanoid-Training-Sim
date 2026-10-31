"""CPU saved-pose CAD mapping, exact limits; no simulator or ceiling claim."""
import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
folder=OUT/'population_motor_source';folder.mkdir(exist_ok=True)
with np.load(OUT/'101_current_run_350_gate_nominal/telemetry.npz') as archive:z={k:archive[k] for k in archive.files}
assert np.all(z['metrics'][:,:,0]>.5)
groups=np.array([str(x) for x in z['case_speeds']]);assert len(z['case_labels'])==32*len(groups)
np.savez_compressed(folder/'telemetry.npz',q=z['q'],qd=z['qd'],joint_names=z['joint_names'],groups=groups,robots_per_group=32)
original=OUT.parent/'adaptive_check_20261005_0240/analyze_physical.py'
namespace={'__file__':str(original),'__name__':'cpu_saved_motor_mapping'}
exec(compile(original.read_text().split('\nresults={}')[0],str(original),'exec'),namespace)
result=namespace['motor_stats'](folder)
result.update(passed=True,source=str(OUT/'101_current_run_350_gate_nominal'),
 source_checkpoint=json.loads((OUT/'diagnostic_pause.json').read_text())['checkpoint'],
 sample_rate_Hz=10,robots_per_command=32,native_caps_unchanged=True,velocity_ceiling_proven=False,
 scope='Full-population320 robots, physical q/qd sampled10Hz from2-20s. Exact CAD pose-dependent Jacobian/reductions and finite-difference checks. Sparse cap proximity cannot establish substep driving clipping, torque attribution, sustained limit or5/4 endpoint.')
(OUT/'motor_speed_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([{'command':r['group'],'top3':r['motors'][:3]} for r in result['commands']],indent=2))
