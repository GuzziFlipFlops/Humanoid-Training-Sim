"""CPU exact CAD mapping; bind interleaved robots to actual case labels."""
import json,hashlib
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
review=json.loads((OUT/'closing_gate_review.json').read_text());pair=review['latest']
source=Path(pair['nominal'])/'telemetry.npz'
folder=OUT/'population_motor_source';folder.mkdir(exist_ok=True)
with np.load(source) as z:
    labels=z['case_labels'];commands=z['case_speeds'];order=np.argsort(labels,kind='stable')
    counts=np.bincount(labels,minlength=len(commands));assert (counts==32).all()
    assert np.array_equal(labels[order],np.repeat(np.arange(len(commands)),32))
    q=z['q'][:,order];qd=z['qd'][:,order];names=z['joint_names']
    assert np.all(z['metrics'][:,:,0]>.5)
    tau=z['tau'];effort=dict(peak_abs_applied_joint_torque_Nm=np.max(abs(tau),axis=(0,1)).tolist(),
        joint_names=names.tolist(),scope='Actual10Hz applied generalized joint torque only; not motor-side available torque or driving saturation.')
# A fixture reproduces the old contiguous-group assumption and correct order.
test_labels=np.arange(12)%3;test=np.argsort(test_labels,kind='stable')
assert np.array_equal(test_labels[test],np.repeat(np.arange(3),4))
assert not np.array_equal(test_labels,np.repeat(np.arange(3),4))
np.savez_compressed(folder/'telemetry.npz',q=q,qd=qd,joint_names=names,
    groups=np.array([str(float(s)) for s in commands]),robots_per_group=32)
original=OUT.parent/'adaptive_check_20261005_0240/analyze_physical.py'
namespace={'__file__':str(original),'__name__':'cpu_saved_motor_mapping'}
exec(compile(original.read_text().split('\nresults={}')[0],str(original),'exec'),namespace)
result=namespace['motor_stats'](folder)
result.update(passed=True,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    source_checkpoint=pair['checkpoint'],source_checkpoint_sha256=pair['sha256'],sample_rate_Hz=10,
    robots_per_command=32,actual_case_labels=labels.tolist(),reordered_original_robot_indices=order.tolist(),
    case_label_fixture_passed=True,applied_joint_effort=effort,native_caps_unchanged=True,velocity_ceiling_proven=False,
    diagnostic_accounting_correction='Older0741 helper assumed contiguous command cohorts, while evaluation saves interleaved case_labels. New audit stably reorders q/qd by saved labels before the unchanged exact CAD mapping. The older artifact is preserved; its per-command attribution is unreliable. Aggregate observed samples/caps and no-ceiling conclusion are unchanged. No production fault or motor-limit change.',
    scope='Full-population320 robots,10Hz post-warmup saved q/qd. Exact CAD Jacobian/reductions and directional finite differences. Sparse samples cannot establish5ms driving clips, sustained torque/speed limitation or5/4 endpoint.')
(OUT/'motor_speed_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([dict(command=r['group'],top3=r['motors'][:3]) for r in result['commands']],indent=2))
