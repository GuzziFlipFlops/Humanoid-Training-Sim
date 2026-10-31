"""Prepare diagnostic-only phase advances on the current exact saved actor."""
import json, hashlib, ast
from pathlib import Path
import numpy as np
from trajectory_parameters import NAMES, ORIGIN, offsets_numpy

OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'adaptive_check_20261005_1746'
selection=json.loads((OLD/'validation_selection.json').read_text())
flight=np.array([selection['best_flight']['parameters'][k] for k in NAMES])
best=np.array([selection['best_return']['parameters'][k] for k in NAMES])
parameters=[ORIGIN.copy(),flight.copy()]
labels=['unchanged_control','previous_best_flight_current_actor']
advance=.02*1.65 # Exact current command period: 1/(.9+.5*1.5).
for ticks in [1,2,3]:
    p=flight.copy();p[6:8]-=ticks*advance
    parameters.append(p);labels.append(f'advance_{ticks}_control_ticks')
parameters.append(best);labels.append('previous_best_return_current_actor')
parameters=np.asarray(parameters)
assert np.isfinite(parameters).all() and (parameters[:,6:8]>=0).all() and (parameters[:,6:8]<.5).all()
assert (parameters[:,8:]>=.03).all() and (parameters[:,8:]<=.12).all()
assert np.array_equal(parameters[1,:6],parameters[2,:6])
assert np.array_equal(parameters[1,8:],parameters[4,8:])
ph=np.linspace(0,1,10001,endpoint=False)
assert np.array_equal(offsets_numpy(ph,np.repeat(ORIGIN[None,:],len(ph),0),5),np.zeros((len(ph),3)))
for p in parameters:
    a=offsets_numpy(ph,np.repeat(p[None,:],len(ph),0),5)
    assert np.isfinite(a).all()
    assert np.all(abs(a)<=abs(p[:3])+abs(p[3:6])+1e-12)
    assert np.allclose(a,offsets_numpy(ph+.5,np.repeat(p[None,:],len(ph),0),5),atol=1e-12)
    assert np.array_equal(offsets_numpy(ph,np.repeat(p[None,:],len(ph),0),3),np.zeros_like(a))
record=dict(parameters=parameters.tolist(),labels=labels,diagnostic_only=True,seed=91,
    control_dt_s=.02,period_s=1/1.65,phase_advance_per_tick=advance,
    source_selection_sha256=hashlib.sha256((OLD/'validation_selection.json').read_bytes()).hexdigest(),
    declared_center_bounds=[0,.5],scope='Six fixed cohorts,32 robots each,20 seconds. Centers advanced by 0/20/40/60 ms; offset amplitudes and widths unchanged. Center range is explicitly wider than prior search bounds. Native action/soft-joint/slew/CAD constraints remain. No production offsets.')
(OUT/'timing_selection.json').write_text(json.dumps(record,indent=2)+'\n')
text=(OLD/'trajectory_validation.py').read_text()
text=text.replace("out=OUT/'trajectory_validation'","out=OUT/'timing_probe'")
text=text.replace('per=32; count=3; n=per*count','per=32; count=6; n=per*count')
text=text.replace('cfg.seed=83','cfg.seed=91')
text=text.replace("OUT/'validation_selection.json'","OUT/'timing_selection.json'")
text=text.replace("candidate=i,parameters=", "candidate=i,label=json.loads((OUT/'timing_selection.json').read_text())['labels'][i],parameters=")
text=text.replace("OUT/'trajectory_validation_failure.json'","OUT/'timing_probe_failure.json'")
text=text.replace('Independent seed83:control,best return,best flight,32 robots each,20s.',
    'Current exact actor,seed91: unchanged control,original best-flight offset,20/40/60ms advances and original best-return offset,32 robots each,20s.')
ast.parse(text)
(OUT/'timing_probe.py').write_text(text)
(OUT/'timing_cpu_checks.json').write_text(json.dumps(dict(passed=True,cohorts=6,
    checks=['finite declared parameters','zero control offsets exact','bounded smooth amplitude','half-cycle periodicity','startup fade zero','amplitude/width unchanged under timing advance','script syntax'],
    production_sources_changed=False),indent=2)+'\n')
print(json.dumps(record,indent=2))
