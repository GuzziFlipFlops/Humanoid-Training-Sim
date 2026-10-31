"""Independent saved-data attribution of driving versus braking overspeed."""
import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;F=OUT/'motor_retention_response'
r=json.loads((F/'result.json').read_text());source=OUT/'control_source/physical_v29/assets'
limits=json.loads((source/'actuator_limits.json').read_text());geometry=json.loads((source/'differential_geometry.json').read_text())
with np.load(F/'substep_motor_200hz.npz') as z:
 ratio=z['cap_ratio'];velocity=z['native_velocity'];request=z['requested_torque'];applied=z['applied_torque'];saved=z['speed_clip'];names=z['motor_names'].tolist()
assert ratio.shape==velocity.shape==request.shape==applied.shape==saved.shape
assert len(ratio)==4000 and np.isfinite(ratio).all() and np.isfinite(request).all() and np.isfinite(applied).all()
rows=[]
for gidx,g in enumerate(r['groups']):
 ids=slice(gidx*32,(gidx+1)*32);ra=ratio[1000:,ids];v=velocity[1000:,ids];req=request[1000:,ids];app=applied[1000:,ids];saved_clip=saved[1000:,ids]
 over=ra>=1;drive=req*v>0;zero_app=app*v<=0
 independently_clipped=over&drive&zero_app
 assert np.array_equal(independently_clipped,saved_clip),(g['name'],'driving clip mask mismatch')
 motors=[]
 for j,name in enumerate(names):
  if '_motor_' in name:
   pair=name.rsplit('_motor_',1)[0];record=limits[geometry[pair]['joint_names'][0]];peak=record['motor_peak_Nm']
   unit='Exact CAD motor output torque Nm after declared transmission mapping; flat current limit plus native velocity cap'
  else:
   record=limits[name];peak=record['effort_Nm'];unit='Native ordinary joint output torque Nm; not a newly measured motor-side available torque'
  x=dict(name=name,torque_unit_scope=unit,declared_peak_torque_Nm=peak,p99_speed_cap_ratio=float(np.quantile(ra[:,:,j],.99)),
   maximum_speed_cap_ratio=float(ra[:,:,j].max()),overspeed_fraction=float(over[:,:,j].mean()),
   overspeed_requested_driving_fraction=float((over[:,:,j]&drive[:,:,j]).mean()),
   overspeed_requested_braking_or_zero_fraction=float((over[:,:,j]&~drive[:,:,j]).mean()),
   actual_driving_clip_fraction=float(saved_clip[:,:,j].mean()),requested_torque_peak_Nm=float(abs(req[:,:,j]).max()),
   applied_torque_peak_Nm=float(abs(app[:,:,j]).max()),requested_above_declared_peak_fraction=float((abs(req[:,:,j])>peak+1e-6).mean()),
   applied_at_declared_peak_fraction=float((abs(app[:,:,j])>=peak-1e-4).mean()))
  motors.append(x)
 rows.append(dict(name=g['name'],command=g['command'],actor=g['actor'],post_warmup_substeps=3000,motors=motors))
result=dict(completed=True,passed=True,checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],sample_rate_Hz=200,records=rows,
 independent_saved_numpy_driving_clip_mask_matches=True,no_native_cap_change=True,ceiling_proven=False,
 scope='Requested/applied torque and velocity recorded from the actual native HardwareActuator. Overspeed with braking demand does not demonstrate blocked forward-driving torque. Declared peak assumptions are not a hardware identification or thermal certificate. Actor-specific tested commands only; final4/5 gates are not passed.')
(OUT/'motor_substep_attribution.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
for row in rows:
 print(json.dumps(dict(name=row['name'],overcap=[m for m in row['motors'] if m['overspeed_fraction']>0],driving_clipped=[m['name'] for m in row['motors'] if m['actual_driving_clip_fraction']>0])))
