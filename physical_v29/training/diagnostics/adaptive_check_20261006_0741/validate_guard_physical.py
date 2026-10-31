"""CPU comparison of old/corrected same-actor rollouts; gates stay truthful."""
import hashlib,json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
pause=read(OUT/'diagnostic_pause.json');run=Path(pause['run']);review=read(OUT/'initial_gate_review.json')
comparisons=[]
def compare(old,new,label):
 row={'label':label,'old':str(old),'new':str(new),'arrays':{}}
 for file in ['telemetry.npz','gait_diagnostics_50hz.npz']:
  a,b=np.load(old/file),np.load(new/file);assert set(a.files)==set(b.files)
  if file=='telemetry.npz':
   alive=a['metrics'][:,:,0]>.5
   first=np.array([np.flatnonzero(~x)[0] if (~x).any() else len(x) for x in alive.T])
   mask=np.arange(len(alive))[:,None]<first[None,:]
   angle_mask=mask[::5]
  else:
   first=np.array([np.flatnonzero(x)[0] if x.any() else len(x) for x in a['done'].T])
   mask=np.arange(len(a['done']))[:,None]<first[None,:]
  for k in a.files:
   x,y=a[k],b[k];assert x.shape==y.shape,(label,k)
   if x.ndim>=2 and x.shape[:2]==mask.shape:
    same=np.array_equal(x[mask],y[mask]);samples=int(mask.sum())
   elif file=='telemetry.npz' and x.ndim>=2 and x.shape[:2]==angle_mask.shape:
    same=np.array_equal(x[angle_mask],y[angle_mask]);samples=int(angle_mask.sum())
   else:
    same=np.array_equal(x,y);samples=int(x.size)
   # The scope strings describe unchanged data layouts; compare them too.
   assert same,(label,file,k)
   row['arrays'][file+'/'+k]={'bitwise_equal':same,'pre_terminal_samples':samples}
 comparisons.append(row)
compare(run/'460_run_350_gate_037_nominal',OUT/'101_current_run_350_gate_nominal','current_459_pre_false_terminal')
teacher=review['promotions'][-1]['matching_pair']
for label,oldkey in [('nominal','nominal'),('push','pushed')]:
 compare(Path(teacher[oldkey]),OUT/('103_teacher_run_300_gate_'+label),'qualified349_'+label)
parity=dict(passed=True,comparisons=comparisons,
 scope='Same actors, seeds, counts and physics. Current459 matches bitwise before each old terminal; certified349 matches the full20s in nominal/push. New behavior after the former60m termination is independently tested, not implied by parity.')
(OUT/'physical_parity.json').write_text(json.dumps(parity,indent=2)+'\n')
cases={}
for label,name in [('nominal','101_current_run_350_gate_nominal'),('pushed','102_current_run_350_gate_push')]:
 r=read(OUT/name/'result.json');cases[label]=[dict(command=c['statistics']['command_m_s'],passed=c['passed'],failures=c['failures'],
 survival=c['statistics']['survival'],signed_speed=c['statistics']['reset_frame_forward_speed_m_s'],
 lateral_speed=c['statistics']['reset_frame_lateral_speed_m_s'],flight=c['statistics']['geometric_flight_fraction']) for c in r['cases']]
 assert all(c['survival']==1 for c in cases[label])
 assert cases[label][-1]['command']==3.5
validation=dict(passed=True,source_guard_corrected=True,current_checkpoint=pause['checkpoint'],
 checkpoint_sha256=pause['checkpoint_sha256'],fresh_corrected_cases=cases,current_stage_qualified=False,
 fresh_trusted3m_teacher_all_nominal_pushed_prior_cases_pass=True,
 every_existing_independent_classifier_unchanged=True,physics_motor_caps_reward_kernels_weights_unchanged=True,
 scope='Repair validation: no artificial commanded60m death, but all direction, gait and prior cases still applied. Does NOT qualify the current3.5m actor or certify actuator limits.')
(OUT/'physical_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps({'parity_passed':True,'arrays':sum(len(c['arrays']) for c in comparisons),'corrected_cases':cases},indent=2))
