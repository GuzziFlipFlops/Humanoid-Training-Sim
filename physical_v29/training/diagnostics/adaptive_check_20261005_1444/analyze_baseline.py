import json,hashlib
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
old=OUT/'original_online_nominal';new=OUT/'original_offline_nominal'
assert read(old/'wrapper_result.json')['checkpoint_sha256']==read(new/'wrapper_result.json')['checkpoint_sha256']
arrays={}
for name in ['telemetry.npz','gait_diagnostics_50hz.npz','startup_diagnostics.npz']:
 a=np.load(old/name);b=np.load(new/name);assert a.files==b.files
 for key in a.files:arrays[name+'/'+key]=bool(np.array_equal(a[key],b[key],equal_nan=True)) if a[key].dtype.kind not in 'US' else bool(np.array_equal(a[key],b[key]))
assert all(arrays.values()),[k for k,v in arrays.items() if not v]
assert read(old/'evaluation.json')==read(new/'evaluation.json')
proof=dict(passed=True,checkpoint_sha256=read(new/'wrapper_result.json')['checkpoint_sha256'],
 robots=192,duration_seconds=20,arrays=arrays,all_bitwise_equal=True,case_results_identical=True,
 ground_root_bytes_identical=True,ground_sha256=read(OUT/'ground_download.json')['sha256'],
 scope='Same actor and seed; original online ground vs exact local copy. No posture or running qualification inferred.')
(OUT/'ground_physical_parity.json').write_text(json.dumps(proof,indent=2)+'\n')
def cases(folder):
 e=read(folder/'evaluation.json');d=np.load(folder/'telemetry.npz');names=list(d['joint_names']);heads=[names.index('head_yaw'),names.index('head_pitch')];rows=[]
 for ci,case in enumerate(e['cases']):
  st=case['statistics'];mask=d['case_labels']==ci;q=d['q'][:,mask][:,:,heads]
  rows.append(dict(command=st['command_m_s'],passed=case['passed'],failures=case['failures'],
   speed=st['reset_frame_forward_speed_m_s'],flight=st['geometric_flight_fraction'],survival=st['survival'],
   head_combined_RMS_deg=float(np.sqrt(np.mean(np.sum(q*q,axis=-1)))*180/np.pi)))
 return rows
(OUT/'baseline_summary.json').write_text(json.dumps(dict(nominal=cases(new)),indent=2)+'\n')
print(json.dumps(dict(arrays=len(arrays),all_bitwise_equal=True,cases=cases(new)),indent=2))
