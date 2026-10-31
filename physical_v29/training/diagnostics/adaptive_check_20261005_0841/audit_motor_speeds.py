"""Exact CAD motor mapping on preserved post-transient1.5m/s physical samples."""
import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
source=OUT/'phase_causal_response/telemetry.npz';data=np.load(source)
ids=np.r_[np.arange(15*32,16*32),np.arange(19*32,20*32)];mask=data['time']>=5.
folder=OUT/'motor_audit_source';folder.mkdir(exist_ok=True)
np.savez_compressed(folder/'telemetry.npz',q=data['q'][mask][:,ids],qd=data['qd'][mask][:,ids],
  joint_names=data['joint_names'],groups=np.array(['current_1.5','phase_context_neutral_head_1.5']),robots_per_group=32)
prior=OUT.parent/'adaptive_check_20261005_0240/analyze_physical.py'
namespace={'__file__':str(prior),'__name__':'cpu_motor_mapping'}
exec(compile(prior.read_text().split('\nresults={}')[0],str(prior),'exec'),namespace)
result=namespace['motor_stats'](folder)
result.update(source_trace=str(source),sample_rate_Hz=50,post_transient_seconds=15.,
  scope='Two32-robot cohorts, recorded50Hz physical q/qd after5s; exact CAD J/reductions. No substep velocity clipping attribution or proven ceiling.')
(OUT/'motor_speed_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([{k:r[k] for k in ['group','samples_per_motor']}|{'top3':r['motors'][:3]} for r in result['commands']],indent=2))
