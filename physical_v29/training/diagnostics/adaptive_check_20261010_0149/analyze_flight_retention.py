"""Measured force/sole/COM traces; no inferred reward clock or causal actuator limit."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_full_state'
read=lambda p:json.loads(Path(p).read_text())
names=['earlier_baseline_nominal','current_nominal_replay']+[f'seed{s}_{b}_128_nominal' for s in [317,331] for b in ['control','earlier']]
records=[];traces=[]
for name in names:
 folder=ROOT/name
 if not (folder/'operation_receipt.json').exists():continue
 ev=read(folder/'evaluation.json');case=next(c for c in ev['cases'] if c['statistics']['command_m_s']==1.5)
 cfg=read(folder/'run_config.json')['arguments'];cp=Path(cfg['checkpoint'])
 with np.load(folder/'gait_diagnostics_50hz.npz') as z:
  ids=np.flatnonzero(np.isclose(z['commands'],1.5));time=z['time'];stats=[]
  first_gate_time=2.+float(z['step_dt'])
  for rid in ids:
   terminals=np.flatnonzero(z['done'][:,rid]);end=int(terminals[0]) if len(terminals) else len(time)
   valid=(np.arange(len(time))<end)&(time>=first_gate_time-1e-6)
   clearance=z['clearance'][:,rid];force=z['foot_force'][:,rid];com=z['com'][:,rid]
   assert np.isfinite(clearance).all() and np.isfinite(force).all() and np.isfinite(com).all()
   flight=(clearance>.001).all(1)&(np.abs(force)<1.).all(1)
   stats.append(dict(representative_index=int(rid),first_terminal_index=int(end),valid_rows=int(valid.sum()),
    actual_postwarmup_time_origin_s=first_gate_time,genuine_flight_fraction=float(flight[valid].mean()),
    minimum_sole_clearance_p95_mm=float(np.quantile(clearance[valid].min(1),.95)*1000),
    simultaneous_unloaded_force_fraction=float((np.abs(force[valid])<1).all(1).mean()),
    COM_height_range_m=float(np.ptp(com[valid,2]))))
  rid=int(ids[0]);end=stats[0]['first_terminal_index'];mask=(time>=4)&(time<=6)&(np.arange(len(time))<end)
  traces.append(dict(label=name,time=time[mask],force=z['foot_force'][mask,rid],clearance=z['clearance'][mask,rid],com=z['com'][mask,rid]))
 records.append(dict(name=name,folder=str(folder),checkpoint=str(cp),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
  full_population_gate=case,representative_50Hz=stats))
fig,axes=plt.subplots(3,len(traces),figsize=(4.3*len(traces),9),squeeze=False)
for col,t in enumerate(traces):
 axes[0,col].set_title(t['label'].replace('_','\n'),fontsize=9)
 for foot,label in enumerate(['left','right']):
  axes[0,col].plot(t['time'],t['force'][:,foot],label=label)
  axes[1,col].plot(t['time'],t['clearance'][:,foot]*1000,label=label)
 axes[0,col].axhline(1,color='black',lw=.7,ls='--');axes[1,col].axhline(1,color='black',lw=.7,ls='--')
 axes[2,col].plot(t['time'],t['com'][:,2]);axes[2,col].set_xlabel('Simulation time (s)')
 for row in range(3):axes[row,col].grid(alpha=.25)
 axes[0,col].legend(fontsize=7)
for row,label in enumerate(['Measured sole force (N)','Sole clearance (mm)','Mass-weighted COM height (m)']):axes[row,0].set_ylabel(label)
fig.suptitle('Actual 1.5 m/s force, clearance and COM — one representative per exact nominal test\nPopulation gates and first-episode sampling scope are recorded separately',fontsize=11)
fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(OUT/'flight_retention_traces.png',dpi=130);plt.close(fig)
d=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),records=records,
 geometric_flight_definition='Both actual sole clearances >1 mm and both absolute sole forces <1 N, matching the unchanged gate.',
 scope='50 Hz selected representative measurements from the actual2.02 s post-warmup origin, first episodes before terminal/reset; full32-robot-per-command independent statistics are retained separately. COM derivative, hidden reward phase and mechanical coupling are not inferred. Different complete policies/states are compared; no complete-return, hardware-ceiling or production-repair qualification from these traces.')
(OUT/'flight_retention_analysis.json').write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(records=len(records),population_flight=[dict(name=r['name'],flight_percent=r['full_population_gate']['statistics']['geometric_flight_fraction']*100,failures=r['full_population_gate']['failures']) for r in records])))
