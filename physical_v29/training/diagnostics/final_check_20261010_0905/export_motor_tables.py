import json,csv,gzip,hashlib,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;T=OUT.parents[1]
sys.path.insert(0,str(T/'automation'))
from motor_shaft_metrics import summarize_motor_arrays
read=lambda p:json.loads(p.read_text())
def export(folder,scope,commands=None):
 mapping=read(folder/'motor_mapping.json');names=mapping['motor_names'];refs=mapping['shaft_references']
 with np.load(folder/'native_motor_200hz.npz') as z:arr={k:z[k] for k in z.files}
 n=arr['applied_torque_Nm'].shape[1];times=arr['interval_start_s'];data_fields=['position_rad','velocity_rad_s','velocity_RPM','requested_torque_Nm','applied_torque_Nm','signed_mechanical_power_W','valid']
 assert len(times)==len(np.unique(times)) and np.allclose(np.diff(times),.005)
 np.testing.assert_allclose(arr['signed_mechanical_power_W'],arr['applied_torque_Nm']*arr['velocity_rad_s'],rtol=2e-6,atol=.0003)
 np.testing.assert_allclose(arr['velocity_RPM'],arr['velocity_rad_s']*60/(2*np.pi),rtol=2e-6,atol=.0001)
 summaries={};trial_stats={};csv_files=[]
 for c in commands or sorted(set(arr['command_m_s'].tolist())):
  ids=np.flatnonzero(np.isclose(arr['command_m_s'],c));mask=arr['alive'][:,ids]&(times[:,None]>=2.)
  d={k:arr[k][:,ids] for k in ['applied_torque_Nm','velocity_RPM','signed_mechanical_power_W','valid']};d.update(motor_names=names,shaft_references=refs)
  summaries[str(c)]=summarize_motor_arrays(d,mask)
  trial_stats[str(c)]=[]
  for ii in ids:
   d={k:arr[k][:,ii:ii+1] for k in ['applied_torque_Nm','velocity_RPM','signed_mechanical_power_W','valid']};d.update(motor_names=names,shaft_references=refs)
   trial_stats[str(c)].append(dict(environment_id=int(arr['environment_ids'][ii]),rows=summarize_motor_arrays(d,arr['alive'][:,ii:ii+1]&(times[:,None]>=2.))))
  target=folder/f'motor_samples_{round(c*100):03d}.csv.gz';csv_files.append(str(target))
  if not target.exists():
   fields=['sample_index','environment_id','command_m_s','interval_start_s','interval_end_s','alive','terminal','truncated','included_after_settle']+[motor+'__'+field for motor in names for field in data_fields]
   with gzip.open(target,'wt',encoding='utf-8',newline='',compresslevel=3) as f:
    w=csv.writer(f);w.writerow(fields)
    for sample,t in enumerate(times):
     for ii in ids:
      header=[sample,int(arr['environment_ids'][ii]),float(c),float(t),float(t+.005),int(arr['alive'][sample,ii]),int(arr.get('terminated',np.zeros_like(arr['alive']))[sample,ii]),int(arr.get('truncated',np.zeros_like(arr['alive']))[sample,ii]),int(t>=2 and arr['alive'][sample,ii])]
      values=np.stack([arr[k][sample,ii] for k in data_fields],axis=1).reshape(-1)
      w.writerow(header+[format(v,'.7g') for v in values])
  fig,axs=plt.subplots(4,6,figsize=(19,11),sharex=True);selected=ids[0];window=(times>=2)&(times<=5)
  for m,ax in enumerate(axs.flat):
   if m>=len(names):ax.axis('off');continue
   ax.plot(times[window],arr['applied_torque_Nm'][window,selected,m],label='torque Nm',lw=.7)
   right=ax.twinx();right.plot(times[window],arr['velocity_RPM'][window,selected,m],color='tab:orange',lw=.6,label='RPM');ax.set_title(names[m],fontsize=8);ax.grid(alpha=.2);ax.tick_params(labelsize=6);right.tick_params(labelsize=6)
  fig.suptitle(f'{scope}: {c:g} m/s command, representative original 200Hz shaft torque (blue) and RPM (orange); population summary separately',fontsize=11)
  fig.tight_layout();fig.savefig(folder/f'motor_torque_rpm_{round(c*100):03d}.png',dpi=135);plt.close(fig)
  # Separate angle, power and speed timeline retains signed motoring/braking.
  fig,axs=plt.subplots(4,6,figsize=(19,11),sharex=True)
  for m,ax in enumerate(axs.flat):
   if m>=len(names):ax.axis('off');continue
   ax.plot(times[window],arr['position_rad'][window,selected,m],color='tab:green',lw=.7);right=ax.twinx();right.plot(times[window],arr['signed_mechanical_power_W'][window,selected,m],color='tab:red',lw=.6);ax.set_title(names[m],fontsize=8);ax.tick_params(labelsize=6);right.tick_params(labelsize=6);ax.grid(alpha=.2)
  fig.suptitle(f'{scope}: {c:g} m/s command, installed-shaft relative angle rad (green) and signed mechanical power W (red)',fontsize=11);fig.tight_layout();fig.savefig(folder/f'motor_angle_power_{round(c*100):03d}.png',dpi=135);plt.close(fig)
 (folder/'motor_population_or_episode_summary.json').write_text(json.dumps(dict(scope=scope,summaries=summaries,per_independent_trial=trial_stats,raw_npz=str(folder/'native_motor_200hz.npz'),raw_csv=csv_files,retained_all_raw_samples=True,excluded_rows='first2s, terminal/reset/post-fall, invalid linkage per motor',power_is_mechanical_not_battery=True),indent=2)+'\n')
 return summaries
def main():
 pop=OUT/'selected_diverse_nominal';summary=export(pop,'32 independent initial-state population',[1.,3.5]);mapping=read(pop/'motor_mapping.json');evals=read(pop/'evaluation.json')
 receipt=read(pop/'completion_receipt.json');tables=[]
 for c,rows in summary.items():
  case=next(x for x in evals['cases'] if abs(x['statistics']['command_m_s']-float(c))<1e-5)
  for r in rows:tables.append(dict(command_m_s=float(c),measured_signed_speed_m_s=case['statistics']['reset_frame_forward_speed_m_s'],actor_sha256=receipt['sha256'],command_case_passed=case['passed'],case_failures=case['failures'],population_trials=32,**r))
 (OUT/'motor_tables.json').write_text(json.dumps(tables,indent=2)+'\n')
 columns=['command_m_s','measured_signed_speed_m_s','actor_sha256','motor','population_trials','samples','torque_RMS_Nm','torque_p95_abs_Nm','torque_peak_abs_Nm','speed_p95_abs_RPM','speed_peak_abs_RPM','signed_mean_mechanical_power_W','mean_positive_mechanical_power_W','mean_braking_mechanical_power_W','peak_abs_mechanical_power_W','simultaneous_peak_power_torque_Nm','simultaneous_peak_power_speed_RPM']
 with (OUT/'motor_tables.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(tables)
 lines=['# Installed motor shaft use at 1 m/s and maximum stable running command','',f"Same exact preserved actor: `{receipt['sha256']}`. Simulation-derived installed-shaft values, native 5ms samples, 32 independently seeded states per command, 18 measured seconds after the first2s settling window. Complete-stage retention remains unqualified. Measured speeds and every failure are in the same-actor gate reports. Separate video episodes are not population averages.",'','Torque columns are RMS / absolute p95 / absolute peak (N·m). RPM columns are absolute p95 / absolute peak. Peaks can occur at different times. The CSV/JSON include signed mechanical power and simultaneous torque/RPM at peak absolute power. Mechanical power is not battery power.','']
 for c in [1.,3.5]:
  rows=[r for r in tables if r['command_m_s']==c];lines += [f"## Command {c:g}m/s; measured signed forward {rows[0]['measured_signed_speed_m_s']:.4f}m/s",'', '| Installed motor | Torque RMS / p95 / peak N·m | RPM p95 / peak | Shaft / external transmission |','|---|---:|---:|---|']
  for r in rows:
   ref=r['shaft'];shaft='CAD differential output/J(q)' if 'reduction' not in ref else f"{ref['shaft']}, {ref['reduction']:g}:1 external, η={ref['modeled_external_efficiency']:g}"
   lines.append(f"| {r['motor']} | {r['torque_RMS_Nm']:.3f} / {r['torque_p95_abs_Nm']:.3f} / {r['torque_peak_abs_Nm']:.3f} | {r['speed_p95_abs_RPM']:.1f} / {r['speed_peak_abs_RPM']:.1f} | {shaft} |")
  lines += ['']
 lines += ['Factory-geared motors and servos use the installed output shaft; inaccessible internal rotor torque is unavailable. DJI hips/knees use bare shaft before external15:1/16:1. Shoulder pitch uses factory output before external4:1 belt. Ankle and waist motors use each exact CAD crank shaft; no constant pitch/roll allocation. Relative angle zero is not encoder-calibrated.','', '[Raw population NPZ](selected_diverse_nominal/native_motor_200hz.npz), [1m/s CSV](selected_diverse_nominal/motor_samples_100.csv.gz), [maximum CSV](selected_diverse_nominal/motor_samples_350.csv.gz), [all motor power and simultaneous operating points](motor_tables.csv).']
 (OUT/'MOTOR_TABLES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('Motor population tables/CSV/NPZ and plots exported.',flush=True)
 if len(sys.argv)>1 and sys.argv[1]=='--episodes':
  dest=Path(read(OUT/'capture_config.json')['destination']);manifest=read(dest/'manifest.json')
  for clip in manifest['clips']:export(dest/f"locomotion_{round(clip['speed']*100):03d}",'Exact synchronized video episode')
if __name__=='__main__':
 if '--available-episodes' in sys.argv:
  dest=Path(read(OUT/'capture_config.json')['destination']);manifest=read(dest/('manifest.json' if (dest/'manifest.json').exists() else 'capture_progress.json'))
  for clip in manifest['clips']:
   folder=dest/f"locomotion_{round(clip['speed']*100):03d}"
   if (folder/'motor_population_or_episode_summary.json').exists():continue
   export(folder,'Exact synchronized video episode')
   print(f"Episode motor CSV/NPZ/angle/torque/RPM/power exported: {clip['speed']:g}m/s",flush=True)
 else:main()
