"""CPU-only descriptive slow-bias/cycle separation on saved actual angles."""
import json,argparse,hashlib
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--gate-review',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
a.output_dir.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
review=read(a.gate_review);pair=review['latest'];sos=butter(4,.5,fs=10.,output='sos')
t=np.arange(2000)*.1
constant=np.ones((2000,1))*13.;trend=sosfiltfilt(sos,constant,axis=0)
assert np.max(np.abs(trend-13.))<1e-10
wave=np.sin(2*np.pi*1.5*t)[:,None];filtered=sosfiltfilt(sos,wave,axis=0)
assert np.sqrt(np.mean(filtered[100:-100]**2))<.001
slow=np.sin(2*np.pi*.1*t)[:,None];filtered=sosfiltfilt(sos,slow,axis=0)
assert np.sqrt(np.mean((filtered-slow)[100:-100]**2))<.001
mixture=constant+slow+wave;filtered=sosfiltfilt(sos,mixture,axis=0)
assert np.sqrt(np.mean((mixture-filtered-wave)[100:-100]**2))<.002
records=[];plot=[]
for label in ['nominal','pushed']:
 folder=Path(pair[label]);z=np.load(folder/'telemetry.npz');names=z['joint_names'].tolist()
 alive=z['metrics'][::5,:,0]>.5;q=np.rad2deg(z['q']).astype(float)
 for case,speed in enumerate(z['case_speeds']):
  ids=np.flatnonzero(z['case_labels']==case)
  # First common continuous episode only; never filter across a reset.
  valid=alive[:,ids].all(1);ends=np.flatnonzero(~valid);end=int(ends[0]) if len(ends) else len(valid)
  if end<80:continue
  data=q[:end,ids];bias=sosfiltfilt(sos,data,axis=0);cycle=data-bias
  # Conservative3s edge exclusion; decomposition remains descriptive.
  w=slice(30,-30);joints={}
  for name in ['head_yaw','head_pitch','left_shoulder_pitch','right_shoulder_pitch','left_shoulder_roll','right_shoulder_roll','left_elbow_pitch','right_elbow_pitch']:
   j=names.index(name)
   span=lambda x:float(np.median(np.quantile(x[w,:,j],.95,axis=0)-np.quantile(x[w,:,j],.05,axis=0)))
   joints[name]=dict(total_p95_p05_deg=span(data),slow_bias_p95_p05_deg=span(bias),cycle_residual_p95_p05_deg=span(cycle),slow_bias_mean_deg=float(bias[w,:,j].mean()),cycle_residual_RMS_deg=float(np.sqrt(np.mean(cycle[w,:,j]**2))))
  start_time=2.+float(z['step_dt']) # Population sampling starts at i>=2s, post-step.
  records.append(dict(test=label,speed_m_s=float(speed),robots=len(ids),continuous_seconds=(end-1)*.1,time_start_s=start_time,time_end_s=start_time+(end-1)*.1,joints=joints,telemetry_sha256=hashlib.sha256((folder/'telemetry.npz').read_bytes()).hexdigest()))
  if label=='nominal' and float(speed) in (2.5,3.):plot.append((float(speed),data[:,0],bias[:,0],cycle[:,0],names,start_time))
result=dict(checkpoint=pair['checkpoint'],sha256=pair['sha256'],filter='Zero-phase4th-order Butterworth0.5Hz,10Hz saved population angles',interpretation='Descriptive separation only. Cutoff chosen below measured gait cadence; residual includes faster non-cycle motion. Angle ranges alone do not certify periodicity, human likeness, or actuator frequency response. No filtering across resets;3s edges excluded from statistics.',synthetic_fixtures_passed=4,records=records)
(a.output_dir/'upper_bias_audit.json').write_text(json.dumps(result,indent=2))
if plot:
 fig,axes=plt.subplots(2,len(plot),figsize=(12,7),squeeze=False,sharex=True)
 for col,(speed,data,bias,cycle,names,start_time) in enumerate(plot):
  for side,color in [('left','tab:blue'),('right','tab:red')]:
   j=names.index(side+'_shoulder_pitch');time=start_time+np.arange(len(data))*.1
   axes[0,col].plot(time,data[:,j],color=color,alpha=.35,label=side+' actual')
   axes[0,col].plot(time,bias[:,j],color=color,label=side+' slow bias')
   axes[1,col].plot(time,cycle[:,j],color=color,label=side+' residual')
  axes[0,col].set_title(f'{speed:g}m/s; first nominal robot,10Hz')
  for ax in axes[:,col]:ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_ylabel('Shoulder pitch(deg)')
  axes[1,col].set_xlabel('Simulation time(s)')
 fig.suptitle('Measured shoulder pitch: slow bias versus faster residual\n0.5Hz descriptive cutoff; full-population statistics saved separately')
 fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(a.output_dir/'upper_bias_decomposition.png',dpi=150);plt.close(fig)
print(json.dumps(dict(checkpoint=pair['checkpoint'],records=len(records),synthetic_fixtures_passed=4)))
