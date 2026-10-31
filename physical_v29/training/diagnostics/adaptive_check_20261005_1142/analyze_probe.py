"""CPU-only objective, phase and exact target-projection audit of saved physics.

No simulator, CUDA, policy update or production observation rewrite. Local
derivatives hold previous targets fixed; they do not predict closed-loop gait.
"""
import hashlib
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch

torch.set_num_threads(1)
OUT = Path(__file__).resolve().parent
HERE = OUT.parents[1]
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
pause = read(OUT/'diagnostic_pause.json')
RUN = Path(pause['run'])
FROZEN = RUN/'source/physical_v29'
sys.path[:0] = [str(FROZEN), str(FROZEN/'training')]
from cad_differential import CadDifferential

archive = np.load(OUT/'physical_response/telemetry.npz', allow_pickle=False)
# NPZ decompresses on every access. Cache bounded arrays once, rather than
# rereading the same full trace inside each cohort/joint loop.
data = {k:archive[k] for k in archive.files}
result = read(OUT/'physical_response/result.json')
assert result['checkpoint_sha256'] == pause['checkpoint_sha256']
names = list(data['joint_names'])
groups = list(data['groups'])
per = int(data['robots_per_group'])
post = data['time'] >= 5.
motion = np.load(FROZEN/'training/assets/motion_reference.npz')
stand = torch.tensor(motion['stand'][[list(motion['names']).index(n) for n in names]])
urdf = ET.parse(FROZEN/'training/assets/robot_training.urdf').getroot()
limits = {j.attrib['name']: (float(j.find('limit').attrib['lower']), float(j.find('limit').attrib['upper']))
          for j in urdf.findall('joint') if j.find('limit') is not None}
lo = torch.tensor([limits[n][0] for n in names])
hi = torch.tensor([limits[n][1] for n in names])
middle = (lo+hi)/2
width = (hi-lo)*.95/2
lo, hi = middle-width, middle+width
for i,n in enumerate(names):
    if n in ['left_ankle_roll','right_ankle_roll']:
        lo[i] = max(lo[i], -.13); hi[i] = min(hi[i], .13)
    if n in ['waist_pitch','waist_roll']:
        lo[i] = -.20; hi[i] = .20
slew = torch.tensor([4.5 if 'elbow' in n else 4. if 'head' in n else 12. for n in names])*.02
geometry = read(FROZEN/'assets/differential_geometry.json')
pairs = [([names.index(n) for n in spec['joint_names']], CadDifferential(spec)) for spec in geometry.values()]

def project(action, previous):
    desired = stand + .25*action.clamp(-3.,3.)
    desired = torch.maximum(torch.minimum(desired, hi), lo)
    desired = previous+(desired-previous).clamp(-slew,slew)
    rejected = torch.zeros(len(desired),len(pairs),dtype=torch.bool)
    for k,(ids,model) in enumerate(pairs):
        valid = model.state(desired[:,ids])[2]
        rejected[:,k] = ~valid
        desired[:,ids] = torch.where(valid[:,None],desired[:,ids],previous[:,ids])
    return desired,rejected

# Replay exact deployed algebra on a bounded, evenly spaced physical sample.
# All rows use their own immediately preceding target, never an unrelated pose.
ticks = np.arange(250,1000,5)
robots = np.arange(0,len(groups)*per,3)
a = torch.from_numpy(data['raw_action'][ticks][:,robots]).reshape(-1,len(names))
p = torch.from_numpy(data['target'][ticks-1][:,robots]).reshape_as(a)
t = torch.from_numpy(data['target'][ticks][:,robots]).reshape_as(a)
replay,rejected = project(a,p)
error = float((replay-t).abs().max())
assert error < 1e-5, ('Projection reconstruction differs',error)
derivatives = {}
for n in names:
    j=names.index(n); delta=torch.zeros_like(a); delta[:,j]=.02
    plus = project(a+delta,p)[0]; minus = project(a-delta,p)[0]
    derivatives[n] = ((plus[:,j]-minus[:,j])/.04).numpy()

summaries=[]
sample_labels=np.tile(np.repeat(np.arange(len(groups)),per)[robots],len(ticks))
for k,name in enumerate(groups):
    ids=slice(k*per,(k+1)*per)
    phase=data['reward_phase'][post,ids]
    speed=result['groups'][k]['command']
    duty=.43 if speed>=1.5 else .55
    leg_phase=np.remainder(phase[...,None]+np.array([0.,.5]),1.)
    expected=leg_phase>=duty
    double=expected.all(-1)
    force=data['force'][post,ids]
    actual=np.abs(force)<1.
    command_error=float(np.abs(data['observed_command'][:,ids]-data['action_command'][:,ids]).max())
    assert command_error==0.
    row=dict(group=str(name),command=speed,command_difference=command_error,
             intended_double_swing_fraction=float(double.mean()),
             mean_total_force_in_double_swing_N=float(force[double].sum(-1).mean()) if double.any() else None,
             loaded_foot_in_double_swing_fraction=float(data['contact'][post,ids][double].any(-1).mean()) if double.any() else None,
             actual_running_unload_in_double_swing=float(data['term_running_unload'][post,ids][double].mean()) if double.any() else None,
             objective_difference_vs_running_baseline={term.removeprefix('term_'):float(data[term][post,ids].mean()-data[term][post,:per].mean()) for term in data if term.startswith('term_')},
             raw_vs_effective_action_RMS=float(np.sqrt(((data['raw_action'][post,ids]-data['effective_action'][post,ids])**2).mean())),
             joints={})
    for j,n in enumerate(names):
        raw=data['raw_action'][post,ids,j];target=data['target'][post,ids,j];q=data['q'][post,ids,j]
        d=derivatives[n][sample_labels==k]
        row['joints'][n]=dict(raw_action_p05=float(np.percentile(raw,5)),raw_action_p95=float(np.percentile(raw,95)),
                              target_mean_rad=float(target.mean()),q_mean_rad=float(q.mean()),
                              target_error_abs_rad=float(np.abs(target-q).mean()),
                              raw_action_clipped_fraction=float((np.abs(raw)>3.).mean()),
                              target_clipped_fraction=float(data['target_clipped'][post,ids,j].mean()),
                              target_slew_fraction=float(data['target_slew'][post,ids,j].mean()),
                              local_target_derivative_zero_fraction=float((np.abs(d)<1e-6).mean()),
                              local_target_derivative_median=float(np.median(d)))
    summaries.append(row)
analysis=dict(checkpoint=pause['checkpoint'],checkpoint_sha256=pause['checkpoint_sha256'],
              source_trace_sha256=hashlib.sha256((OUT/'physical_response/telemetry.npz').read_bytes()).hexdigest(),
              projection_replay=dict(samples=len(a),maximum_error_rad=error,
                                    pair_rejection_fraction=rejected.float().mean(0).tolist(),
                                    pair_order=list(geometry),passed=True),groups=summaries,
              qualifications=False,
              scope='Saved 50 Hz truthful-state physical replay. Local action/target derivative holds previous target fixed, includes native action/joint/slew/CAD constraints, and is not a policy gradient or a motor ceiling.',
              legacy_helper_erratum='result.json mean_unload_score and telemetry unload_score use the old average-foot helper. Only term_running_unload is the actual deployed v8 summed-load reward. Do not use the old helper to infer objective incentive.')
(OUT/'physical_objective_analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')

# Reuse the preserved exact-CAD CPU motor mapper on this new saved trace.
prior=HERE/'diagnostics/adaptive_check_20261005_0240/analyze_physical.py'
namespace={'__file__':str(prior),'__name__':'cpu_motor_mapping'}
exec(compile(prior.read_text().split('\nresults={}')[0],str(prior),'exec'),namespace)
assert namespace['RUN']==RUN
motor=namespace['motor_stats'](OUT/'physical_response')
motor.update(sample_rate_Hz=50,scope='All 288 physical robots and 20 seconds at 50 Hz, exact CAD Jacobians/reductions and finite-difference unit checks. Includes intervention transient. Not substep clipping attribution or a verified actuator ceiling.')
(OUT/'motor_speed_audit.json').write_text(json.dumps(motor,indent=2)+'\n')
print(json.dumps(dict(projection=analysis['projection_replay'],cohorts=[dict(group=r['group'],
      double_swing_load_N=r['mean_total_force_in_double_swing_N'],
      zero_target_derivative={n:round(r['joints'][n]['local_target_derivative_zero_fraction'],3) for n in ['left_knee_pitch','right_knee_pitch','left_ankle_pitch','right_ankle_pitch']}) for r in summaries],
      motor_top=[dict(group=r['group'],top=r['motors'][:2]) for r in motor['commands']]),indent=2))
