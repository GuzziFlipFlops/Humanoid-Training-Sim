import hashlib,json,shutil
from pathlib import Path
out=Path(__file__).resolve().parent;old=out.parent/'adaptive_check_20261006_1243'
run=Path(json.loads((out/'live_handoff.json').read_text())['active_run']['run'])
src=run/'source/physical_v29';dst=out/'control_source/physical_v29'
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','*.pyc','usd','*.usd','*.usda'),dirs_exist_ok=False)
files=[]
for p in src.rglob('*'):
 if p.is_file() and p.suffix in ['.py','.json','.urdf','.npz'] and '__pycache__' not in p.parts:
  q=dst/p.relative_to(src);assert q.is_file() and p.read_bytes()==q.read_bytes();files.append(str(p.relative_to(src)))
(out/'protected_source_copy.json').write_text(json.dumps({'authored_files_identical':files,'source':str(src)},indent=2))
for name in ['pause_for_diagnosis.py','resume_after_diagnosis.py']:
 t=(old/name).read_text();t=t.replace("['steering_probe.py','capture_current_gradients.py','worker_unitree.py']","['arm_response_probe.py','capture_current_gradients.py','worker_unitree.py']")
 (out/name).write_text(t)
t=(old/'steering_probe.py').read_text()
t=t.replace("out=OUT/'steering_response'","out=OUT/'arm_response'")
oldgroups="groups=[(f'{speed}_'+kind,speed,vy,.40,yaw) for speed in [1.,3.5] for kind,vy,yaw in [('control',0.,None),('left_yaw',0.,-.2),('right_yaw',0.,.2),('left_velocity',-.2,None),('right_velocity',.2,None)]]"
newgroups="groups=[(f'{speed}_'+kind,speed,0.,damping,None) for speed in [1.,3.5] for kind,damping in [('control',1.),('damping2',2.),('damping4',4.)]]\n    groups += [(f'0.0_sine{frequency}',0.,0.,1.,None) for frequency in [.75,1.5,2.66,3.25]]"
assert oldgroups in t;t=t.replace(oldgroups,newgroups)
marker="raw.reset();reference_heading=raw.heading_target.clone();observation=raw._get_observations();alive=torch.ones(n,dtype=torch.bool,device=raw.device)"
addition="""
    shoulder_ids=[raw.lookup[x+'_shoulder_pitch'] for x in ['left','right']]
    original_damping=raw.actuator.damping.clone()
    values=torch.tensor([g[3] for g in groups],device=raw.device).repeat_interleave(per)
    raw.actuator.damping[:,shoulder_ids]=values[:,None]
    assert float(original_damping[:,shoulder_ids].min())==1. and float(original_damping[:,shoulder_ids].max())==1.
    frequencies=torch.tensor([float(g[0].split('sine')[1]) if 'sine' in g[0] else 0. for g in groups],device=raw.device).repeat_interleave(per)
    original_pre=raw._pre_physics_step
    def diagnostic_sine_targets(actions):
        original_pre(actions)
        phase_t=(raw.episode_length_buf.float()+1)*raw.step_dt
        offsets=.04*torch.sin(2*torch.pi*frequencies*phase_t)*(frequencies>0)
        desired=raw.targets[:,shoulder_ids]+offsets[:,None]*torch.tensor([1.,-1.],device=raw.device)
        desired=torch.maximum(torch.minimum(desired,raw.target_hi[shoulder_ids]),raw.target_lo[shoulder_ids])
        desired=raw.previous_targets[:,shoulder_ids]+(desired-raw.previous_targets[:,shoulder_ids]).clamp(-raw.slew[shoulder_ids]*raw.step_dt,raw.slew[shoulder_ids]*raw.step_dt)
        raw.targets[:,shoulder_ids]=desired
        raw.delayed_targets[:,shoulder_ids]=torch.where(raw.domain[:,2:3]>.5,raw.previous_targets[:,shoulder_ids],desired)
    raw._pre_physics_step=diagnostic_sine_targets
"""
assert marker in t;t=t.replace(marker,marker+addition)
t=t.replace("'healthy']}","'healthy','heading_error','stiffness','damping']}")
t=t.replace("healthy=raw.reward_port.last_diagnostics['healthy'])","healthy=raw.reward_port.last_diagnostics['healthy'],heading_error=raw.reward_proxy.cad_metrics['heading_error_rad'],stiffness=raw.actuator.stiffness,damping=raw.actuator.damping)")
t=t.replace("requested_lateral_velocity=g[2],requested_yaw_rate=g[4],robots=per", "requested_lateral_velocity=g[2],requested_yaw_rate=g[4],shoulder_damping_Nms_rad=g[3],diagnostic_sine_frequency_Hz=float(g[0].split('sine')[1]) if 'sine' in g[0] else None,robots=per")
t=t.replace("action_pulses=False,","action_pulses=False,diagnostic_shoulder_target_sines=True,diagnostic_shoulder_damping_changes=True,production_deployed=False,")
t=t.replace("320 physical robots,20s,32 per cohort at1/3.5m/s and truthful yaw/lateral command intervention; no action offsets, support, phase/input rewrite or new trained actor. Native constraints retained. Not a qualification gate.","320 physical robots,20s: normal-limit shoulder damping1/2/4 at1/3.5 and0.04rad stopped shoulder sine tests0.75/1.5/2.66/3.25Hz. Truthful student observations, unchanged leg control/free base/gravity/native motor caps/joint/slew limits. Diagnostic only; no changed gain, targets or actor deployed.")
(out/'arm_response_probe.py').write_text(t)
print(json.dumps({'protected_files_identical':len(files),'physical_probe_script_sha256':hashlib.sha256(t.encode()).hexdigest()}))
