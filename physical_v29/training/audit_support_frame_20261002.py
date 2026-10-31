"""CPU reconstruction check of the frozen standing support gate coordinate frame."""
from pathlib import Path
import json, math, xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from build_unitree_review import fk, root_pose

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'diagnostics/old_walker_comparison_20261002'
comparison = json.loads((OUT / 'comparison.json').read_text())
run = Path(comparison['current_state']['checkpoint']).parent.parent
folder = run / comparison['current_nominal_gates'][-1]['folder']
data = np.load(folder / 'gait_diagnostics_50hz.npz', allow_pickle=False)
metrics = np.load(folder / 'telemetry.npz',allow_pickle=False)['metrics']
tree = ET.parse(run / 'source/physical_v29/training/assets/robot_training.urdf').getroot()
links = {}
for link in tree.findall('link'):
    inertial = link.find('inertial')
    mass = float(inertial.find('mass').get('value'))
    origin = inertial.find('origin')
    com = np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
    links[link.get('name')] = (mass, com)
names = data['joint_names'].tolist()
indices = np.flatnonzero(data['time']>=2.)[::5]
full_out, yaw_out, com_points, errors = [], [], [], []

def outside(com, feet, rotation, contacts):
    # Only common translation is removed: it cancels in all comparisons.
    c = rotation.T @ com
    f = feet @ rotation
    return bool((c[0]<f[:,0].min()-.055) or (c[0]>f[:,0].max()+.055) or
                (c[1]<f[:,1].min()-.035) or (c[1]>f[:,1].max()+.035) or not contacts.all())

for tick in indices:
    frame_com = []; full_flags=[];yaw_flags=[]
    for robot in range(data['root_state'].shape[1]):
        root = data['root_state'][tick,robot]
        transforms, _ = fk(tree, dict(zip(names,data['q'][tick,robot])),root)
        weighted = sum(mass*(transforms[name][:3,:3]@local + transforms[name][:3,3]) for name,(mass,local) in links.items())
        com = weighted / sum(value[0] for value in links.values())
        feet = data['foot_position'][tick,robot]
        reconstructed_feet = np.stack([transforms[s+'_foot'][:3,3] for s in ['left','right']])
        errors.append(float(np.linalg.norm(reconstructed_feet-feet,axis=1).max()))
        R = root_pose(root)[:3,:3]
        yaw = math.atan2(R[1,0],R[0,0])
        Ryaw = np.array([[math.cos(yaw),-math.sin(yaw),0],[math.sin(yaw),math.cos(yaw),0],[0,0,1]])
        full_flags.append(outside(com-root[:3],feet-root[:3],R,data['contact'][tick,robot]))
        yaw_flags.append(outside(com-root[:3],feet-root[:3],Ryaw,data['contact'][tick,robot]))
        frame_com.append(com)
    full_out.append(full_flags);yaw_out.append(yaw_flags);com_points.append(frame_com)
assert max(errors)<.001
# Reproduce the frame error with a COM exactly over two loaded feet.
pitch=math.radians(12.5)
R=np.array([[math.cos(pitch),0,math.sin(pitch)],[0,1,0],[-math.sin(pitch),0,math.cos(pitch)]])
synthetic_com=np.array([0.,0.,-.35]); synthetic_feet=np.array([[0.,-.10,-.90],[0.,.10,-.90]])
synthetic_full=outside(synthetic_com,synthetic_feet,R,np.array([True,True]))
synthetic_yaw=outside(synthetic_com,synthetic_feet,np.eye(3),np.array([True,True]))
assert synthetic_full and not synthetic_yaw
scope = dict(source=str(folder), robots=data['root_state'].shape[1], reconstruction_ticks=len(indices),
    mass_kg=sum(v[0] for v in links.values()),max_foot_fk_error_m=max(errors),
    raw_recorded_first16_full_rotation_support_outside=float(metrics[:,:16,15].mean()),
    reconstructed_full_rotation_support_outside=float(np.mean(full_out)),
    reconstructed_yaw_only_support_outside=float(np.mean(yaw_out)),
    synthetic_COM_over_feet=dict(full_rotation_false_failure=synthetic_full,yaw_rotation_failure=synthetic_yaw),
    limitation='CPU COM reconstruction from frozen URDF inertias; no direct saved simulator COM. First16, sampled10Hz, not a new128-robot gate.',
    remaining_failures='Drift and standing speed still fail regardless of this support-frame bug. No standing promotion.')
(OUT/'support_frame_audit.json').write_text(json.dumps(scope,indent=2),encoding='utf-8')

# Compare the actual approved walking contact/encoder traces with the current
# standing trace. Different commands are stated explicitly on every panel.
old = ROOT / 'humanoid_project/logs/gait_diagnostics_20260902/nominal'
meta = json.loads((old/'manifest.json').read_text())
physics=np.load(old/'physics.npy',mmap_mode='r')
old_names=meta['joint_names'];ids=meta['cases'][0]['env_ids']
lo,hi=meta['fields']['q'];old_q=physics[:,ids[0],lo:hi]
lo,hi=meta['fields']['foot_force_w'];old_force=physics[:,ids[0],lo:hi].reshape(-1,2,3)
old_t=(np.arange(len(physics))+1)*meta['physics_dt']
old_cut=(old_t>=3)&(old_t<=7)
new_t=data['time'];new_cut=(new_t>=3)&(new_t<=7)
fig,axes=plt.subplots(2,2,figsize=(12,6.4),layout='constrained')
colors=['#247ba0','#d36b18']
for side,color in zip(['left','right'],colors):
    axes[0,0].plot(old_t[old_cut]-3,np.rad2deg(old_q[old_cut,old_names.index(side+'_knee_pitch')]),color=color,label=side)
    axes[0,1].plot(new_t[new_cut]-3,np.rad2deg(data['q'][new_cut,0,names.index(side+'_knee_pitch')]),color=color,label=side)
for j,color in enumerate(colors):
    axes[1,0].plot(old_t[old_cut]-3,old_force[old_cut,j,2],color=color)
    axes[1,1].plot(new_t[new_cut]-3,data['foot_force'][new_cut,0,j],color=color)
axes[0,0].set_title('Archived model_2098 — command 1.0 m/s')
axes[0,1].set_title('Current R22 600-update gate — command 0.0 m/s')
for ax in axes[0]:ax.set_ylabel('Knee angle (degrees)');ax.legend();ax.grid(alpha=.18)
for ax in axes[1]:ax.set_ylabel('Vertical foot force (N)');ax.set_xlabel('Seconds after 3 s warmup');ax.grid(alpha=.18)
fig.suptitle('Saved physical telemetry: walking versus unqualified standing\nDifferent robots and commands; this is not a matched policy-performance test.',fontsize=12)
fig.savefig(OUT/'walking_vs_standing.png',dpi=150)
plt.close(fig)
print(json.dumps(scope,indent=2))
