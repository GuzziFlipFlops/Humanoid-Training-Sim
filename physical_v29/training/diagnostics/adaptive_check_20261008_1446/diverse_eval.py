"""Diagnostic nominal-physics evaluation with independently seeded joint states.

Only reset q/qd vary: +/-0.008rad and +/-0.01rad/s. Native physics, actor inputs,
motor caps, commands and every existing gate remain. Receipts describe this
extra protocol; repeated invocation seeds on unchanged clones are insufficient.
"""
import builtins,runpy,sys,json,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent;TRAIN=OUT/'control_source/physical_v29/training'
sys.path[:0]=[str(TRAIN),str(TRAIN.parent)]
assert sys.argv[sys.argv.index('--mode')+1]=='eval'
folder=Path(sys.argv[sys.argv.index('--output')+1]);seed=int(sys.argv[sys.argv.index('--seed')+1])
candidate=False
original_import=builtins.__import__;installed=False
def install(name,*args,**kwargs):
 global installed
 result=original_import(name,*args,**kwargs)
 if name=='environment_unitree' and not installed:
  installed=True
  import numpy as np,torch
  cls=sys.modules[name].LocomotionEnv
  original_reset=cls._reset_idx
  def reset(self,env_ids):
   original_reset(self,env_ids)
   if not hasattr(self,'stand'):return
   assert self.cfg.evaluation
   ids=self.robot._ALL_INDICES if env_ids is None else env_ids
   joint_states=[];joint_velocities=[]
   for env_id in ids.cpu().tolist():
    # Independent SeedSequence per robot; same realization across branches.
    rng=np.random.default_rng(np.random.SeedSequence([seed,env_id]))
    joint_states.append(rng.uniform(-.008,.008,23));joint_velocities.append(rng.uniform(-.01,.01,23))
   q=self.stand.expand(len(ids),-1).clone()+torch.tensor(np.array(joint_states),device=self.device,dtype=self.stand.dtype)
   qd=torch.tensor(np.array(joint_velocities),device=self.device,dtype=q.dtype)
   assert ((q>=self.target_lo)&(q<=self.target_hi)).all(),'Initial state exceeds protected joint range'
   self.robot.write_joint_state_to_sim(q,qd,env_ids=ids)
   self.targets[ids]=q;self.previous_targets[ids]=q;self.previous_vel[ids]=qd
   if getattr(self,'fixed_evaluation_speed',None) is not None and len(ids)==self.num_envs:
    values=q.cpu().numpy();velocity=qd.cpu().numpy();commands=self.fixed_evaluation_speed.cpu().numpy()
    unique={}
    for command in sorted(set(commands.tolist())):
     mask=commands==command;states=np.concatenate([values[mask],velocity[mask]],1)
     count=len({row.tobytes() for row in states});assert count==int(mask.sum()) and count>=32
     unique[str(command)]=count
    receipt=dict(protocol='independent_seeded_nominal_joint_state_v1',seed=seed,
     per_robot_seed_sequence='[seed, environment_id]',joint_position_offset_bounds_rad=[-.008,.008],
     joint_velocity_bounds_rad_s=[-.01,.01],unique_initial_states_by_command=unique,
     original_native_physics_and_all_gate_classifiers=True,root_and_material_and_motor_caps_unchanged=True,
     no_observation_or_action_override=True,initial_q=values.tolist(),initial_qd=velocity.tolist(),
     initial_command=commands.tolist(),initial_state_sha256=hashlib.sha256(np.concatenate([values,velocity],1).tobytes()).hexdigest())
    folder.mkdir(exist_ok=True);(folder/'initial_diversity.json').write_text(json.dumps(receipt,indent=2))
  cls._reset_idx=reset;builtins.__import__=original_import
 return result
builtins.__import__=install
runpy.run_path(str(TRAIN/'worker_unitree.py'),run_name='__main__')
