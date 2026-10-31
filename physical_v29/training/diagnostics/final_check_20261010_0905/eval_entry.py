import builtins,runpy,sys,json,hashlib,os
from pathlib import Path
OUT=Path(__file__).resolve().parent;TRAIN=OUT.parents[1]/'runs/locomotion_20261007_r40_goal_budget_full_state/source/physical_v29/training'
sys.path[:0]=[str(TRAIN),str(TRAIN.parent),str(OUT)]
assert sys.argv[sys.argv.index('--mode')+1]=='eval'
folder=Path(sys.argv[sys.argv.index('--output')+1]);seed=int(sys.argv[sys.argv.index('--seed')+1])
original_import=builtins.__import__;installed=False
def install(name,*args,**kwargs):
 global installed
 result=original_import(name,*args,**kwargs)
 if name=='environment_unitree' and not installed:
  installed=True;builtins.__import__=original_import
  import numpy as np,torch
  from native_motor_capture import NativeCapture
  cls=sys.modules[name].LocomotionEnv
  orig_reset=cls._reset_idx;orig_pre=cls._pre_physics_step;orig_step=cls.step;orig_close=cls.close
  def reset(self,env_ids):
   orig_reset(self,env_ids)
   if os.environ.get('FINAL_DIVERSE')!='1' or not hasattr(self,'stand'):return
   ids=self.robot._ALL_INDICES if env_ids is None else env_ids
   q0=[];qd0=[]
   for i in ids.cpu().tolist():
    rng=np.random.default_rng(np.random.SeedSequence([seed,i]));q0.append(rng.uniform(-.008,.008,23));qd0.append(rng.uniform(-.01,.01,23))
   q=self.stand.expand(len(ids),-1).clone()+torch.tensor(np.array(q0),device=self.device,dtype=self.stand.dtype);qd=torch.tensor(np.array(qd0),device=self.device,dtype=q.dtype)
   assert ((q>=self.target_lo)&(q<=self.target_hi)).all()
   self.robot.write_joint_state_to_sim(q,qd,env_ids=ids);self.targets[ids]=q;self.previous_targets[ids]=q;self.previous_vel[ids]=qd
   if getattr(self,'fixed_evaluation_speed',None) is not None and len(ids)==self.num_envs:
    values=q.cpu().numpy();vel=qd.cpu().numpy();cmd=self.fixed_evaluation_speed.cpu().numpy();unique={}
    for c in sorted(set(cmd.tolist())):
     states=np.concatenate([values[cmd==c],vel[cmd==c]],1);unique[str(c)]=len({r.tobytes() for r in states});assert unique[str(c)]>=32
    (folder/'initial_diversity.json').write_text(json.dumps(dict(protocol='independent_seeded_nominal_joint_state_v1',seed=seed,unique_initial_states_by_command=unique,initial_q=values.tolist(),initial_qd=vel.tolist(),commands=cmd.tolist(),position_bounds_rad=[-.008,.008],velocity_bounds_rad_s=[-.01,.01],all_physics_caps_classifiers_unchanged=True),indent=2))
  def pre(self,action):
   orig_pre(self,action)
   if os.environ.get('FINAL_NATIVE')=='1' and not hasattr(self,'_final_native'):self._final_native=NativeCapture(self,folder)
  def step(self,action):
   result=orig_step(self,action)
   if hasattr(self,'_final_native'):self._final_native.done(result[2],result[3])
   return result
  def close(self):
   if hasattr(self,'_final_native'):self._final_native.save()
   return orig_close(self)
  cls._reset_idx=reset;cls._pre_physics_step=pre;cls.step=step;cls.close=close
 return result
builtins.__import__=install
runpy.run_path(str(TRAIN/'worker_unitree.py'),run_name='__main__')
