"""Prepare two passive same-full-state objective captures, diagnostic only."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'high_speed_extrapolation'
(OUT/'slow_tracking_kernel.py').write_bytes((ROOT/'slow_tracking_kernel.py').read_bytes())
text=(ROOT/'capture_current_gradients.py').read_text(encoding='utf-8')
text=text.replace("sys.path[:0]=[str(TRAIN),str(TRAIN.parent)]","sys.path[:0]=[str(OUT),str(TRAIN),str(TRAIN.parent)]\nSAVE=Path(sys.argv[sys.argv.index('--output')+1])\nSAVE.mkdir(parents=True,exist_ok=True)")
assert "torch.save(record,OUT/f'current_gradient_{number:02d}.pt')" in text
text=text.replace("torch.save(record,OUT/f'current_gradient_{number:02d}.pt')","torch.save(record,SAVE/f'current_gradient_{number:02d}.pt')")
text=text.replace('current_batch=None\ndef passive_batches','current_batch=None\nrollout_return_value_stats=None\ndef passive_batches')
text=text.replace('global current_batch\n for batch','global current_batch,rollout_return_value_stats\n unnormalized=(self.returns-self.values).detach()\n rollout_return_value_stats=dict(unnormalized_advantage_mean=float(unnormalized.mean()),unnormalized_advantage_std=float(unnormalized.std()),rollout_return_mean=float(self.returns.mean()),rollout_value_mean=float(self.values.mean()))\n for batch')
text=text.replace("record['advantages']=current_batch.advantages.detach().cpu().clone()","record['old_critic_values']=current_batch.values.detach().cpu().clone()\n record['rollout_return_value_stats']=rollout_return_value_stats\n record['advantages']=current_batch.advantages.detach().cpu().clone()")
token="runpy.run_path(str(TRAIN/'worker_unitree.py'),run_name='__main__')"
assert text.count(token)==1
text=text.replace(token,"""import builtins,hashlib
from slow_tracking_kernel import score,VERSION
applied='--slow-tracking-tolerance' in sys.argv
if applied:sys.argv.remove('--slow-tracking-tolerance')
(SAVE/'experimental_reward.json').write_text(json.dumps(dict(applied=applied,version=VERSION,
 original_signature_does_not_certify_executed_override=applied,
 kernel_sha256=hashlib.sha256((OUT/'slow_tracking_kernel.py').read_bytes()).hexdigest(),
 diagnostic_actor_deployed=False),indent=2))
old_import=builtins.__import__;installed=False
def install(name,*args,**kwargs):
 global installed
 result=old_import(name,*args,**kwargs)
 if name=='environment_unitree' and not installed:
  installed=True
  if applied:
   import unitree_reward_port as port
   def kernel(env,std,command_name,asset_cfg=None):
    d=env.scene['robot'].data
    velocity=port.quat_apply_inverse(port.yaw_quat(d.root_quat_w),d.root_lin_vel_w)
    return score(env.command_manager.get_command(command_name),velocity,std)
   port.FUNCTIONS['track_lin_vel_xy_yaw_frame_exp']=kernel
  builtins.__import__=old_import
 return result
builtins.__import__=install
"""+token)
compile(text,str(OUT/'capture_objective_advantages.py'),'exec')
(OUT/'capture_objective_advantages.py').write_text(text,encoding='utf-8')
p=OUT/'resume_after_diagnosis.py';s=p.read_text();s=s.replace("'high_speed_probe.py'}","'high_speed_probe.py','capture_objective_advantages.py'}");p.write_text(s,encoding='utf-8')
print('Prepared original/narrow reward passive PPO capture without policy or optimizer reset.')
