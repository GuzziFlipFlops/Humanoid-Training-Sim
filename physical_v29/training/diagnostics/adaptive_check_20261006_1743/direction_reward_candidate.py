"""Explicit diagnostic runtime wrapper. Altered reward is NOT production-certified.

Original authored sources/checkpoint validators stay intact for the bounded
experiment; their signatures do not certify this hashed runtime override.
The candidate may never be deployed without a new frozen reward version and
tested, narrowly bound provenance migration. All learned diagnostic actors are
otherwise retained solely as rejected/experimental evidence.
"""
import builtins,runpy,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;TRAIN=OUT/'control_source/physical_v29/training'
sys.path[:0]=[str(TRAIN),str(TRAIN.parent),str(OUT)]
original_import=builtins.__import__;installed=False
def install(name,*args,**kwargs):
 global installed
 result=original_import(name,*args,**kwargs)
 if name=='environment_unitree' and not installed:
  installed=True
  import unitree_reward_port as port
  from direction_kernel import score
  def fixed_reset(env,std,command_name,asset_cfg=port.Entity('robot')):
   asset=env.scene[asset_cfg.name]
   vel=port.quat_apply_inverse(port.yaw_quat(asset.data.root_quat_w),asset.data.root_lin_vel_w[:,:3])
   return score(env.command_manager.get_command(command_name),vel,env.cad_metrics['heading_error_rad'],std)
  port.FUNCTIONS['track_lin_vel_xy_yaw_frame_exp']=fixed_reset
  builtins.__import__=original_import
 return result
builtins.__import__=install
runpy.run_path(str(TRAIN/'worker_unitree.py'),run_name='__main__')
