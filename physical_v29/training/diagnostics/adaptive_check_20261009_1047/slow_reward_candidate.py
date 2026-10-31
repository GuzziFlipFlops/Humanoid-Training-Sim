"""Separate-process reward experiment. Original signatures do NOT certify it."""
import builtins, hashlib, json, runpy, sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
TRAIN=OUT/'control_source/physical_v29/training'
sys.path[:0]=[str(OUT),str(TRAIN),str(TRAIN.parent)]
from slow_tracking_kernel import score, VERSION
applied='--slow-tracking-tolerance' in sys.argv
if applied:sys.argv.remove('--slow-tracking-tolerance')
folder=Path(sys.argv[sys.argv.index('--output')+1]);folder.mkdir(parents=True,exist_ok=True)
receipt=dict(version=VERSION,applied=applied,training_only=True,
    original_signature_does_not_certify_executed_reward_override=applied,
    explicit_local_reward_change='For 0.1<command norm<=1, existing speed tolerance max(0.10,0.12*speed) replaces 0.5 m/s Gaussian width. Original yaw frame, weights and every other term remain.',
    stopped_and_faster_rewards_unchanged=True,physics_commands_caps_PPO_teachers_upper_unchanged=True,
    truthful_observations=True,diagnostic_actor_deployed=False,
    kernel_sha256=hashlib.sha256((OUT/'slow_tracking_kernel.py').read_bytes()).hexdigest())
(folder/'experimental_reward.json').write_text(json.dumps(receipt,indent=2)+'\n')
original_import=builtins.__import__;installed=False
def install(name,*args,**kwargs):
    global installed
    result=original_import(name,*args,**kwargs)
    if name=='environment_unitree' and not installed:
        installed=True
        if applied:
            import unitree_reward_port as port
            def kernel(env,std,command_name,asset_cfg=None):
                data=env.scene['robot'].data
                velocity=port.quat_apply_inverse(port.yaw_quat(data.root_quat_w),data.root_lin_vel_w)
                return score(env.command_manager.get_command(command_name),velocity,std)
            port.FUNCTIONS['track_lin_vel_xy_yaw_frame_exp']=kernel
        builtins.__import__=original_import
    return result
builtins.__import__=install
runpy.run_path(str(TRAIN/'worker_unitree.py'),run_name='__main__')
