"""Prepare a separate, bounded higher-command diagnostic; never promote it."""
from pathlib import Path
import hashlib, json

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'high_speed_extrapolation'
OUT.mkdir(exist_ok=True)

def change(text, old, new):
    assert text.count(old)==1, old
    return text.replace(old,new)

for name in ['prepare_and_pause.py','resume_after_diagnosis.py']:
    text=(ROOT/name).read_text(encoding='utf-8')
    text=change(text,'HERE=OUT.parents[1]','HERE=OUT.parents[2]')
    if name=='prepare_and_pause.py':
        text=change(text,'Bounded slow-command reward precision experiment on the current exact full state versus unchanged control; all gates, weights and native limits retained',
            'Bounded untrained higher-command extrapolation at 3.5/4/4.5/5 m/s with unchanged physics, current exact actor and all original gates; no promotion or endpoint inference')
    else:
        text=change(text,"'slow_reward_candidate.py'}","'slow_reward_candidate.py','high_speed_probe.py'}")
    (OUT/name).write_text(text,encoding='utf-8')

(OUT/'diverse_eval.py').write_bytes((ROOT/'diverse_eval.py').read_bytes())
text=(ROOT/'motor_retention_probe.py').read_text(encoding='utf-8')
text=change(text,'HERE=OUT.parents[1]','HERE=OUT.parents[2]')
text=change(text,"out=OUT/'motor_retention_response'","out=OUT/'physical_response'")
text=change(text,"actor_files={'current':checkpoint,'qualified_prior':Path(certificate['checkpoint'])}","actor_files={'current':checkpoint}")
text=change(text,'for seed in [4491,4511]:','for seed in [4591,4611]:')
text=change(text,"for actor in ['current','qualified_prior']:","for actor in ['current']:")
text=change(text,"for speed in ([.15,.35,.65,1.,1.5,3.,3.5] if actor=='current' else [.15,.35,.65,1.,1.5,3.]):","for speed in [3.5,4.,4.5,5.]:")
text=change(text,'cfg.stage_speed=3.5','cfg.stage_speed=5.')
text=change(text,'initial_state_seeds=[4491,4511]','initial_state_seeds=[4591,4611]')
text=change(text,"scope='832 physical robots,20s,32 paired independent initial states per cohort at each of two seeds. Exact current actor versus gate-certified prior; unchanged native fixed commands/heading feedback. Actual5ms requested/applied torque, motor speeds/driving clips and representative physical q/qd/target/phase. No action offsets,support,phase/input rewrite or learned diagnostic actor. This is controlled diagnosis, not complete independent locomotion qualification.'",
    "scope='256 robots,20s,32 independent initial states per command at each of two seeds. Current actor under untrained3.5/4/4.5/5 commands, original native fixed commands/heading feedback, limits and truthful histories. First-episode alive-only motor attribution excludes reset/post-fall spikes. This is bounded extrapolation, not trained-stage promotion, final qualification or a causal actuator ceiling.'")
text=change(text,"scope='Current actor and qualified prior, unchanged fixed commands at two initial-state seeds. Actual requested/applied torque, native speed ratio and driving clips at5ms. No command/action/observation intervention; one test cannot prove a high-speed ceiling.'",
    "scope='Current actor only. Untrained higher fixed commands at two initial-state seeds. Actual requested/applied torque, native speed and driving clips at5ms; summary excludes the terminal control tick and all subsequent resets/post-fall samples. No cap change, promotion or causal ceiling inference.'")
text=change(text,"    failed=torch.zeros(n,dtype=torch.int64,device=raw.device)","""    failed=torch.zeros(n,dtype=torch.int64,device=raw.device)
    original_dones=raw._get_dones
    terminal_root=[];terminal_causes=[]
    reason_names=['low_height','pelvis_tilt','torso_tilt','invalid_differential','nonfinite','excessive_drift','nonfoot_impact']
    def passive_dones():
        done=original_dones()
        terminal_root.append(raw.robot.data.root_state_w.detach().cpu().numpy().copy())
        terminal_causes.append(torch.stack([raw.termination_reasons[k] for k in reason_names],1).detach().cpu().numpy().copy())
        return done
    raw._get_dones=passive_dones""")
text=change(text,"    summaries=[]","""    data['pre_reset_root']=np.asarray(terminal_root)
    data['termination_causes']=np.asarray(terminal_causes)
    assert len(terminal_root)==1000
    def safe_mean(x):
        return float(np.mean(x)) if np.size(x) else None
    summaries=[]""")
for old,new in [
    ("float(body_velocity[:,:,1][good].mean())","safe_mean(body_velocity[:,:,1][good])"),
    ("float(data['root'][window,ids,12][good].mean())","safe_mean(data['root'][window,ids,12][good])"),
    ("float(abs(data['root'][window,ids,8])[good].mean())","safe_mean(abs(data['root'][window,ids,8])[good])"),
    ("float(np.asarray(v)[window,ids][good].mean())","safe_mean(np.asarray(v)[window,ids][good])"),
    ("float(np.rad2deg(np.sqrt((data['q'][window,ids][:,:,[raw.lookup['head_yaw'],raw.lookup['head_pitch']]]**2).sum(-1)[good].mean())))","(float(np.rad2deg(np.sqrt(((data['q'][window,ids][:,:,[raw.lookup['head_yaw'],raw.lookup['head_pitch']]]**2).sum(-1))[good].mean()))) if good.any() else None)"),
    ("float(data['healthy'][window,ids][good].mean())","safe_mean(data['healthy'][window,ids][good])")]:
    text=change(text,old,new)
text=change(text,"        ids=slice(i*per,(i+1)*per);x=ratios[1000:,ids];c=clips[1000:,ids]","""        ids=slice(i*per,(i+1)*per)
        alive_mask=np.repeat(data['alive'][:,ids],4,axis=0)[1000:]
        x=ratios[1000:,ids];c=clips[1000:,ids]&alive_mask[:,:,None]""")
text=change(text,"p99_cap_ratio=float(np.quantile(x[:,:,j],.99)),\n            maximum_cap_ratio=float(x[:,:,j].max()),speed_clipped_fraction=float(c[:,:,j].mean()),",
    "p99_cap_ratio=(float(np.quantile(x[:,:,j][alive_mask],.99)) if alive_mask.any() else None),\n            maximum_cap_ratio=(float(x[:,:,j][alive_mask].max()) if alive_mask.any() else None),speed_clipped_fraction=(float(c[:,:,j][alive_mask].mean()) if alive_mask.any() else None),")
text=change(text,"motor_rows.append(dict(name=g[0],command=g[1],motors=motors))","motor_rows.append(dict(name=g[0],command=g[1],first_episode_alive_substep_samples=int(alive_mask.sum()),motors=motors))")
text=change(text,"robots_per_group=per)\n    result=dict", "robots_per_group=per,termination_reason_names=np.array(reason_names))\n    result=dict")
text=change(text,"json.dumps(result,indent=2)+'\\n'","json.dumps(result,indent=2,allow_nan=False)+'\\n'")
compile(text,str(OUT/'high_speed_probe.py'),'exec')
(OUT/'high_speed_probe.py').write_text(text,encoding='utf-8')

manifest=dict(prepared=True,source_probe_sha256=hashlib.sha256((ROOT/'motor_retention_probe.py').read_bytes()).hexdigest(),
    higher_probe_sha256=hashlib.sha256(text.encode()).hexdigest(),commands=[3.5,4,4.5,5],two_initial_state_seeds=[4591,4611],
    population=256,robots_per_command_per_seed=32,duration_seconds=20,
    actor='Fresh exact owned saved boundary after the matched reward experiment resumes',
    no_training_or_policy_modification=True,all_original_classifiers=True,no_ceiling_or_promotion_inference=True,
    first_episode_motor_summary_excludes_terminal_tick_and_all_postfall_reset_samples=True)
(OUT/'preparation.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(manifest,indent=2))
