"""Record post-resume learning/gates and verify the preserved automation identity."""
import json,tomllib,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parent;TRAINING=OUT.parents[1];AUTO=TRAINING/'automation'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
owned=read(TRAINING/'diagnostics/walk_regression_20261003/live_handoff.json')
e=read(OUT/'evidence.json');s=owned['state'];progress=owned['worker_progress'];run=Path(owned['active_run']['run'])
assert s['status']=='running' and len(owned['actual_worker_pids'])==1
assert len(owned['frozen_source_integrity'])==46 and all(owned['frozen_source_integrity'].values())
write(OUT/'final_owned_observation.json',owned)
latest=e['gates'][-2:];block=e['blocks'][-1]
proof={'recorded_utc':owned['recorded_utc'],'status':s['status'],'stage':s['stage'],
    'completed_main_updates':s['total_training_iterations'],'operation':s['operation_path'],
    'current_environment_count':s['environment_count'],'actual_worker_pids':owned['actual_worker_pids'],
    'current_iteration':progress.get('iteration'),'current_pre_update_kl':progress.get('ppo',{}).get('pre_update_kl'),
    'gpu':owned['gpu'],'host_available_mb':owned['system_memory']['available_mb'],
    'frozen_hashes_verified':46,'latest_matched_gates':latest,'completed_post_resume_block':block,
    'qualification_scope':'No running-stage or5m/s qualification; same supervisor resumed and completed actual PPO block after GUI exit.'}
write(OUT/'resume_learning_verification.json',proof)
note=f'''\n\nFinal post-resume evidence at{owned['recorded_utc']}: same R34 remains run_150 at{s['total_training_iterations']} completed main updates; block068 completed64 actual PPO updates at{block['active_samples_per_second']:.0f} active/{block['worker_effective_samples_per_second']:.0f} worker-effective samples/s, minimumGPU{block['minimum_free_gpu_mb']}MiB andhost{block['minimum_host_mb']}MiB, all pre-update KL/command mismatches0. Matching069nominal/070pushed on exact068 achieve signed1.479/1.479m/s,100% survival,29/30 alternating landings, but geometric flight0%/0.00694%, below1.5%. Retained0.35 narrowly fails nominal speed at0.449m/s; all pushed priors pass. Thus the earlier065 all-prior pass did not establish sustained retention. Current block071 runs6144 with one actual CUDA worker and46hashes verified. Read resume_learning_verification.json and final_owned_observation.json. No new source/reward change; preserved038 flight actor and qualified walking baseline remain. CPU history inventory:34physical runs,97legacy launches,61original RSL sessions,31CAD groups,1164operations,8historical missing manifest artifacts,0differing hashes,0read warnings. Three-hour heartbeat remains active with completed video priority removed and controlled diagnosis priorities.\n'''
for path in [OUT/'REPORT.md',TRAINING/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),
             Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md'),
             Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/Learning/HumanoidTraining/FILE_LOCATIONS.md'),
             Path('C:/Users/rshan/.codex/skills/humanoid-robot-training/references/physical-v29-locomotion-current.md')]:
    with path.open('a',encoding='utf-8') as f:f.write(note)
handoff=read(AUTO/'NEXT_CHECK.json')
handoff['latest_verified_state']={k:proof[k] for k in ['recorded_utc','status','stage','completed_main_updates','operation','current_environment_count','actual_worker_pids','current_iteration','current_pre_update_kl','gpu','host_available_mb','frozen_hashes_verified']}
handoff['latest_verified_state']['run']=str(run);handoff['latest_verified_state']['checkpoint']=s['checkpoint']
handoff['latest_observation']=str(OUT/'final_owned_observation.json')
handoff['latest_post_resume_gates']=latest;handoff['completed_post_resume_throughput']=block
write(AUTO/'NEXT_CHECK.json',handoff)
args=read(OUT/'automation_update_args.json')
args['prompt']+='\n\nFinal post-resume observation, October5 00:16 UTC: R34 completed block068 and reached1,089 main updates. Matching069 nominal/070 pushed achieve1.479/1.479m/s with100% survival and29/30 alternating landings, but genuine flight0%/0.00694% still fails. Retained0.35 narrowly fails nominal speed at0.449m/s; all pushed priors pass. Current block071 runs6,144; one worker and46hashes verify. Block068 throughput42.7k active/36.6k worker-effective samples/s, minimumGPU1,017MiB andhost5,721MiB, zero pre-update KL/command mismatch. Read resume_learning_verification.json and final_owned_observation.json. This confirms actual resumed learning, not qualification. The new controlled diagnosis priorities remain necessary if later matched gates still plateau.'
write(OUT/'automation_update_args.json',args)
print(json.dumps({k:proof[k] for k in ['status','stage','completed_main_updates','current_iteration','current_environment_count','frozen_hashes_verified']}))
