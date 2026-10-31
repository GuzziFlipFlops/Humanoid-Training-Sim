"""Fresh resumed-learning proof and historical comparison; no training changes."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
live=read(HERE/'diagnostics/walk_regression_20261003/live_handoff.json');run=Path(live['active_run']['run'])
assert live['state']['status']=='running' and len(live['actual_worker_pids'])==1
assert len(live['frozen_source_integrity'])==47 and all(live['frozen_source_integrity'].values())
write(OUT/'final_owned_observation.json',live)
folder=run/'138_run_150_train_043';proof=read(folder/'full_load_proof.json')
assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
rows=[json.loads(x) for x in (folder/'progress.jsonl').read_text().splitlines()]
assert len(rows)==64 and all(x['ppo']['pre_update_kl']==0 and x['behavior']['action_command_difference']==0 for x in rows)
wrapper=next(a for a in live['state']['attempts'] if Path(a['folder']).resolve()==folder.resolve())
post={'completed_resumed_updates':len(rows),'source_checkpoint_sha256':proof['source_sha256'],'full_load_proof':proof,
    'maximum_pre_update_KL':0,'maximum_command_difference':0,
    'minimum_free_GPU_MiB':min(x['gpu']['free_mb'] for x in rows),
    'minimum_host_MiB':min(x['gpu']['ram_available_mb'] for x in rows),
    'private_memory_growth_MiB':rows[-1]['gpu']['process_private_mb']-rows[0]['gpu']['process_private_mb'],
    'active_samples_s':64*6144*24/sum(x['collect_seconds']+x['learn_seconds'] for x in rows),
    'worker_effective_samples_s':64*6144*24/wrapper['seconds'],
    'worker_reported_samples_s':64*6144*24/read(folder/'result.json')['seconds'],
    'worker_wrapper_seconds':wrapper['seconds'],'evaluations':[]}
for name in ['139_run_150_gate_044_nominal','140_run_150_gate_044_push']:
    f=run/name;e=read(f/'evaluation.json');z=np.load(f/'telemetry.npz');names=list(z['joint_names']);ids=[names.index('head_yaw'),names.index('head_pitch')]
    values=[]
    for i,c in enumerate(e['cases']):
        s=c['statistics'];head=z['q'][:,z['case_labels']==i][:,:,ids]
        values.append({'command':s['command_m_s'],'passed':c['passed'],'failures':c['failures'],
            'signed_speed':s['reset_frame_forward_speed_m_s'],'flight_fraction':s['geometric_flight_fraction'],
            'survival':s['survival'],'head_RMS_deg':float(np.rad2deg(np.sqrt((head**2).sum(-1).mean())))})
    post['evaluations'].append({'folder':str(f),'checkpoint':str(folder/'candidate.pt'),'cases':values})
write(OUT/'resumed_learning_verification.json',post)
actor=HERE.parents[1]/'humanoid_project/logs/rsl_rl/humanoid5ft_natural_walk/2026-08-31_17-06-10/model_2098.pt'
assert actor.is_file();digest=hashlib.sha256(actor.read_bytes()).hexdigest()
assert digest=='dbddd7222f591d8d43d5e13d679a8ec4f9deb251dd864a15f78e9f88dcf6e174'
evidence=read(OUT/'evidence.json');old_rate=55518.
historical={'old_actor':str(actor),'sha256_verified':digest,'old_nominal_signed_speed':.9938,
    'old_randomized_signed_speed':.9946,'old_nominal_falls':0,'old_trials_per_condition':8,
    'old_control_dt_evidence':'Saved200Hz physical arrays;24-second1.0m/s test, first3seconds excluded',
    'historical_training_collection_steps_s':old_rate,
    'current_completed_main_active_samples_s':evidence['throughput']['active_collection_PPO_samples_s'],
    'current_over_historical_training_rate':evidence['throughput']['active_collection_PPO_samples_s']/old_rate,
    'historical_source':'diagnostics/old_walker_comparison_20261002/REPORT.md and comparison.json',
    'not_controlled_benchmark':True,'old_actor_contract':'80inputs/22outputs,38kg,implicitPD,self collision disabled',
    'current_actor_contract':'400inputs/23outputs,34.46kg,CAD differentials/native limits,self collision enabled',
    'original_gate_not_current_qualification':True}
write(OUT/'original_walker_comparison_refresh.json',historical)
marker='final_20261005_0540_observation'
text=f'''\n\n<!-- {marker} -->
## Final resumed learning and original-walker comparison

Fresh owned snapshot{live['recorded_utc']}: sameR35, {live['state']['total_training_iterations']}completednewmain, `{Path(live['state']['operation_path']).name}`, oneactualworker,47hashes. Resumed block138 completes64actualupdates with exact actor/critic/Adam/exploration load from135, zero maximum pre-updateKL/command mismatch, minGPU{post['minimum_free_GPU_MiB']}MiB/minhost{post['minimum_host_MiB']}MiB/private growth{post['private_memory_growth_MiB']}MiB. Active collection+PPO{post['active_samples_s']:.0f}/worker-effective{post['worker_effective_samples_s']:.0f}samples/s. Matching139/140 outcomes:

| Command | Nominal signed speed | Pushed signed speed | Nominal/pushed genuine flight | Pair qualification |
|---|---|---|---|---|
'''
for a,b in zip(post['evaluations'][0]['cases'],post['evaluations'][1]['cases']):
    text+=f"| {a['command']:.2f} | {a['signed_speed']:.3f} | {b['signed_speed']:.3f} | {100*a['flight_fraction']:.3f}% / {100*b['flight_fraction']:.3f}% | {'Pass' if a['passed'] and b['passed'] else ', '.join(sorted(set(a['failures']+b['failures'])))} |\n"
text+=f'''
Running head RMS{post['evaluations'][0]['cases'][-1]['head_RMS_deg']:.3f}/{post['evaluations'][1]['cases'][-1]['head_RMS_deg']:.3f}degrees. These fresh gates still do not qualify the entire stage. [Exact resumed-learning proof](resumed_learning_verification.json), [final owned snapshot](final_owned_observation.json).

The original five-foot `model_2098.pt` SHA is recomputed and matches the successful historical comparison. Preserved eight-robot24s nominal/randomized1.0-command traces achieved0.9938/0.9946m/s with0falls; they are historical evidence, not a newly rerun stricter32-trial gate. Its collection history recorded55,518steps/s; current completed-main active rate{historical['current_completed_main_active_samples_s']:.0f}is{historical['current_over_historical_training_rate']*100:.1f}%ofthat rate. The former approximately10x-throughput concern is not representative of today's optimized learner. This is not a matched hardware/task benchmark: original80inputs22actions/38kg/implicitPD/no self collision versus current400inputs23actions/34.46kg/CAD torque-speed-slew constraints/self collision. Old walking success is not evidence that current running or bilateral/direction gates pass. [Reverified original evidence](original_walker_comparison_refresh.json), [full historical comparison](../old_walker_comparison_20261002/REPORT.md).

The existing automation is verifiedACTIVE, sameID/thread/three-hour cadence/notification preference/durable scope. History inventory:35physicalruns,1322operations,8historicalmissingmanifestartifacts,0differinghashes/0readwarnings. No5m/s or relevantvelocityceilingendpoint. Leavecomputeron.
'''
report=OUT/'REPORT.md';original=report.read_text()
if marker in original:original=original.split('<!-- '+marker+' -->')[0].rstrip()+'\n\n'
report.write_text(original+text,encoding='utf-8')
brief=f'''\n\nFinal check observation{live['recorded_utc']}: sameR35 at{live['state']['total_training_iterations']}newmainupdates, oneactualCUDAworker/47hashes. Resumed138completed64updates, exact135actor/critic/Adam/exploration, allpreKL/action-commanddifferences0, minGPU{post['minimum_free_GPU_MiB']}MiB/host{post['minimum_host_MiB']}MiB/private+{post['private_memory_growth_MiB']}MiB. Matching139/140 still do not qualify running/retention. Original5ftwalkerSHAreverified,0.9938/0.9946m/swith0/8falls; currentcompleted-mainactive43.8kversushistorical55.5kcollectionsteps/s(differentmodels/contracts,notmatchedbenchmark). See resumed_learning_verification.json, original_walker_comparison_refresh.json and final_owned_observation.json. Same3hautomationACTIVEverified; noendpoint.\n'''
for p in [HERE/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),
    Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
    if live['recorded_utc'] not in p.read_text(encoding='utf-8-sig'):
        with p.open('a',encoding='utf-8') as f:f.write(brief)
n=read(HERE/'automation/NEXT_CHECK.json');n.update(latest_observation=str(OUT/'final_owned_observation.json'),
    latest_verified_state=live,new_main_updates=live['state']['total_training_iterations'],
    resumed_learning_verification=str(OUT/'resumed_learning_verification.json'),original_walker_comparison=str(OUT/'original_walker_comparison_refresh.json'))
write(HERE/'automation/NEXT_CHECK.json',n)
args=read(OUT/'automation_update_args.json');args['prompt']+=brief
write(OUT/'automation_update_final_args.json',args)
print(json.dumps({'final_snapshot':live['recorded_utc'],'completed_main':live['state']['total_training_iterations'],
    'resumed':{k:v for k,v in post.items() if k!='evaluations'},'gates':post['evaluations']},indent=2))
