"""Bind actual first deployed block to its complete independent gate pair."""
import json,hashlib,tomllib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
write=lambda p,v:p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
live=read(HERE/'diagnostics/walk_regression_20261003/live_handoff.json');run=Path(live['active_run']['run']);s=live['state']
assert s['status']=='running' and len(live['actual_worker_pids'])==1
assert len(live['frozen_source_integrity'])==49 and all(live['frozen_source_integrity'].values())
first=run/'011_run_150_train_001';proof=read(first/'full_load_proof.json');deployment=read(OUT/'deployment.json')
assert proof['source_sha256']==deployment['checkpoint_sha256']
assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
rows=[json.loads(r) for r in (first/'progress.jsonl').read_text().splitlines()]
assert len(rows)==64 and read(first/'result.json')['completed']
assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
wrapper=next(a for a in s['attempts'] if Path(a['folder']).resolve()==first.resolve());assert wrapper['completed']
def describe(folder):
 e=read(folder/'evaluation.json');cfg=read(folder/'run_config.json');z=np.load(folder/'telemetry.npz');names=list(z['joint_names'])
 assert Path(cfg['arguments']['checkpoint']).resolve()==(first/'candidate.pt').resolve()
 ids=[names.index('head_yaw'),names.index('head_pitch')];values=[]
 for i,c in enumerate(e['cases']):
  t=c['statistics'];q=z['q'][:,z['case_labels']==i][:,:,ids];head=np.rad2deg(np.linalg.norm(q,axis=-1))
  assert t['trials']>=32
  values.append(dict(command=t['command_m_s'],passed=c['passed'],failures=c['failures'],signed_speed=t['reset_frame_forward_speed_m_s'],
   flight_fraction=t['geometric_flight_fraction'],survival=t['survival'],heading_deg=t['final_heading_abs_mean_deg'],
   lateral_speed=t['reset_frame_lateral_speed_m_s'],head_RMS_deg=float(np.sqrt((head**2).mean())),
   head_p95_deg=float(np.percentile(head,95)),fraction_head_within_5deg=float((head<=5).mean())))
 return dict(folder=str(folder),checkpoint_sha256=hashlib.sha256((first/'candidate.pt').read_bytes()).hexdigest(),passed=e['passed'],cases=values)
pair=[describe(run/n) for n in ['012_run_150_gate_002_nominal','013_run_150_gate_002_push']]
samples=sum(r['num_envs']*24 for r in rows)
record=dict(recorded_utc=live['recorded_utc'],completed_main_updates=64,first_deployed_block=str(first),
 exact_full_load=proof,maximum_pre_update_KL=0,maximum_command_difference=0,
 prior_conditioned_samples=sum(r['locomotion_retention']['teacher_phase_conditioned_samples'] for r in rows),
 minimum_free_GPU_MiB=min(r['gpu']['free_mb'] for r in rows),minimum_host_MiB=min(r['gpu']['ram_available_mb'] for r in rows),
 private_memory_growth_MiB=rows[-1]['gpu']['process_private_mb']-rows[0]['gpu']['process_private_mb'],
 torch_allocation_growth_MiB=rows[-1]['gpu']['torch_allocated_mb']-rows[0]['gpu']['torch_allocated_mb'],
 environments=8192,active_samples_s=samples/sum(r['collect_seconds']+r['learn_seconds'] for r in rows),
 main_worker_effective_samples_s=samples/wrapper['seconds'],worker_internal_samples_s=samples/read(first/'result.json')['seconds'],
 same_checkpoint_nominal_pushed=pair,running_qualified=all(x['passed'] for x in pair),
 retained_main_updates=5825,velocity_ceiling_proven=False,live_operation=s['operation_path'],
 actual_worker_pids=live['actual_worker_pids'],frozen_hashes_verified=49,
 scope='One completed deployed block, not long-term stability. Qualification belongs to exact saved actor and full gates.')
write(OUT/'deployed_gate_verification.json',record);write(OUT/'final_owned_observation.json',live)
evidence=read(OUT/'evidence.json');evidence.update(final_deployed_verification=record,latest_owned_snapshot=str(OUT/'final_owned_observation.json'));write(OUT/'evidence.json',evidence)
text=f'''\n\n## First complete deployed block and independent pair

At {live['recorded_utc']}, R36 has completed 64 genuine main updates at 8,192 environments. Exact actor, critic, Adam and exploration load from R35 checkpoint 213 is verified. All pre-update KL and observed/action command differences are zero. Teacher conditioning processed {record['prior_conditioned_samples']:,} samples. Minimum free GPU memory is {record['minimum_free_GPU_MiB']} MiB and host available is {record['minimum_host_MiB']} MiB; private memory grows {record['private_memory_growth_MiB']} MiB and Torch allocation grows {record['torch_allocation_growth_MiB']} MiB. Active collection-plus-PPO throughput is {record['active_samples_s']:.0f}, main-worker-effective throughput is {record['main_worker_effective_samples_s']:.0f}, and worker-internal throughput is {record['worker_internal_samples_s']:.0f} samples/s. These are block measurements, not whole-chain rates.

The exact saved actor from block 011 is evaluated in matching 192-robot, 20-second nominal 012 and pushed 013 tests, with at least 32 independent trials per command:

| Command | Nominal / pushed signed m/s | Nominal / pushed true flight | Matching pair |
|---|---|---|---|
'''
for a,b in zip(pair[0]['cases'],pair[1]['cases']):
 text+=f"| {a['command']:.2f} | {a['signed_speed']:.3f} / {b['signed_speed']:.3f} | {100*a['flight_fraction']:.4f}% / {100*b['flight_fraction']:.4f}% | {'Pass' if a['passed'] and b['passed'] else ', '.join(sorted(set(a['failures']+b['failures'])))} |\n"
text+=f"\nOne actual CUDA worker and all 49 frozen hashes verify. The subsequent operation is `{Path(s['operation_path']).name}`. Preserve this learned state and judge later changes against these exact gates. The same adaptive three-hour automation is verified active; no 5 m/s or actuator-ceiling endpoint is established. [Complete deployed verification](deployed_gate_verification.json), [fresh owned snapshot](final_owned_observation.json), [automation verification](automation_verified.json).\n"
with (OUT/'REPORT.md').open('a',encoding='utf-8') as f:f.write(text)
brief=f"\n\nFinal deployed observation {live['recorded_utc']}: R36 completed its first 64 main updates at 8,192 with exact original full-state load, zero KL/command mismatch, minimum GPU {record['minimum_free_GPU_MiB']} MiB and host {record['minimum_host_MiB']} MiB. Active/main-worker-effective throughput {record['active_samples_s']:.0f}/{record['main_worker_effective_samples_s']:.0f} samples/s. Matching 012/013 results are saved in deployed_gate_verification.json and REPORT.md. This is one actual learned block, not capacity or long-term stability. One worker and 49 hashes verify; same adaptive three-hour automation remains active and computer stays on.\n"
for p in [HERE/'ITERATION_HISTORY.md',Path('E:/Aditya/robot_sim/docs/robot/TRAINING_JOURNAL_20261001.md'),Path('C:/Users/rshan/Documents/GitHub/Humanoid-Robot/docs/robot/TRAINING_JOURNAL_20261001.md')]:
 with p.open('a',encoding='utf-8') as f:f.write(brief)
n=read(HERE/'automation/NEXT_CHECK.json');n.update(recorded_utc=live['recorded_utc'],new_main_updates=s['total_training_iterations'],
 latest_observation=str(OUT/'final_owned_observation.json'),deployed_gate_verification=str(OUT/'deployed_gate_verification.json'),
 latest_verified_state=dict(recorded_utc=live['recorded_utc'],status=s['status'],stage=s['stage'],operation=s['operation_path'],
 actual_workers=live['actual_worker_pids'],frozen_hashes_verified=49,full_snapshot=str(OUT/'final_owned_observation.json'),gpu=live['gpu'],host=live['system_memory']))
write(HERE/'automation/NEXT_CHECK.json',n)
args=read(OUT/'automation_update_args.json')
summary='Final first-block matching gates: '+ '; '.join(f"{a['command']:.2f}: {'pass' if a['passed'] and b['passed'] else '/'.join(sorted(set(a['failures']+b['failures'])))} ({a['signed_speed']:.3f}/{b['signed_speed']:.3f} m/s, flight {100*a['flight_fraction']:.4f}%/{100*b['flight_fraction']:.4f}%)" for a,b in zip(pair[0]['cases'],pair[1]['cases']))
args['prompt']+=f"\n\nFinal observed update {live['recorded_utc']}: first deployed block 011 completed 64 genuine updates at 8,192, exact full checkpoint load, zero KL/command mismatch, minimum GPU {record['minimum_free_GPU_MiB']} MiB and host {record['minimum_host_MiB']} MiB, private growth {record['private_memory_growth_MiB']} MiB. Active/main-worker-effective rates {record['active_samples_s']:.0f}/{record['main_worker_effective_samples_s']:.0f} samples/s. Read deployed_gate_verification.json for exact 012/013 matching tests. {summary}. Later operation {Path(s['operation_path']).name} is active. Preserve useful progress; these are not 5 m/s or motor-ceiling proof."
write(OUT/'automation_update_args.json',args)
print(json.dumps(record,indent=2))
