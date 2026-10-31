"""CPU-only completed/recovered update accounting and exact gate trends."""
import json,re,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;HERE=OUT.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=read(OUT/'state_before_diagnosis.json');pause=read(OUT/'diagnostic_pause.json');run=Path(pause['run'])
blocks=[]
for a in s['attempts']:
 f=Path(a['folder'])
 if a['operation']!='train' or not re.search(r'_run_\d+_train_\d+',f.name) or not a['completed']:continue
 rows=[json.loads(x) for x in (f/'progress.jsonl').read_text().splitlines()];load=read(f/'full_load_proof.json')
 assert all(load[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
 assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
 gpu=[r['gpu'] for r in rows]
 samples=sum(r['num_envs']*24 for r in rows);active=sum(r['collect_seconds']+r['learn_seconds'] for r in rows)
 blocks.append(dict(folder=str(f),updates=len(rows),num_envs=rows[0]['num_envs'],samples=samples,
  active_seconds=active,worker_seconds=a['seconds'],source_sha256=load['source_sha256'],
  exact_actor_critic_Adam_exploration=True,maximum_preupdate_KL=0,maximum_action_command_mismatch=0,
  minimum_free_GPU_MiB=min(x['free_mb'] for x in gpu),minimum_host_available_MiB=min(x['ram_available_mb'] for x in gpu),
  private_growth_MiB=gpu[-1]['process_private_mb']-gpu[0]['process_private_mb'],torch_allocation_growth_MiB=gpu[-1]['torch_allocated_mb']-gpu[0]['torch_allocated_mb']))
assert sum(b['updates'] for b in blocks)==s['total_training_iterations']
recovered=[]
for a in s['attempts']:
 f=Path(a['folder']);cp=f/'memory_recovery.pt'
 if a['operation']!='train' or not a.get('memory_failure') or not cp.exists():continue
 h=sha(cp);receivers=[b['folder'] for b in blocks if b['source_sha256']==h];assert receivers
 rows=[json.loads(x) for x in (f/'progress.jsonl').read_text().splitlines()]
 recovered.append(dict(folder=str(f),checkpoint_sha256=h,updates=len(rows),exact_receivers=receivers,
  samples=sum(r['num_envs']*24 for r in rows),active_seconds=sum(r['collect_seconds']+r['learn_seconds'] for r in rows),worker_seconds=a['seconds']))
endpoint=(OUT/'state_before_diagnosis.json').stat().st_mtime
pauses=[]
for p in HERE.glob('diagnostics/*/diagnostic_resume.json'):
 before=p.parent/'diagnostic_pause.json'
 if not before.exists():continue
 a,b=read(before),read(p)
 if Path(a['run']).resolve()==run.resolve() and a['supervisor_pid']==s['supervisor_pid'] and (p.parent/'state_before_diagnosis.json').stat().st_mtime<endpoint-1:
  pauses.append(dict(receipt=str(p),seconds=b['pause_seconds']))
# This pre-repair measurement ends at the saved-boundary suspension, before
# the current validation pause. The saved-state mtime closely bounds suspend.
endpoint=(OUT/'state_before_diagnosis.json').stat().st_mtime
samples=sum(x['samples'] for x in blocks+recovered)
elapsed=endpoint-s['started_unix']-sum(x['seconds'] for x in pauses)
rates=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),run=str(run),measurement_endpoint_unix=endpoint,
 completed_operation_main_updates=s['total_training_iterations'],verified_recovered_main_updates=sum(x['updates'] for x in recovered),
 actual_retained_main_updates=s['total_training_iterations']+sum(x['updates'] for x in recovered),inherited_main_updates=s['prior_training_iterations'],
 actual_retained_samples=samples,active_collection_PPO_samples_s=samples/sum(x['active_seconds'] for x in blocks+recovered),
 worker_effective_samples_s=samples/sum(x['worker_seconds'] for x in blocks+recovered),
 whole_chain_samples_s=samples/elapsed,excluded_prior_pause_seconds=sum(x['seconds'] for x in pauses),
 recovered_evidence=recovered,blocks=blocks,scope='All completed/proven retained R39 main updates through the saved diagnosis boundary. Capacity/smoke/diagnostic samples excluded. Current validation pause begins after this timing endpoint. Historical documented R39 pauses excluded; no denominator mixes.')
(OUT/'throughput_and_loads.json').write_text(json.dumps(rates,indent=2)+'\n')
g=read(OUT/'initial_gate_review.json');pairs=[x for x in g['gate_pairs'] if '_run_350_' in x['nominal']]
fig,ax=plt.subplots(2,1,figsize=(11,7),sharex=True);x=np.arange(len(pairs))
for test,ls in [('nominal_cases','-'),('pushed_cases','--')]:
 c=[p[test][-1] for p in pairs]
 ax[0].plot(x,[v['survival']*100 for v in c],ls,label=test)
 ax[1].plot(x,[v['lateral_speed'] for v in c],ls,label=test)
ax[0].set(ylabel='Reported survival (%)',title='R39 3.5 m/s gates after repaired command-travel guard')
ax[1].set(ylabel='Fixed-frame lateral speed (m/s)',xlabel='Saved checkpoint operation')
ax[1].set_xticks(x[::3],[Path(p['checkpoint']).parent.name.split('_')[0] for p in pairs[::3]])
for a in ax:a.grid(alpha=.2);a.legend()
fig.tight_layout();fig.savefig(OUT/'pre_repair_gate_trends.png',dpi=145);plt.close(fig)
print(json.dumps({k:v for k,v in rates.items() if k not in ['blocks','recovered_evidence']},indent=2))
