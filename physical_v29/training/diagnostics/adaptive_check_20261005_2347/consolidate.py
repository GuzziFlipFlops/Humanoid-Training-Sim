"""Consolidate completed full-state learning, exact gates and CPU diagnostics."""
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
write=lambda p,d:Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
live=read(OUT/'closing_owned_observation.json');s=live['state'];run=Path(live['active_run']['run'])
assert s['status']=='running' and s['supervisor_pid']==32492 and len(live['actual_worker_pids'])==1
assert len(live['frozen_source_integrity'])==55 and all(live['frozen_source_integrity'].values())
blocks=[];samples=active=worker=0.
for a in s['attempts']:
    f=Path(a['folder'])
    if a['operation']!='train' or not re.search(r'_run_\d+_train_\d+',f.name) or not a['completed']:continue
    assert read(f/'result.json')['completed']
    rows=[json.loads(x) for x in (f/'progress.jsonl').read_text().splitlines()];load=read(f/'full_load_proof.json')
    assert all(load[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
    assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
    count=sum(r['num_envs']*24 for r in rows);sec=sum(r['collect_seconds']+r['learn_seconds'] for r in rows)
    gpu=[r['gpu'] for r in rows];samples+=count;active+=sec;worker+=a['seconds']
    blocks.append(dict(folder=str(f),updates=len(rows),environment_count=rows[0]['num_envs'],samples=count,active_seconds=sec,
        worker_seconds=a['seconds'],active_samples_s=count/sec,worker_effective_samples_s=count/a['seconds'],
        checkpoint_sha256=sha(f/'candidate.pt'),source_sha256=load['source_sha256'],exact_full_state_load=True,
        zero_preupdate_KL_command_mismatch=True,min_free_GPU_MiB=min(x['free_mb'] for x in gpu),
        min_host_MiB=min(x['ram_available_mb'] for x in gpu),private_growth_MiB=gpu[-1]['process_private_mb']-gpu[0]['process_private_mb'],
        torch_allocation_growth_MiB=gpu[-1]['torch_allocated_mb']-gpu[0]['torch_allocated_mb']))
assert sum(b['updates'] for b in blocks)==s['total_training_iterations']
recovered=[]
for a in s['attempts']:
    f=Path(a['folder']);cp=f/'memory_recovery.pt'
    if a['operation']!='train' or not a.get('memory_failure') or not cp.exists():continue
    digest=sha(cp);receivers=[b['folder'] for b in blocks if b['source_sha256']==digest];assert receivers
    rows=[json.loads(x) for x in (f/'progress.jsonl').read_text().splitlines()]
    recovered.append(dict(folder=str(f),checkpoint_sha256=digest,updates=len(rows),samples=sum(r['num_envs']*24 for r in rows),
        active_seconds=sum(r['collect_seconds']+r['learn_seconds'] for r in rows),worker_seconds=a['seconds'],exact_receivers=receivers))
assert sum(a['updates'] for a in recovered)==58
pauses=[]
for p in HERE.glob('diagnostics/*/diagnostic_resume.json'):
    before=p.parent/'diagnostic_pause.json'
    if not before.exists():continue
    a=read(before);b=read(p)
    if Path(a['run']).resolve()==run.resolve() and a['supervisor_pid']==s['supervisor_pid'] and a['paused_unix']>=s['started_unix']:
        pauses.append(dict(receipt=str(p),seconds=b['pause_seconds']))
pause_seconds=sum(a['seconds'] for a in pauses);elapsed=live['recorded_unix']-s['started_unix']-pause_seconds
total_samples=samples+sum(a['samples'] for a in recovered)
rates=dict(completed_operation_main_updates=s['total_training_iterations'],verified_recovered_main_updates=58,
    actual_retained_main_updates=s['total_training_iterations']+58,declared_inherited_updates=s['prior_training_iterations'],
    actual_retained_samples=int(total_samples),active_collection_PPO_samples_s=total_samples/(active+sum(a['active_seconds'] for a in recovered)),
    main_worker_effective_samples_s=total_samples/(worker+sum(a['worker_seconds'] for a in recovered)),whole_chain_samples_s=total_samples/elapsed,
    excluded_pause_seconds=pause_seconds,pauses=pauses,recovery_evidence=recovered,blocks=blocks,
    scope='All completed and proven retained recovery updates across 1.5 and 2.0 stages. Capacity, smoke and diagnostic actors excluded. Active: collection+PPO; worker-effective: adds main startup/exit; whole-chain: includes R37 startup/capacity/gates minus both documented R37 pauses. No pause this check.')
write(OUT/'throughput_and_loads.json',rates)
g=read(OUT/'closing_gate_review.json');promoted=g['promotions'][-1]['matching_pair'];latest=g['latest']
write(OUT/'qualified_run_150.json',promoted);write(OUT/'closing_matching_gates.json',latest)
previous=read(OUT.parent/'adaptive_check_20261005_2046/final_liveness.json')
audit=read(OUT/'promoted_policy_audit.json')
focus=[
    'Preserve the same improving R37 full-state learner. Checkpoint199 newly qualifies running1.5 with exact200/201 full nominal/pushed/stopped/prior gates. The2.0 case now repeatedly passes actual speed, geometric flight and direction, but prior1.5 flight fails. Compare complete newer pairs; only promote2.0 when every stopped/prior case passes on that same actor. Monitor the2.5 frontier only after full qualification; no5m/s or ceiling is established.',
    'Investigate prior-teacher context provenance before any estimator or reward redesign. Promotion replaced the prior teacher with neutral-head checkpoint199 but the old R34 head table still applies to walking commands. Exact actual-history CPU replay reproduces artificial self-teacher lower target RMS0.231-0.258 on199 versus~1e-6 without conditioning. On222 truthful current states, labels change0.236-0.279RMS; the student is closer to conditioned labels. This is an identified target/fixed-point mismatch, not proof of the1.5 flight regression: the table does not affect1.5 labels. If retention remains stalled, test teacher-specific context binding, keeping old R34 conditioning for its original teacher and truthful labels for newer certified neutral-head teachers. Measure actual per-command advantages and PPO/prior gradients, then compare unchanged current full-state control and candidate at64/128 updates with every prior/head/direction/flight gate. Preserve currentactor/critic/Adam/exploration, truthful student observations and all motor caps. Do not deploy solely from CPU sensitivity or erase posture/2.0 progress.',
    'Verify55 frozen hashes, one actual worker,6144/8192 memory and exact full-state recovery. Count58 retained recovery updates separately from supervisor completed operations; exclude the679.7s historical R37 pauses. No pause occurred this check. The current6144-to8192 ramp requires about2091MiB freeGPU and180continuous healthyTRAININGseconds plus host guards. New exact10Hz CAD Jacobian motor audit at2.0 has ankle p99 cap ratios~0.88/0.84 and brief sampled crossings, not driving clipping or a ceiling. Perform actor-specific5ms clipping/torque attribution if high-speed plateau appears. Preserve rejected timing offsets, box smoothing, running-row transfer,44% exposure and teacher-clipping-equivalence decisions absent new evidence; CAPG remains an undeployed lower-priority lead.'
]
decision=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),action='Preserve improving same full-state learner; replace obsolete stalled1.5 diagnosis with running-retention and teacher-provenance audits',
    runtime_reward_physics_caps_optimizer_actor_changed=False,training_paused=False,
    run_150_historically_qualified=True,current_run_200_qualified=latest['nominal_all_pass'] and latest['pushed_all_pass'],
    velocity_ceiling_proven=False,actual_5_m_s_proven=False,next_tests=focus,
    teacher_context_candidate_deployed=False,teacher_context_causal_learning_validation=False,
    actual_PPO_advantage_gradient_capture_performed=False)
write(OUT/'decision.json',decision)
evidence=dict(recorded_utc=decision['recorded_utc'],owned_snapshot_utc=live['recorded_utc'],run=str(run),supervisor=s['supervisor_pid'],
    actual_workers=live['actual_worker_pids'],frozen_hashes_verified=55,status=s['status'],stage=s['stage'],operation=s['operation_path'],
    operation_kind=s['operation'],environment_count=s['environment_count'],next_training_environment_count=s.get('next_training_environment_count'),
    gpu=live['gpu'],host=live['system_memory'],main_updates_since_previous_owned_check=s['total_training_iterations']-previous['completed_operation_main_updates'],
    decision=decision,rates={k:v for k,v in rates.items() if k!='blocks'},qualified_1_5=promoted,latest_pair=latest,latest_head=g['latest_head'],
    audit=str(OUT/'promoted_policy_audit.json'),scope='Gate pairs bind exact saved actors. Owned state, CPU analysis time, and live partial progress are separate timestamps.')
write(OUT/'evidence.json',evidence)
# Plot exact full-population gate trends; do not interpolate or imply monotonic gains.
pairs=g['gate_pairs'][-20:];x=np.arange(len(pairs));fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
for command,color in [(1.5,'#2453a6'),(2.,'#c65318')]:
    for key,ls in [('nominal_cases','-'),('pushed_cases','--')]:
        vals=[next((c['flight']*100 for c in p[key] if abs(c['command']-command)<1e-6),np.nan) for p in pairs]
        speeds=[next((c['signed_speed'] for c in p[key] if abs(c['command']-command)<1e-6),np.nan) for p in pairs]
        axes[0].plot(x,vals,ls,color=color,marker='.',label=f'{command:.1f} m/s {key.split("_")[0]}')
        axes[1].plot(x,speeds,ls,color=color,marker='.')
axes[0].axhline(1.5,color='#333',lw=1,ls=':',label='Existing flight minimum1.5%')
axes[0].set(ylabel='Geometric flight (%)',title='Exact saved-policy independent tests: new frontier and running retention')
axes[1].set(ylabel='Signed forward speed (m/s)',xlabel='Saved checkpoint operation number')
axes[1].set_xticks(x[::2],[Path(p['checkpoint']).parent.name.split('_')[0] for p in pairs[::2]])
for ax in axes:ax.grid(alpha=.2)
axes[0].legend(ncol=3,fontsize=9);fig.tight_layout();fig.savefig(OUT/'running_retention_trends.png',dpi=145);plt.close(fig)
print(json.dumps(dict(stage=s['stage'],completed=s['total_training_iterations'],retained=s['total_training_iterations']+58,
    rates={k:rates[k] for k in ['active_collection_PPO_samples_s','main_worker_effective_samples_s','whole_chain_samples_s']},
    latest=Path(latest['checkpoint']).parent.name,cases=[(c['command'],c['signed_speed'],c['flight'],c['passed']) for c in latest['nominal_cases']],
    workers=live['actual_worker_pids'],unchanged=True),indent=2))
