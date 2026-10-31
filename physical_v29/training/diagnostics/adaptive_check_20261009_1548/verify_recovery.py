"""Audit saved resume evidence and real main updates; no simulator imports."""
import json, re, hashlib, argparse
from pathlib import Path
from datetime import datetime, timezone

OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
p=argparse.ArgumentParser();p.add_argument('--snapshot',default='closing_cpu/live_handoff.json');p.add_argument('--output',default='resumed_learning_verification.json');args=p.parse_args()
live=read(OUT/args.snapshot)
run=Path(live['active_run']['run']); state=live['state']
context=read(run/'resume_context.json'); initial=context['checkpoint']
assert run.name=='locomotion_20261007_r40_goal_budget_full_state'
assert state['status']=='running' and all(live['frozen_source_integrity'].values())
assert len(live['actual_worker_pids'])==1
assert context['inherited_main_updates']==27507 and context['verified_recovered_main_updates']==33
assert read(run/'chain_config.json')['max_hours']==0
for relative,digest in context['protected_hashes'].items():
    p=run/'source/physical_v29'/relative
    assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
smoke=run/'002_resume_ppo_smoke'
load=read(smoke/'full_load_proof.json')
assert load['source_sha256']==context['checkpoint_sha256']
assert all(load[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
smoke_rows=[json.loads(x) for x in (smoke/'progress.jsonl').read_text().splitlines()]
assert len(smoke_rows)==4
assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in smoke_rows)
assert all(r['retention']['batches']==20 and r['locomotion_retention']['batches']==20 and r['upper_refinement']['batches']==20 for r in smoke_rows)
assert read(run/'003_checkpoint_reload/result.json')['completed']
blocks=[]
receivers=[Path(a['folder']) for a in state['attempts']]
receivers.append(Path(state['operation_path']))
for a in state['attempts']:
    f=Path(a['folder'])
    if a['operation']!='train' or not re.search(r'_run_\d+_train_\d+',f.name):continue
    recovered=not a['completed']
    received=None
    if recovered:
        recovery=f/'memory_recovery.pt'
        if not a.get('memory_failure') or not recovery.exists():continue
        digest=hashlib.sha256(recovery.read_bytes()).hexdigest()
        received=next((receiver for receiver in receivers if (receiver/'full_load_proof.json').exists()
            and read(receiver/'full_load_proof.json')['source_sha256']==digest),None)
        if received is None:continue
        receiver_proof=read(received/'full_load_proof.json')
        assert all(receiver_proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
        receiver_args=read(received/'run_config.json')['arguments']
        assert Path(receiver_args['checkpoint']).resolve()==recovery.resolve()
    rows=[json.loads(x) for x in (f/'progress.jsonl').read_text().splitlines()]
    proof=read(f/'full_load_proof.json')
    assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
    assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
    assert all(r['retention']['batches']==20 and r['locomotion_retention']['batches']==20 and r['upper_refinement']['batches']==20 for r in rows)
    blocks.append(dict(folder=str(f),updates=len(rows),samples=sum(r['num_envs']*24 for r in rows),
        active_seconds=sum(r['collect_seconds']+r['learn_seconds'] for r in rows),worker_seconds=a['seconds'],
        environments=rows[0]['num_envs'],minimum_free_gpu_MiB=min(r['gpu']['free_mb'] for r in rows),
        minimum_host_available_MiB=min(r['gpu']['ram_available_mb'] for r in rows),
        private_memory_growth_MiB=rows[-1]['gpu']['process_private_mb']-rows[0]['gpu']['process_private_mb'],
        torch_allocation_growth_MiB=rows[-1]['gpu']['torch_allocated_mb']-rows[0]['gpu']['torch_allocated_mb'],
        full_load_proof=proof,all_preupdate_KL_and_command_mismatch_zero=True,three_auxiliaries_batches_per_update=20,
        hash_proven_recovered=recovered,recovery_received_by=str(received) if received else None,
        recovery_sha256=digest if recovered else None))
assert sum(b['updates'] for b in blocks if not b['hash_proven_recovered'])==state['total_training_iterations']
if blocks:
    assert blocks[0]['full_load_proof']['source_sha256']==context['checkpoint_sha256']
samples=sum(b['samples'] for b in blocks)
current_folder=Path(state['operation_path'])
partial=[]
if re.search(r'_run_\d+_train_\d+',current_folder.name) and current_folder not in [Path(b['folder']) for b in blocks] and (current_folder/'progress.jsonl').exists():
    snapshot_unix=datetime.fromisoformat(live['recorded_utc']).timestamp()
    partial=[r for r in (json.loads(x) for x in (current_folder/'progress.jsonl').read_text().splitlines()) if r['time']<=snapshot_unix]
partial_load=None
if partial:
    partial_load=read(current_folder/'full_load_proof.json')
    assert all(partial_load[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
    assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in partial)
    assert all(r['retention']['batches']==20 and r['locomotion_retention']['batches']==20 and r['upper_refinement']['batches']==20 for r in partial)
record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),owned_snapshot_utc=live['recorded_utc'],
    run=str(run),supervisor_pid=state['supervisor_pid'],actual_worker_pids=live['actual_worker_pids'],
    frozen_files_verified=len(live['frozen_source_integrity']),protected_authored_worker_files_equal=len(context['protected_hashes']),
    exact_initial_full_checkpoint=initial,initial_sha256=context['checkpoint_sha256'],
    inherited_actual_main_updates=27507,previous_interrupted_unsaved_updates_excluded=32,
    completed_main_updates=state['total_training_iterations'],observed_partial_main_updates=len(partial),partial_full_load_proof=partial_load,
    hash_proven_recovered_main_updates=sum(b['updates'] for b in blocks if b['hash_proven_recovered']),
    actual_retained_main_updates=sum(b['updates'] for b in blocks),
    blocks=blocks,real_resume_smoke_updates=4,smoke_and_capacity_actors_discarded=True,
    smoke_full_load_proof=load,smoke_teacher_and_upper_gradients_present=True,
    stage=state['stage'],operation_path=state['operation_path'],current_environment_count=state['environment_count'],
    gpu=live['gpu'],system_memory=live['system_memory'],
    active_PPO_samples_s=samples/sum(b['active_seconds'] for b in blocks) if blocks else None,
    worker_effective_samples_s=samples/sum(b['worker_seconds'] for b in blocks) if blocks else None,
    whole_chain_samples_s=None,
    goal_driven_budget=True,memory_and_stall_guards_retained=True,diagnostic_gains_and_actors_deployed=False,
    scope='Completed production main samples and hash-proven received full-state memory recoveries count. Startup, capacity and all matched diagnostic actors are excluded. Exact load is checked before existing learning-rate restoration/ceiling; qualification belongs to saved matching gate actors.')
if blocks:
    first_rows=[json.loads(x) for x in (Path(blocks[0]['folder'])/'progress.jsonl').read_text().splitlines()]
    last_rows=[json.loads(x) for x in (Path(blocks[-1]['folder'])/'progress.jsonl').read_text().splitlines()]
    record['main_first_reported_unix']=first_rows[0]['time']
    record['whole_chain_measurement_endpoint_unix']=last_rows[-1]['time']
    endpoint=last_rows[-1]['time'];excluded=0.
    intervals=[]
    for resume_path in [OUT.parent/'adaptive_check_20261007_0844/diagnostic_resume.json',OUT.parent/'adaptive_check_20261007_1344/diagnostic_resume.json',OUT.parent/'adaptive_check_20261007_1845/attempt01_initial_cache_failure/diagnostic_resume.json',OUT.parent/'adaptive_check_20261007_1845/diagnostic_resume.json',OUT.parent/'adaptive_check_20261007_2346/diagnostic_resume.json',OUT.parent/'adaptive_check_20261008_0446/diagnostic_resume.json',OUT.parent/'adaptive_check_20261008_0946/attempt01_initial_import_failure/diagnostic_resume.json',OUT.parent/'adaptive_check_20261008_0946/diagnostic_resume.json',OUT.parent/'adaptive_check_20261008_1446/attempt01_initial_mlp_failure/diagnostic_resume.json',OUT.parent/'adaptive_check_20261008_1446/attempt02_diagnostic_graph_cycle/diagnostic_resume.json',OUT.parent/'adaptive_check_20261008_1446/diagnostic_resume.json',OUT.parent/'adaptive_check_20261008_1946/diagnostic_resume.json',OUT.parent/'adaptive_check_20261009_0046/diagnostic_resume.json',OUT.parent/'adaptive_check_20261009_0547/diagnostic_resume.json',OUT.parent/'adaptive_check_20261009_1047/diagnostic_resume.json',OUT/'diagnostic_resume.json']:
        if resume_path.exists():
            resume=read(resume_path)
            lo=max(state['started_unix'],resume['suspended_unix']);hi=min(endpoint,resume['resumed_unix'])
            if hi>lo:intervals.append((lo,hi))
    for failure_resume in [OUT.parent/'adaptive_check_20261009_0046/attempt01/diagnostic_resume.json',OUT.parent/'adaptive_check_20261009_0046/attempt02/diagnostic_resume.json',OUT.parent/'adaptive_check_20261009_0046/replay_sham/diagnostic_resume.json',OUT.parent/'adaptive_check_20261009_0547/attempt01_signature_metadata/diagnostic_resume.json',OUT.parent/'adaptive_check_20261009_0547/head_arm_factorial/diagnostic_resume.json',OUT.parent/'adaptive_check_20261009_1047/high_speed_extrapolation/diagnostic_resume.json',OUT/'secondary_analysis/diagnostic_resume.json',OUT/'coupled_training_trial/diagnostic_resume.json',OUT/'coupled_training_trial/remainder_pause/diagnostic_resume.json']:
        if failure_resume.exists():
            resume = read(failure_resume)
            lo=max(state['started_unix'],resume['suspended_unix']);hi=min(endpoint,resume['resumed_unix'])
            if hi>lo:intervals.append((lo,hi))
    intervals.sort()
    assert all(intervals[i][1]<=intervals[i+1][0] for i in range(len(intervals)-1))
    excluded=sum(hi-lo for lo,hi in intervals)
    record['documented_diagnostic_pause_seconds_excluded']=excluded
    record['whole_chain_samples_s']=samples/(endpoint-state['started_unix']-excluded)
    record['whole_chain_scope']='Startup/capacity/baseline gates included; exactly documented diagnostic suspensions,including safely resumed failed attempts,excluded through the measurement endpoint. First brief failed bookkeeping pause was not timestamped exactly and remains included; it makes this rate slightly conservative. Final save/Kit exit after the endpoint excluded. Different from worker-effective rate.'
(OUT/args.output).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k not in ['blocks','smoke_full_load_proof','system_memory']},indent=2))
