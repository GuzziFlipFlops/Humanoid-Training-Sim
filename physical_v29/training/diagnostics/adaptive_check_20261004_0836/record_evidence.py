"""CPU-only check evidence; never imports a simulator or changes a run."""
import datetime
import hashlib
import json
import statistics
from pathlib import Path
import psutil
import subprocess

ROOT=Path(__file__).resolve().parents[4]
HERE=ROOT/'physical_v29/training'
OUT=Path(__file__).parent
R30=HERE/'runs/locomotion_20261004_r30_exact_graph_execution'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def frozen(run):
    manifest=read(run/'chain_config.json')['sources_sha256']
    results=[]
    for name,digest in manifest.items():
        if name in ['supervisor_unitree.py','vram_ramp.py']: p=run/name
        elif name.startswith('physical_assets/'): p=run/'source/physical_v29/assets'/name.split('/',1)[1]
        elif name.startswith('physical_v29/'): p=run/'source/physical_v29'/name.split('/',1)[1]
        else:p=run/'source/physical_v29/training'/name
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        results.append(dict(name=name,sha256=actual,matches=actual==digest))
    assert all(r['matches'] for r in results)
    return results

gates=[]
for path in sorted(R30.glob('*_gate_*/result.json')):
    report=read(path)
    gates.append(dict(folder=path.parent.name,passed=report['passed'],cases=[dict(
        command=c['statistics']['command_m_s'],passed=c['passed'],failures=c['failures'],
        signed_forward_m_s=c['statistics']['reset_frame_forward_speed_m_s'],
        lateral_m_s=c['statistics']['reset_frame_lateral_speed_m_s'],
        final_heading_deg=c['statistics']['final_heading_abs_mean_deg'],survival=c['statistics']['survival'],
        steps_left=c['statistics']['steps_left'],steps_right=c['statistics']['steps_right'],
        slip_m_s=c['statistics']['slip_m_s'],hop_fraction=c['statistics']['hop_landing_fraction'],
        same_foot_fraction=c['statistics']['same_foot_fraction']) for c in report['cases']]))
state=read(R30/'state.json')
blocks=[]
for attempt in state['attempts']:
    if attempt['operation']!='train' or '_train_' not in Path(attempt['folder']).name:continue
    p=Path(attempt['folder']);progress=p/'progress.jsonl'
    if not progress.exists():continue
    lines=[json.loads(line) for line in progress.read_text().splitlines()]
    warm=lines[1:] or lines
    updates=read(p/'result.json')['iterations'] if (p/'result.json').exists() else len(lines)
    n=attempt['environments'];samples=n*24*updates
    active=sum(x['collect_seconds']+x['learn_seconds'] for x in lines)
    blocks.append(dict(folder=p.name,environments=n,updates=updates,samples=samples,
        active_ppo_samples_s=n*24/statistics.mean(x['collect_seconds']+x['learn_seconds'] for x in warm),
        worker_effective_samples_s=samples/attempt['seconds'],wall_seconds=attempt['seconds'],
        minimum_free_vram_mb=min(x['gpu']['free_mb'] for x in lines),
        minimum_host_available_mb=min(x['gpu']['ram_available_mb'] for x in lines),
        first_private_mb=lines[0]['gpu']['process_private_mb'],last_private_mb=lines[-1]['gpu']['process_private_mb'],
        first_torch_reserved_mb=lines[0]['gpu']['torch_reserved_mb'],last_torch_reserved_mb=lines[-1]['gpu']['torch_reserved_mb']))
pointer=read(HERE/'ACTIVE_RUN.json');run=Path(pointer['run']);live=read(run/'state.json')
progress_path=Path(live.get('operation_path',run))/'progress.jsonl'
progress_rows=[json.loads(line) for line in progress_path.read_text().splitlines()] if progress_path.exists() else []
latest_progress=progress_rows[-1] if progress_rows else None
completed_main=[a for a in live['attempts'] if a['operation']=='train' and '_train_' in Path(a['folder']).name and a['completed']]
last_main=None
if completed_main:
    attempt=completed_main[-1];folder=Path(attempt['folder'])
    rows=[json.loads(line) for line in (folder/'progress.jsonl').read_text().splitlines()]
    warm=rows[1:] or rows
    result=read(folder/'result.json');n=attempt['environments']
    last_main=dict(folder=str(folder),iterations=result['iterations'],environments=n,checkpoint=result['checkpoint'],
        active_PPO_samples_s=n*24/statistics.mean(x['collect_seconds']+x['learn_seconds'] for x in warm),
        worker_effective_samples_s=n*24*result['iterations']/attempt['seconds'],
        minimum_free_vram_mb=min(x['gpu']['free_mb'] for x in rows),
        minimum_host_available_mb=min(x['gpu']['ram_available_mb'] for x in rows),
        first_private_mb=rows[0]['gpu']['process_private_mb'],last_private_mb=rows[-1]['gpu']['process_private_mb'],
        pre_update_KL_max=max(x['ppo']['pre_update_kl'] for x in rows),
        action_command_difference_max=max(x['behavior']['action_command_difference'] for x in rows),
        last_progress=rows[-1])
new_gates=[]
for path in sorted(run.glob('*_gate_*/result.json')):
    r=read(path);new_gates.append(dict(folder=path.parent.name,passed=r['passed'],cases=r['cases']))
live_training_rate=None
if progress_rows and live.get('operation')=='train':
    warm=progress_rows[1:] or progress_rows
    live_training_rate=live['environment_count']*24/statistics.mean(x['collect_seconds']+x['learn_seconds'] for x in warm)
owned=[]
for p in psutil.process_iter(['name','cmdline']):
    args=p.info['cmdline'] or []
    if p.info['name']=='python.exe' and any(Path(a).name in ['worker_unitree.py','supervisor_unitree.py'] for a in args) and '--output' in args:
        target=Path(args[args.index('--output')+1]).resolve()
        if target==run.resolve() or target.is_relative_to(run.resolve()):
            owned.append(dict(pid=p.pid,parent_pid=p.ppid(),creation_time=p.create_time(),arguments=args))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
end=read(OUT/'r30_drain.json')['recorded_utc'] if (OUT/'r30_drain.json').exists() else datetime.datetime.now(datetime.timezone.utc).isoformat()
elapsed=datetime.datetime.fromisoformat(end).timestamp()-state['started_unix']
evidence=dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    prior_run=str(R30),prior_state_summary={k:state.get(k) for k in ['status','stage','checkpoint','total_training_iterations','prior_training_iterations','promotions']},
    prior_frozen=frozen(R30),prior_gates=gates,prior_training_blocks=blocks,
    prior_total_main_samples=sum(b['samples'] for b in blocks),prior_observed_elapsed_s=elapsed,
    prior_full_chain_main_samples_s=sum(b['samples'] for b in blocks)/elapsed,
    active_pointer=pointer,live_state={k:live.get(k) for k in ['status','stage','checkpoint','operation','operation_path','environment_count','total_training_iterations','prior_training_iterations','promotions','next_training_environment_count']},
    active_frozen=frozen(run),owned_processes=owned,gpu_smi=gpu,
    current_worker_progress=latest_progress,current_worker_completed_updates=len(progress_rows),current_active_PPO_samples_s=live_training_rate,
    last_completed_main=last_main,current_gates=new_gates,
    current_checkpoint_sha256=hashlib.sha256(Path(live['checkpoint']).read_bytes()).hexdigest() if live.get('checkpoint') else None,
    host_available_mb=psutil.virtual_memory().available//2**20,
    counterfactual_tests=read(OUT/'cpu_test.json'),running_extension_tests=read(OUT/'running_extension_tests.json'),
    scope='Owned process/resource snapshots and saved independent gates; fixture tests are not locomotion qualification')
(OUT/'evidence.json').write_text(json.dumps(evidence,indent=2))
print(json.dumps(dict(active_run=run.name,live=evidence['live_state'],prior_main_samples_s=evidence['prior_full_chain_main_samples_s'],
    frozen_files_verified=len(evidence['active_frozen']),gpu_smi=gpu,host_available_mb=evidence['host_available_mb']),indent=2))
