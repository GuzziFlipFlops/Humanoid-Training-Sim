"""CPU-only saved-gate and throughput audit; never imports the simulator."""
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parents[1]
pointer = json.loads((TRAINING / 'ACTIVE_RUN.json').read_text(encoding='utf-8-sig'))
run = Path(pointer['run']) if 'run' in pointer else Path(pointer['output'])
state = json.loads((run / 'state.json').read_text(encoding='utf-8-sig'))
gates = []
for path in sorted(run.glob('*run_150_gate*/result.json')):
    result = json.loads(path.read_text(encoding='utf-8-sig'))
    config = json.loads((path.parent / 'run_config.json').read_text(encoding='utf-8-sig'))
    row = {'folder': path.parent.name, 'checkpoint': config['arguments']['checkpoint'],
           'passed': result.get('passed'), 'cases': []}
    for case in result.get('cases', []):
        s = case['statistics']
        row['cases'].append({
            'command': s['command_m_s'], 'trials': s['trials'],
            'passed': case['passed'], 'failed': [k for k,v in case['checks'].items() if not v],
            'speed': s.get('reset_frame_forward_speed_m_s'),
            'lateral': s.get('reset_frame_lateral_speed_m_s'),
            'flight_percent': 100*s.get('geometric_flight_fraction', 0),
            'survival': s['survival'], 'stance_asymmetry': s.get('stance_asymmetry'),
            'steps_left': s.get('steps_left'), 'steps_right': s.get('steps_right'),
            'cadence': s.get('cycle_hz'), 'final_heading': s.get('final_heading_abs_mean_deg')})
    gates.append(row)
blocks = []
for path in sorted(run.glob('*run_150_train*/result.json')):
    r = json.loads(path.read_text(encoding='utf-8-sig'))
    progress_path = path.parent / 'progress.jsonl'
    records = [json.loads(x) for x in progress_path.read_text(encoding='utf-8-sig').splitlines() if x.strip()] if progress_path.exists() else []
    active_seconds = sum(x.get('collect_seconds',0)+x.get('learn_seconds',0) for x in records)
    samples = sum(x.get('num_envs',r.get('num_envs',0))*24 for x in records)
    attempt = next((a for a in state.get('attempts',[]) if Path(a['folder'])==path.parent),{})
    blocks.append({'folder':path.parent.name, 'completed':r.get('completed'),
        'updates': len(records), 'environments':r.get('num_envs'),
        'active_samples_per_second':samples/active_seconds if active_seconds else None,
        'worker_effective_samples_per_second':samples/attempt['seconds'] if attempt.get('seconds') else None,
        'minimum_free_gpu_mb':min((x['gpu']['free_mb'] for x in records),default=None),
        'minimum_host_mb':min((x['gpu'].get('ram_available_mb',0) for x in records),default=None),
        'max_pre_update_kl':max((x.get('ppo',{}).get('pre_update_kl',0) for x in records),default=None),
        'max_command_difference':max((x.get('behavior',{}).get('action_command_difference',0) for x in records),default=None),
        'memory_failure':attempt.get('memory_failure'), 'failure':attempt.get('failure')})
snapshot = {'observed_utc':datetime.now(timezone.utc).isoformat(),'run':str(run),
    'state':{k:state.get(k) for k in ('status','stage','checkpoint','total_training_iterations','full_training_iterations','operation','operation_path','environment_count','next_training_environment_count','vram_ramp','memory_fallback','error')},
    'gates':gates, 'blocks':blocks}
snapshot['latest_checkpoint_sha256'] = hashlib.file_digest(Path(state['checkpoint']).open('rb'), 'sha256').hexdigest()
owned_path = TRAINING / 'diagnostics/walk_regression_20261003/live_handoff.json'
if owned_path.exists():
    owned = json.loads(owned_path.read_text(encoding='utf-8-sig'))
    (HERE/'owned_observation.json').write_text(json.dumps(owned,indent=2),encoding='utf-8')
(HERE/'state_snapshot.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
(HERE/'evidence.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
print(json.dumps({'run':str(run),'state':snapshot['state'],
    'gate_trends':[{'folder':x['folder'],'checkpoint':Path(x['checkpoint']).parent.name,'passed':x['passed'],
        'running':next((c for c in x['cases'] if c['command']==1.5),None),
        'failed_priors':[c for c in x['cases'] if c['command']<1.5 and not c['passed']]} for x in gates[-2:]],
    'blocks':blocks[-2:]},indent=2))
