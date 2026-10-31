"""Preserved paused-run provenance and throughput; no simulator or Torch."""
import datetime
import hashlib
import json
from pathlib import Path
import tomllib
import xml.etree.ElementTree as ET

DIAG = Path(__file__).resolve().parent
BASE = DIAG.parents[3]
training = BASE / 'physical_v29/training'
run = Path(json.loads((training/'ACTIVE_RUN.json').read_text())['run'])
state = json.loads((run/'state.json').read_text())
config = json.loads((run/'chain_config.json').read_text())
source = run/'source/physical_v29'
integrity = {}
for name, expected in config['sources_sha256'].items():
    if name in ('supervisor_unitree.py', 'vram_ramp.py'):
        p = run/name
    elif name.startswith('physical_assets/'):
        p = source/'assets'/name.removeprefix('physical_assets/')
    elif name.startswith('physical_v29/'):
        p = source/name.removeprefix('physical_v29/')
    else:
        p = source/'training'/name
    integrity[name] = hashlib.sha256(p.read_bytes()).hexdigest() == expected
assert all(integrity.values())
blocks = []
samples = collection = learning = worker_wall = 0
for f in sorted(run.glob('*train*/result.json')):
    result = json.loads(f.read_text())
    rows = [json.loads(x) for x in (f.parent/'progress.jsonl').read_text().splitlines()]
    count = result['num_envs']*24*result['iterations']
    c = sum(x['collect_seconds'] for x in rows)
    l = sum(x['learn_seconds'] for x in rows)
    attempt = next(a for a in state['attempts'] if Path(a['folder']) == f.parent)
    blocks.append(dict(folder=f.parent.name, updates=result['iterations'], samples=count,
                       collection_seconds=c, learning_seconds=l,
                       active_steps_per_second=count/(c+l), worker_wall_seconds=attempt['seconds'],
                       private_mb_range=[min(x['gpu']['process_private_mb'] for x in rows),max(x['gpu']['process_private_mb'] for x in rows)],
                       pre_update_kl_max=max(x['ppo']['pre_update_kl'] for x in rows),
                       action_reward_command_difference_max=max(x['behavior']['action_command_difference'] for x in rows)))
    samples += count
    collection += c
    learning += l
    worker_wall += attempt['seconds']
gates = []
for f in sorted(run.glob('*gate*/result.json')):
    result = json.loads(f.read_text())
    gates.append(dict(folder=f.parent.name, passed=result['passed'],
                      seed=result['seed'], duration_s=result['duration_s'],
                      cases=[dict(statistics=c['statistics'], failures=c['failures'], passed=c['passed']) for c in result['cases']]))
urdf = ET.parse(source/'training/assets/robot_training.urdf').getroot()
profile = json.loads((DIAG/'physics_profile/result.json').read_text())
step_ms = profile['profiled_seconds']*1000/profile['profiled_ticks']
cad_fraction = profile['sections']['cad_state']['cpu_dispatch_ms_per_control_tick']/step_ms
automation = tomllib.loads(Path(r'C:\Users\rshan\.codex\automations\humanoid-training-research-and-repair-every-four-hours\automation.toml').read_text(encoding='utf-8'))
last_result = Path(state['checkpoint']).parent/'result.json'
elapsed = last_result.stat().st_mtime-state['started_unix']
record = dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), run=str(run),
              status=state['status'], stage=state['stage'], total_updates=state['total_training_iterations'],
              automation_status=automation['status'], source_integrity=integrity,
              pause=json.loads((DIAG/'pause_completion.json').read_text()),
              training_blocks=blocks, independent_gates=gates,
              throughput=dict(total_samples=samples, collection_seconds=collection, learning_seconds=learning,
                              collection_fraction=collection/(collection+learning),
                              active_ppo_steps_per_second=samples/(collection+learning),
                              worker_effective_steps_per_second=samples/worker_wall,
                              wall_to_final_result_seconds=elapsed,
                              end_to_end_steps_per_second=samples/elapsed,
                              archived_five_foot_ppo_steps_per_second=55518,
                              archived_over_current_active_factor=55518/(samples/(collection+learning)),
                              scope='Current whole run includes maintenance smoke/startup and eight independent gates; old archived rate is active PPO on a different model.'),
              asset=dict(links=len(urdf.findall('link')),joints=len(urdf.findall('joint')),
                         collision_shapes=[c.find('geometry')[0].tag for l in urdf.findall('link') for c in l.findall('collision')],
                         visual_shapes=[v.find('geometry')[0].tag for l in urdf.findall('link') for v in l.findall('visual')]),
              profile=profile, profile_interpretation=dict(cad_fraction=cad_fraction,
                  physics_fraction=profile['sections']['physx_step']['cpu_dispatch_ms_per_control_tick']/step_ms,
                  hypothetical_ceiling_removing_all_cad_cost=1/(1-cad_fraction),
                  scope='Eight instrumented control ticks at4096; overlapping parent timings are not summed. Ceiling is not a promised production gain.'),
              reward_status='Alternating steps learned. Current command/heading robustness remains unqualified; newest256 actor has no matching independent gate yet.')
(DIAG/'evidence.json').write_text(json.dumps(record,indent=2))
print(json.dumps({k:record[k] for k in ['status','total_updates','automation_status','throughput','profile_interpretation']},indent=2))
