"""CPU-only exact-runtime maintenance evidence; never starts a simulator."""
import datetime
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[4]
TRAINING = BASE / 'physical_v29/training'
OUT = Path(__file__).resolve().parent
PRIOR = TRAINING / 'runs/locomotion_20261004_r28_heading_alignment_cache_fixed'
RUN = TRAINING / 'runs/locomotion_20261004_r29_host_capacity_checked'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


state = read(RUN / 'state.json')
files = {}
for p in (RUN / 'source/physical_v29').rglob('*'):
    rel = p.relative_to(RUN / 'source/physical_v29')
    if p.is_file() and p.suffix != '.pyc' and rel.parts[:3] != ('training', 'assets', 'usd'):
        files[rel.as_posix()] = p.read_bytes() == (PRIOR / 'source/physical_v29' / rel).read_bytes()
assert len(files) == 41 and all(files.values())
operations = []
for p in sorted(RUN.iterdir()):
    if not p.is_dir() or not p.name[:3].isdigit():
        continue
    item = dict(name=p.name, result=read(p / 'result.json') if (p / 'result.json').exists() else None,
                arguments=read(p / 'command.json')['argv'], actor_transfer=False)
    item['actor_transfer'] = '--actor_warm_start' in item['arguments']
    assert not item['actor_transfer'], 'Maintenance must preserve the full checkpoint'
    if (p / 'progress.jsonl').exists():
        rows = [json.loads(x) for x in (p / 'progress.jsonl').read_text().splitlines()]
        item['progress'] = dict(updates=len(rows), pre_update_kl_max=max(r['ppo']['pre_update_kl'] for r in rows),
            private_mb=[r['gpu']['process_private_mb'] for r in rows],
            cuda_mb=[r['gpu']['torch_allocated_mb'] for r in rows],
            command_difference_max=max(r['behavior']['action_command_difference'] for r in rows))
    operations.append(item)
value = dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), active_state=state,
             authored_runtime_files_identical=files, runtime_files=41,
             initialization_full_checkpoint_sha256=hashlib.sha256(Path(read(TRAINING / 'ACTIVE_RUN.json')['initial_full_checkpoint']).read_bytes()).hexdigest(),
             prior_standing=read(PRIOR / 'standing_gate_completion_after_drain.json'),
             operations=operations,
             benchmark_scope='Identical-runtime R28 measured capacity reclassified by full-budget/host-RAM/VRAM guards',
             scope='Full actor/critic/Adam continuation; smoke execution is not locomotion qualification')
(OUT / 'r29_execution_handoff.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
print(json.dumps(dict(recorded_utc=value['recorded_utc'], state=state['status'], stage=state['stage'],
    operation_path=state.get('operation_path'), next_training_environments=state.get('next_training_environment_count'),
    runtime_files_identical=41, full_training_iterations=state['total_training_iterations'],
    operations=[dict(name=i['name'], completed=i['result'].get('completed') if i['result'] else False,
                     progress=i.get('progress')) for i in operations]), indent=2))
