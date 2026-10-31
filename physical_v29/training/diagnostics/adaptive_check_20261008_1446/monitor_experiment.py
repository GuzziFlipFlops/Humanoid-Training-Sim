"""CPU-only compact status of the sequential diagnostic; no process changes."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
root = OUT / 'matched_sequence'
rows = []
for folder in sorted(root.iterdir()):
    if not folder.is_dir():
        continue
    result = folder / 'result.json'
    progress = folder / 'progress.jsonl'
    if result.exists():
        data = json.loads(result.read_text())
        rows.append(dict(operation=folder.name, completed=data.get('completed'),
                         elapsed=data.get('elapsed_seconds')))
    elif progress.exists():
        lines = progress.read_text().splitlines()
        try:
            values = [json.loads(v) for v in lines]
        except json.JSONDecodeError:
            values = [json.loads(v) for v in lines[:-1]]
        if values:
            last = values[-1]
            audit = last.get('upper_refinement', {}).get('experimental_sequence_mean', {})
            rows.append(dict(operation=folder.name, observed_updates=len(values),
                free_gpu_MiB=last['gpu']['free_mb'], available_host_MiB=last['gpu']['ram_available_mb'],
                torch_allocated_MiB=last['gpu']['torch_allocated_mb'],
                post_warmup_allocation_span_MiB=max(r['gpu']['torch_allocated_mb'] for r in values[3:])-
                    min(r['gpu']['torch_allocated_mb'] for r in values[3:]) if len(values)>3 else None,
                KL=last['ppo']['pre_update_kl'], command_difference=last['behavior']['action_command_difference'],
                sequence_batches=audit.get('batches'), new_gradient_applied=audit.get('applied')))
print(json.dumps(rows))
