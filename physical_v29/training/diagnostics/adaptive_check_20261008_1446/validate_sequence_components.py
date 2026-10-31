"""Separate exact physical/PPO/load/source/diversity validation from verdict."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
OUT=Path(__file__).resolve().parent;ROOT=OUT/'matched_sequence'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
pause=read(OUT/'diagnostic_pause.json')
from validate_physical_replay import validate
parity=validate();details=parity['array_details']
smoke=ROOT/'candidate_smoke';proof=read(smoke/'full_load_proof.json')
assert all(proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
rows=[json.loads(v) for v in (smoke/'progress.jsonl').read_text().splitlines()]
assert len(rows)==8 and all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in rows)
assert read(ROOT/'candidate_reload/result.json')['completed']
assert any(r['upper_refinement']['experimental_sequence_mean']['batches']>0 for r in rows)
memory=ROOT/'candidate_memory_smoke'
memory_proof=read(memory/'full_load_proof.json')
assert all(memory_proof[k] for k in ['actor_equal','critic_equal','optimizer_equal','exploration_equal'])
memory_rows=[json.loads(v) for v in (memory/'progress.jsonl').read_text().splitlines()]
assert len(memory_rows)==16 and read(memory/'result.json')['completed']
assert all(r['ppo']['pre_update_kl']==0 and r['behavior']['action_command_difference']==0 for r in memory_rows)
allocation_span=max(r['gpu']['torch_allocated_mb'] for r in memory_rows[3:])-min(r['gpu']['torch_allocated_mb'] for r in memory_rows[3:])
assert allocation_span<64
copy=read(OUT/'protected_source_copy.json');original=Path(copy['source']);copied=OUT/'control_source/physical_v29'
for relative,digest in copy['protected_hashes'].items():
    assert hashlib.sha256((copied/relative).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((original/relative).read_bytes()).hexdigest()==digest
for suffix in ['diverse_nominal','diverse_push']:
    a=read(ROOT/f'control_128_{suffix}/initial_diversity.json');b=read(ROOT/f'candidate_128_{suffix}/initial_diversity.json')
    assert a['initial_state_sha256']==b['initial_state_sha256']
    assert all(n>=32 for n in a['unique_initial_states_by_command'].values())
    assert a['initial_q']==b['initial_q'] and a['initial_qd']==b['initial_qd']
record=dict(passed=True,checks=['same_actor_all_nominal_cases_bitwise_equal','same_actor_all_physical_arrays_bitwise_equal',
    'eight_actual_PPO_updates_exact_full_load_zero_KL_command_mismatch','actual_complete_cycle_derivative_executed',
    'checkpoint_reload_completed','sixteen_actual_1024_environment_memory_smoke_updates',
    'protected_sources_identical','identical_independently_seeded_initial_states_between_branches'],
    physical_arrays=len(details),array_details=details,original_checkpoint=pause['checkpoint'],sha256=pause['checkpoint_sha256'],
    memory_smoke_post_warmup_Torch_allocation_span_MiB=allocation_span,
    new_gradient_experiment_not_certified_by_old_signature=True,diagnostic_actors_deployed=False)
(OUT/'experiment_validation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(passed=True,physical_arrays=len(details),protected_files=copy['files'])))
