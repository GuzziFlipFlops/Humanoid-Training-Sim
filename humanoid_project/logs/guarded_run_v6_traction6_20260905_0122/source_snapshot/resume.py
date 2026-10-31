"""Resolve the measured v5 upright checkpoint whose only failed gate was slip."""
import hashlib
import json
from pathlib import Path


def accepted_source(state_path, original_hashes):
    state_path=Path(state_path)
    state=json.loads(state_path.read_text())
    if state['status']!='stopped_gate' or state.get('stage')!='torso':
        raise RuntimeError('Expected the stopped guarded-v5 torso stage')
    if state['original_hashes']!=original_hashes:
        raise RuntimeError('Original assets/policies changed since v3')
    checkpoint=Path(state['last_accepted_checkpoint'])
    gate=state['last_gate']
    prefix=gate['evaluated_as']
    if gate['checkpoint']!=str(checkpoint) or not gate['walking_pass'] or gate['stage_pass']:
        raise RuntimeError('Expected walking retention plus a narrowly failed task gate')
    failures=[]
    for batch in gate['checks']:
        failures.extend(name for name,value in batch['stage'].items() if not value)
    if set(failures)!={'slip'}:
        raise RuntimeError(f'V5 resume is allowed only for an isolated slip miss, got {failures}')
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for seed in (47,83):
        result=json.loads((state_path.parent/f'{prefix}_eval_{seed}'/'result.json').read_text())
        if not result.get('completed') or result['source_sha256']!=digest:
            raise RuntimeError('Checkpoint differs from evaluated artifact')
    return checkpoint,state['action_scale'],gate
