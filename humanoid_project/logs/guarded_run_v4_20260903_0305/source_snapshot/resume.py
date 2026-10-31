"""Resolve only a checkpoint with a recorded passing gate and matching evaluation hash."""
import hashlib
import json
from pathlib import Path


def accepted_source(state_path, original_hashes):
    state_path=Path(state_path)
    state=json.loads(state_path.read_text())
    if state['status']!='stopped_gate' or state['stage']!='targets':
        raise RuntimeError('Expected stopped v3 targets stage')
    if state['original_hashes']!=original_hashes:
        raise RuntimeError('Original assets/policies changed since v3')
    checkpoint=Path(state['last_accepted_checkpoint'])
    prefix=checkpoint.parent.name.rsplit('_',1)[0]
    gate=json.loads((state_path.parent/(prefix+'_gate.json')).read_text())
    if gate['checkpoint']!=str(checkpoint) or not gate['walking_pass'] or not gate['stage_pass']:
        raise RuntimeError('Resume checkpoint was not approved by both gates')
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for seed in (47,83):
        result=json.loads((state_path.parent/f'{prefix}_eval_{seed}'/'result.json').read_text())
        if not result.get('completed') or result['source_sha256']!=digest:
            raise RuntimeError('Checkpoint differs from evaluated artifact')
    return checkpoint,state['action_scale'],gate
